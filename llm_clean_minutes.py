#!/usr/bin/env python3
"""
LLM-powered cleaner for FOMC minutes .txt files, with automatic model fallback.

- Reads:  minutes_text/<year>/*.txt  (configurable with --in-root)
- Writes: minutes_text_llm/<year>/*.txt  (or --inplace)
- Uses OpenAI API to smart-clean text (joins broken lines sensibly, removes page headers,
  fixes hyphenation, preserves wording).

Fallback behavior:
  * If the preferred model is unavailable / no-access / not found, the script tries a
    fallback chain until one works. The first successful model is cached for the run.

Examples (PowerShell):
  $years = 1936..1945
  python llm_clean_minutes.py --years $years --model gpt-5
  python llm_clean_minutes.py --years 1941 --model gpt-5 --inplace
  # Custom fallback chain:
  python llm_clean_minutes.py --years 1941 --model gpt-5 --fallback "gpt-5,gpt-5-mini,gpt-4.1"
"""

from __future__ import annotations
import argparse, os, re, time
from pathlib import Path
from typing import List, Optional, Tuple

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Prefer modern SDK (Responses API); fallback to legacy Chat Completions if needed
try:
    from openai import OpenAI  # pip install openai
    _HAS_NEW_SDK = True
except Exception:
    import openai  # legacy
    _HAS_NEW_SDK = False

# ---------------- Prompt ----------------
SYSTEM_PROMPT = """You clean OCR/extracted meeting minutes into readable plain text.
Follow these STRICT rules:

1) Preserve meaning, punctuation, capitalization, and numbers. Do NOT rewrite or summarize.
2) Join lines that were split by formatting (including blank-line breaks inside sentences).
   - Join ONLY when it continues the same sentence or clause.
3) Remove page artifacts:
   - page numbers like "-5-" or isolated digits on their own line,
   - date headers like "3/17/41" on their own line,
   - form feed/page break markers.
4) Fix hyphenation caused by line breaks (e.g., "eco-\\nnomy" -> "economy"). Keep real hyphens.
5) Keep paragraph breaks where there were real blank lines between paragraphs.
6) NEVER merge common function-word pairs (e.g., "of the", "in the", "to the"). If in doubt, leave a space.
7) Output ONLY the cleaned text. No commentary, no code fences.
"""

# ---------------- Light pre-sanitization (saves tokens) ----------------
DATE_LINE_RX = re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{1,4}\s*$", re.MULTILINE)
DASHED_PAGENO_RX = re.compile(r"^\s*[-–—]\s*\d+\s*[-–—]?\s*$", re.MULTILINE)
CTRL_RX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

def light_presanitize(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\f", "\n\n")
    s = CTRL_RX.sub("", s)
    s = DATE_LINE_RX.sub("", s)
    s = DASHED_PAGENO_RX.sub("", s)
    return s.strip()

def chunk_text(s: str, max_chars: int = 12000) -> List[str]:
    if len(s) <= max_chars:
        return [s]
    chunks, start = [], 0
    while start < len(s):
        end = min(len(s), start + max_chars)
        cut = s.rfind("\n\n", start, end)
        if cut == -1 or cut <= start + int(0.5 * max_chars):
            cut = end
        chunks.append(s[start:cut].strip())
        start = cut
    return chunks

# ---------------- Error categorization ----------------
class TemporaryLLMError(Exception): pass
class AccessLLMError(Exception): pass  # model not found / no access

def _classify_error_message(msg: str) -> str:
    m = msg.lower()
    if any(k in m for k in ("insufficient_quota", "rate", "timeout", "temporar", "overload", "gateway", "429", "500", "502", "503", "504")):
        return "transient"
    if any(k in m for k in ("no access", "access denied", "model_not_found", "not found", "does not exist", "unsupported model", "unknown model")):
        return "access"
    return "other"

# ---------------- OpenAI call with retry ----------------
@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(TemporaryLLMError),
)
def _call_once(client, model: str, system: str, user: str) -> str:
    try:
        if _HAS_NEW_SDK:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user",   "content": [{"type": "text", "text": user}]},
                ],
                text_format={"type": "plain_text"},
            )
            out = getattr(resp, "output_text", None)
            if not out and getattr(resp, "output", None):
                parts = resp.output[0].content if resp.output else None
                if parts and len(parts) and getattr(parts[0], "text", None):
                    out = parts[0].text
            if not out:
                raise TemporaryLLMError("Empty response")
            return out.strip()
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0,
            )
            return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        kind = _classify_error_message(str(e))
        if kind == "transient":
            raise TemporaryLLMError(str(e))
        if kind == "access":
            raise AccessLLMError(str(e))
        raise

def _try_models(client, candidates: List[str], system: str, user: str) -> Tuple[str, str]:
    """
    Try models in order until one succeeds. Returns (cleaned_text, model_used).
    Raises the last error if none succeed.
    """
    last_err: Optional[Exception] = None
    for m in candidates:
        try:
            print(f"[TRY] model={m}")
            out = _call_once(client, m, system, user)
            print(f"[USE] model={m}")
            return out, m
        except AccessLLMError as e:
            print(f"[FALLBACK] no access to {m}: {e}")
            last_err = e
            continue
        except TemporaryLLMError as e:
            print(f"[RETRYABLE] transient error on {m}: {e} (will be retried automatically)")
            last_err = e
            # tenacity already retried; fall through to next candidate
            continue
        except Exception as e:
            print(f"[WARN] non-retryable error on {m}: {e}")
            last_err = e
            continue
    assert last_err is not None
    raise last_err

# ---------------- Pipeline ----------------
def clean_with_llm(client, model_chain: List[str], raw_text: str, chunk_chars: int) -> Tuple[str, str]:
    txt = light_presanitize(raw_text)
    parts = chunk_text(txt, max_chars=chunk_chars)
    cleaned_parts: List[str] = []
    chosen_model: Optional[str] = None

    for idx, chunk in enumerate(parts, 1):
        user = f"Clean this OCR/extracted minutes text. Return only cleaned plain text.\n\nCHUNK {idx}/{len(parts)}:\n{chunk}"
        # Once we find a working model, stick with it for the rest of the file for consistency
        candidates = [chosen_model] + model_chain if chosen_model else model_chain
        # Remove Nones and dedupe preserving order
        candidates = [m for i, m in enumerate(candidates) if m and m not in candidates[:i]]

        cleaned, used = _try_models(client, candidates, SYSTEM_PROMPT, user)
        chosen_model = used
        cleaned_parts.append(cleaned.rstrip())
        time.sleep(0.3)  # be gentle

    return ("\n\n".join(cleaned_parts)).strip(), chosen_model or model_chain[0]

# ---------------- CLI ----------------
def _default_fallback_for(preferred: str) -> List[str]:
    p = (preferred or "").strip()
    if not p:
        return ["gpt-5-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4.1-mini"]
    if p.startswith("gpt-5"):
        return [p, "gpt-5-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4.1-mini"]
    # generic chain
    return [p, "gpt-4.1", "gpt-4o-mini", "gpt-4.1-mini"]

def parse_args():
    ap = argparse.ArgumentParser(description="LLM cleaner with automatic model fallback.")
    ap.add_argument("--years", nargs="+", type=int, required=True, help="Years to process, e.g., 1936 1937 ...")
    ap.add_argument("--in-root", default="minutes_text", help="Input root containing <year>/*.txt")
    ap.add_argument("--out-root", default="minutes_text_llm", help="Output root (default: minutes_text_llm)")
    ap.add_argument("--inplace", action="store_true", help="Overwrite input files instead of writing to out-root")
    ap.add_argument("--model", default="gpt-5", help="Preferred model to try first")
    ap.add_argument("--fallback", default="", help="Comma-separated fallback chain (overrides defaults).")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N files per year (0 = all)")
    ap.add_argument("--chunk-chars", type=int, default=12000, help="Max characters per chunk")
    return ap.parse_args()

def main() -> int:
    args = parse_args()
    if _HAS_NEW_SDK:
        client = OpenAI()  # reads OPENAI_API_KEY
    else:
        client = None  # legacy module uses global openai key env var

    # Build model chain
    chain = [m.strip() for m in args.fallback.split(",") if m.strip()] or _default_fallback_for(args.model)
    print(f"[CHAIN] {chain}")

    in_root = Path(args.in_root)
    out_root = in_root if args.inplace else Path(args.out_root)
    if not in_root.exists():
        print(f"[ERR] Input root not found: {in_root}")
        return 1
    out_root.mkdir(parents=True, exist_ok=True)

    total = ok = 0
    for year in args.years:
        in_year = in_root / str(year)
        if not in_year.exists():
            print(f"[WARN] Missing folder for {year}: {in_year}")
            continue

        files = sorted(p for p in in_year.rglob("*.txt") if p.is_file())
        if args.limit > 0:
            files = files[:args.limit]
        print(f"[INFO] {year}: {len(files)} files")

        used_model_overall: Optional[str] = None

        for f in files:
            total += 1
            rel = f.relative_to(in_root)
            out_path = out_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                raw = f.read_text(encoding="utf-8", errors="ignore")
                cleaned, used_model = clean_with_llm(client, chain, raw, args.chunk_chars)
                used_model_overall = used_model_overall or used_model
                out_path.write_text(cleaned, encoding="utf-8")
                ok += 1
                print(f"[OK]   {rel} -> {out_path.relative_to(out_root)} (chars {len(cleaned)}) via {used_model}")
            except Exception as e:
                print(f"[ERR]  {rel}: {e}")

        if used_model_overall:
            print(f"[INFO] {year}: first working model was {used_model_overall}")

    print(f"[DONE] cleaned={ok}/{total} -> {out_root.resolve()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
