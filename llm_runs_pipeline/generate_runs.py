#!/usr/bin/env python3
"""
generate_runs.py — automated open-model trace generation for the CSUREMM project.

Loops over (model x seed x run), calls a hosted open-model API (Groq or
OpenRouter — both OpenAI-compatible, both have free tiers), robustly extracts
the JSON claim array (fixing the malformed-output problem that produced empty
files), verifies each claim with seed_harness, and saves as
    runs/<model_tag>_seed<i>_run<r>.json
in the same schema as the hand-pasted files.

SETUP
-----
1. Get a free API key:
     Groq:       https://console.groq.com   (fast, generous free tier)
     OpenRouter: https://openrouter.ai       (many models, some :free)
2. Set it as an environment variable (do NOT hardcode it):
     PowerShell:  $env:GROQ_API_KEY = "gsk_..."
             or:  $env:OPENROUTER_API_KEY = "sk-or-..."
3. Run:
     python generate_runs.py --provider groq --seeds 0,1,2,3,4 --runs 3
     python generate_runs.py --provider openrouter --models "meta-llama/llama-3.1-8b-instruct,qwen/qwen-2.5-7b-instruct" --seeds 0,1,2,3,4 --runs 3

Requires: requests, and seed_harness.py in the same folder.
"""

import os
import re
import sys
import json
import time
import argparse
import requests

import seed_harness as sh   # reuse SEED_BANK, GEN_SYSTEM, GEN_USER, parse_claims, verify, report


# ── provider config ──────────────────────────────────────────────────────────
PROVIDERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "default_models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ],
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "default_models": [
            "meta-llama/llama-3.1-8b-instruct",
            "qwen/qwen-2.5-7b-instruct",
            "deepseek/deepseek-r1-distill-qwen-7b",
        ],
    },
}


def model_tag(model: str) -> str:
    """Filename-safe short tag distinguishing model family AND size."""
    m = model.lower()
    fam = "model"
    if "llama" in m:
        fam = "llama"
    elif "deepseek" in m:      # must come BEFORE qwen — distill names contain both
        fam = "deepseek"
    elif "qwen" in m:
        fam = "qwen"
    elif "gpt-oss" in m or "gpt_oss" in m:
        fam = "gptoss"
    
    # pull a size like 8b / 70b / 7b if present
    size = ""
    msize = re.search(r"(\d+)\s*b", m)
    if msize:
        size = msize.group(1) + "b"
    tag = f"{fam}-{size}" if size else fam
    return tag


# ── robust JSON extraction (the fix for empty/garbled files) ─────────────────
def extract_json_array(text: str):
    """Pull a JSON array of claim objects out of a model response that may be
    wrapped in prose, ```json fences, or trailing junk. Returns a list or None.

    Strategy, in order:
      1. strip code fences and try json.loads on the whole thing
      2. find the outermost [...] block and parse that
      3. as a last resort, salvage the longest prefix of objects that parses
    """
    if not text or not text.strip():
        return None

    # 1) strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    # 2) outermost bracketed array
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        block = cleaned[start:end + 1]
        try:
            obj = json.loads(block)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass

    # 3) salvage: from the first '[', parse object-by-object, keep every
    #    complete {...} object even if the array was never closed (truncation).
    if start != -1:
        tail = cleaned[start + 1:]
        objs = []
        depth = 0
        buf = ""
        in_str = False
        esc = False
        for ch in tail:
            buf += ch
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    frag = buf.strip().lstrip(",").strip()
                    try:
                        objs.append(json.loads(frag))
                    except Exception:
                        pass
                    buf = ""
        if objs:
            return objs

    return None


# ── API call ─────────────────────────────────────────────────────────────────
def call_open_model(system: str, user: str, model: str, provider_cfg: dict,
                    api_key: str, temperature: float = 0.7,
                    max_tokens: int = 16000, timeout: int = 180) -> str:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if "gpt-oss" in model.lower():
        payload["reasoning"] = {"effort": "low"}

    resp = requests.post(provider_cfg["url"], headers=headers,
                         json=payload, timeout=timeout)
    if resp.status_code != 200:
        # surface Groq's actual error message (explains 413 / model / limits)
        try:
            msg = resp.json().get("error", {}).get("message", resp.text[:300])
        except Exception:
            msg = resp.text[:300]
        raise RuntimeError(f"{resp.status_code}: {msg}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ── one generation with retries ──────────────────────────────────────────────
def generate_one(model, seed_idx, provider_cfg, api_key, n_claims=50,
                 max_retries=3):
    seed_claim = sh.SEED_BANK[seed_idx]["claim"]
    system = sh.GEN_SYSTEM
    user = sh.GEN_USER.format(n=n_claims, seed=seed_claim)

    for attempt in range(1, max_retries + 1):
        try:
            raw = call_open_model(system, user, model, provider_cfg, api_key)
        except Exception as e:
            msg = str(e)
            m = re.search(r"try again in ([\d.]+)s", msg)
            wait = float(m.group(1)) + 2 if m else 3 * attempt
            print(f"      API error (attempt {attempt}): {e}")
            print(f"      waiting {wait:.0f}s ...")
            time.sleep(wait)
            continue

        claims = extract_json_array(raw)
        if claims and len(claims) >= 5:
            if len(claims) < n_claims:
                print(f"      WARNING: got {len(claims)}/{n_claims} claims "
                      f"— likely truncated (raise max_tokens)")
            return claims, raw
        print(f"      parse failed / too few claims (attempt {attempt}, "
              f"got {len(claims) if claims else 0})")
        if raw:
            print("      RAW HEAD:", repr(raw[:400]))
        time.sleep(2)

    return None, None


# ── main loop ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=list(PROVIDERS), default="groq")
    ap.add_argument("--models", type=str, default="",
                    help="comma-separated model ids; blank = provider defaults")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--n-claims", type=int, default=50)
    ap.add_argument("--out-dir", type=str, default="runs_dag")
    ap.add_argument("--start-run", type=int, default=1,
                    help="run number to start at (so you don't overwrite run1..k)")
    args = ap.parse_args()

    provider_cfg = PROVIDERS[args.provider]
    api_key = os.environ.get(provider_cfg["key_env"])
    if not api_key:
        print(f"ERROR: set {provider_cfg['key_env']} in your environment first.")
        print(f"  PowerShell:  $env:{provider_cfg['key_env']} = \"your-key\"")
        sys.exit(1)

    models = ([m.strip() for m in args.models.split(",") if m.strip()]
              or provider_cfg["default_models"])
    tags = [model_tag(m) for m in models]
    dupes = {t for t in tags if tags.count(t) > 1}
    if dupes:
        print(f"ERROR: models map to duplicate tags {dupes} — they'd overwrite each other.")
        print("  " + "\n  ".join(f"{model_tag(m)}  <-  {m}" for m in models))
        sys.exit(1)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    os.makedirs(args.out_dir, exist_ok=True)

    total = len(models) * len(seeds) * args.runs
    done = 0
    print(f"Generating {total} traces: {len(models)} models x {len(seeds)} seeds "
          f"x {args.runs} runs  (provider: {args.provider})\n")

    
    for model in models:
        tag = model_tag(model)
        for seed_idx in seeds:
            for run in range(args.start_run, args.start_run + args.runs):
                done += 1
                fname = os.path.join(args.out_dir, f"{tag}_seed{seed_idx}_run{run}.json")
                if os.path.exists(fname):
                    print(f"[{done}/{total}] SKIP exists: {fname}")
                    continue

                print(f"[{done}/{total}] {tag} seed{seed_idx} run{run} ...")
                claims, raw = generate_one(model, seed_idx, provider_cfg,
                                           api_key, n_claims=args.n_claims)
                if not claims:
                    print(f"      FAILED after retries — skipping")
                    continue

                # verify with seed_harness (fills truth + truth_reason)
                try:
                    claims = sh.verify(claims)
                except Exception as e:
                    print(f"      verify error: {e} — saving unverified")

                # keep only the schema fields, in order (defensive)
                clean = []
                for c in claims:
                    clean.append({
                        "node_id":     c.get("node_id"),
                        "parents":     c.get("parents", []),
                        "claim":       c.get("claim", ""),
                        "predicate":   c.get("predicate", ""),
                        "confidence":  c.get("confidence"),
                        "is_terminal": c.get("is_terminal", False),
                        "truth":       c.get("truth"),
                        "truth_reason": c.get("truth_reason"),
                    })

                with open(fname, "w", encoding="utf-8") as f:
                    json.dump(clean, f, indent=2)

                nf = sum(1 for c in clean if c["truth"] is False)
                ng = sum(1 for c in clean if c["truth"] is None)
                print(f"      saved {fname}: {len(clean)} claims, "
                      f"{nf} false, {ng} uncheckable")

    print(f"\nDone. Files in {args.out_dir}/")


if __name__ == "__main__":
    main()