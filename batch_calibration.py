#!/usr/bin/env python3
"""
BATCH CALIBRATION AGGREGATOR
============================
Wraps the (untouched) calibration_harness.py to score MANY model runs at once.

KEY DESIGN DECISION: runs are grouped BY MODEL. Each model is calibrated
separately (pooling only its own repeated runs, which is where the statistical
power comes from), and the models are then compared side by side. Different
models are never blended into one calibration number — that would conflate them.

WORKFLOW THIS SUPPORTS
----------------------
1. Paste the generation prompt into a fresh chat (web search OFF).
2. Save each model output to a file whose name encodes the model + run index,
   e.g.  claude_run1.json, claude_run2.json, gemini_run1.json, gemini_run2.json
   The model key is the filename with a trailing "_run<number>" stripped, so
   claude_run1.json and claude_run7.json both belong to model "claude".
3. Run:   python batch_calibration.py            # scans the current folder
   or:     python batch_calibration.py <folder> --glob "*_run*.json"

WHAT YOU GET (per model <M>)
----------------------------
- <M>_pooled.json      : pooled calibration metrics for that model
- <M>_claims.csv       : every unique predicate that model produced, how often,
                         its truth, and the mean confidence it stated — sorted so
                         repeated high-confidence FALSE claims (systematic
                         hallucinations) float to the top
- <M>_reliability.png  : pooled reliability diagram for that model
AND per individual run <R>:
- <R>_dag.png          : the claim DAG node graph (teal=true, red=false)
- <R>_calibration.png  : that run's own reliability diagram
PLUS a printed CROSS-MODEL comparison table.

It does NOT modify calibration_harness.py — it imports the verified machinery
(parse_claims, verify, calibration, report, draw) from it.
"""
import os, sys, re, csv, glob, json
# seed_harness is a single-variable superset of calibration_harness (same
# calibration/report/draw, plus the richer helpers gcd/is_prime/phi/tau/sigma/vp).
# Using it means batch scoring works for BOTH old runs and new seed-bank runs.
import seed_harness as ch

# batch mode: never pop up interactive windows (draw() checks sys.argv for this)
if "--no-show" not in sys.argv:
    sys.argv.append("--no-show")


# ───────────────────────── file discovery + model grouping ──────────────────
def find_run_files(folder, pattern):
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    return [f for f in files if not f.endswith("_verified.json")]


def model_key(path):
    """Map a run filename to its MODEL, pooling all seeds/repeats of that model:
       chat5.5_seed3_run1.json -> 'chat5.5'
       claudesonnet4.6_run.json -> 'claudesonnet4.6'
       claude_run1.json -> 'claude'
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"[_-]?seed\d+", "", stem, flags=re.IGNORECASE)   # drop seed marker
    return re.sub(r"[_-]?run[_-]?\d*$", "", stem, flags=re.IGNORECASE) or stem


# ───────────────────────── loading + verifying runs ─────────────────────────
def load_run(path):
    """Parse + verify one run file. Returns (label, claims) or (label, None)."""
    label = os.path.splitext(os.path.basename(path))[0]
    try:
        claims = ch.verify(ch.parse_claims(open(path, encoding="utf-8").read()))
    except Exception as e:
        print(f"  [skip] {label}: could not parse/verify ({e})")
        return label, None
    for c in claims:
        c["_run"] = label
    return label, claims


def sanitize(claims):
    """Keep only claims with a usable numeric confidence so calibration() is safe."""
    out = []
    for c in claims:
        try:
            float(c.get("confidence"))
        except (TypeError, ValueError):
            continue
        out.append(c)
    return out


# ───────────────────────── per-claim dedup table ────────────────────────────
def claim_table(all_claims):
    """Group claims by predicate string (whitespace-normalized) within a model."""
    groups = {}
    for c in all_claims:
        key = " ".join(str(c.get("predicate", "")).split())
        g = groups.setdefault(key, {
            "predicate": c.get("predicate", ""), "claim": c.get("claim", ""),
            "truth": c.get("truth"), "confs": [], "runs": set(), "count": 0,
        })
        g["count"] += 1
        g["runs"].add(c.get("_run", "?"))
        try:
            g["confs"].append(float(c.get("confidence")))
        except (TypeError, ValueError):
            pass
        if g["truth"] is None and c.get("truth") is not None:
            g["truth"] = c.get("truth")
    rows = []
    for g in groups.values():
        confs = g["confs"]
        rows.append({
            "predicate": g["predicate"], "claim": g["claim"], "truth": g["truth"],
            "n_occurrences": g["count"], "n_runs": len(g["runs"]),
            "mean_confidence": round(sum(confs) / len(confs), 3) if confs else None,
        })
    rows.sort(key=lambda r: (r["truth"] is not False, -r["n_occurrences"],
                             -(r["mean_confidence"] or 0)))
    return rows


def write_claims_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "truth", "n_occurrences", "n_runs", "mean_confidence", "claim", "predicate"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})


# ───────────────────────── pooled reliability plot ──────────────────────────
def draw_pooled_reliability(m, out):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not (m and any(d for d in m["diagram"])):
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], ls="--", color="#9c9a92", label="perfect calibration")
    pts = [d for d in m["diagram"] if d]
    xs = [d[0] for d in pts]; ys = [d[1] for d in pts]
    ss = [20 + 200 * d[2] / m["n"] for d in pts]
    ax.scatter(xs, ys, s=ss, color="#1D9E75", zorder=3)
    ax.plot(xs, ys, color="#1D9E75", alpha=.5)
    ax.set_xlabel("stated confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_title(f"{out}: pooled reliability  (ECE={m['ece']:.3f}, "
                 f"Brier={m['brier']:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(f"{out}_reliability.png", dpi=140)
    import matplotlib.pyplot as _plt; _plt.close(fig)
    print(f"    saved {out}_reliability.png")


# ───────────────────────── main ─────────────────────────────────────────────
def main():
    a = [x for x in sys.argv[1:] if x != "--no-show"]
    folder = next((x for x in a if not x.startswith("--")), ".")
    pattern = a[a.index("--glob") + 1] if "--glob" in a else "*run*.json"
    make_dag = "--no-dag" not in a

    files = find_run_files(folder, pattern)
    if not files:
        print(f"No run files matching '{pattern}' in '{folder}'. "
              f"Save model outputs as e.g. claude_run1.json there.")
        return

    # group files by model
    models = {}
    for f in files:
        models.setdefault(model_key(f), []).append(f)
    print(f"Found {len(files)} run file(s) across {len(models)} model(s): "
          f"{', '.join(sorted(models))}\n")

    comparison = []   # one row per model for the final table
    for mdl in sorted(models):
        print("#" * 86)
        print(f"MODEL: {mdl}   ({len(models[mdl])} run(s))")
        print("#" * 86)

        pooled = []
        for path in sorted(models[mdl]):
            label, claims = load_run(path)
            if not claims:
                continue
            claims = sanitize(claims)
            pooled.extend(claims)
            # per-run DAG node graph + that run's own reliability diagram
            if make_dag:
                mrun = ch.calibration(claims)
                try:
                    ch.draw(claims, mrun, save_prefix=label)
                except Exception as e:
                    print(f"    [dag skipped for {label}: {e}]")

        if not pooled:
            print("  no usable claims.\n")
            continue

        m = ch.report(pooled)          # harness's formatted pooled report
        if m:
            json.dump({k: v for k, v in m.items() if k != "diagram"},
                      open(f"{mdl}_pooled.json", "w"), indent=2)
            print(f"    saved {mdl}_pooled.json")
            draw_pooled_reliability(m, mdl)
            rows = claim_table(pooled)
            write_claims_csv(rows, f"{mdl}_claims.csv")
            print(f"    saved {mdl}_claims.csv  ({len(rows)} unique predicates)")
            comparison.append((mdl, len(models[mdl]), m))
        print()

    # ───── cross-model comparison ─────
    if comparison:
        print("=" * 96)
        print("CROSS-MODEL COMPARISON  (each model calibrated separately)")
        print("=" * 96)
        print(f"{'model':<24}{'runs':>5}{'claims':>7}{'err%':>7}"
              f"{'Brier':>8}{'ECE':>7}{'conf|TRUE':>11}{'conf|FALSE':>12}")
        print("-" * 96)
        for mdl, nruns, m in comparison:
            err = 100 * m["n_false"] / m["n"] if m["n"] else 0
            ct = f"{m['mean_conf_right']:.2f}" if m["mean_conf_right"] is not None else "  -"
            cf = f"{m['mean_conf_wrong']:.2f}" if m["mean_conf_wrong"] is not None else "  -"
            print(f"{mdl[:23]:<24}{nruns:>5}{m['n']:>7}{err:>6.0f}%"
                  f"{m['brier']:>8.3f}{m['ece']:>7.3f}{ct:>11}{cf:>12}")
        print("=" * 96)
        print("conf|FALSE is the danger metric: high = confidently wrong (overconfident).")


if __name__ == "__main__":
    main()
