#!/usr/bin/env python3
"""
SEED-BANK HARNESS  —  varied-root number-theory calibration (single variable)
=============================================================================
A successor to calibration_harness.py (which it imports and does NOT modify).

What's different from the original single-seed harness:

1. VARIED ROOT PER RUN.
   Instead of always rooting at "n^3 - n divisible by 6", there is a SEED BANK
   of diverse root claims spanning subfields (divisibility, congruences, parity,
   factorization identities, gcd, primality-dependent, finite sums/products,
   divisor functions). Each run uses a DIFFERENT seed (--seed i), so across many
   runs you sample calibration over many starting points, not one.
   One seed is deliberately FALSE (Euler's n^2+n+41 prime), so you also measure
   whether confidence propagates sensibly down a doomed branch.

2. RICHER CLAIM TYPES + HELPERS, still SINGLE VARIABLE n.
   Predicates remain universal statements in one integer variable n (tested over
   n in -500..500). No multi-variable families — the goal is MANY distinct nodes,
   not few generalized ones. The extra helpers (phi, tau, sigma, vp, is_prime)
   simply let the model branch into divisor-function / primality territory.

QUANTIFIER / HYPOTHESIS CONVENTION:
   Every claim is "for all integers n, ...". Encode hypotheses as implications in
   the predicate, e.g. "if n is prime then P" -> "(not is_prime(n)) or (P)".

HELPERS available inside predicates:
   gcd(a,b)  is_prime(x)  phi(x)  tau(x)  sigma(x)  vp(x,p)
   abs pow range all any sum min max len int round math factorial
   (aliases isprime=is_prime, totient=phi kept for back-compat with old runs.)

VERIFICATION TRANSPARENCY (added):
   verify() now records WHY each node landed where it did. Every claim gets a
   c["truth_reason"] tag alongside c["truth"]:
     holds | counterexample | no-checkable-n | timeout |
     build:<Err> | runtime:<Err> | forbidden
   report() prints a one-line histogram of these for the uncheckable nodes, so a
   wall of gray is diagnosable at a glance instead of by hand.

USAGE
-----
  python seed_harness.py --list-seeds
       -> show the seed bank with indices
  python seed_harness.py --print-prompt --seed 3
       -> print the generation prompt rooted at seed #3 (paste into a FRESH chat)
  python seed_harness.py --from-json run.json
       -> verify + calibration + plots (reuses the original harness report/draw)
  python seed_harness.py --selftest
       -> run the pipeline on baked-in mock data (no API key)
  python seed_harness.py --generate --seed 3 --model claude-...   (needs API key)

Verified runs are scored by batch_calibration.py exactly as before.
"""
import os, sys, re, json, math
from collections import Counter
import multiprocessing as mp
import calibration_harness as ch   # reuse trusted machinery, do not modify it

parse_claims = ch.parse_claims
calibration  = ch.calibration
draw         = ch.draw
call_model   = ch.call_model

N_RANGE = list(range(-500, 501))
_HELPER_CAP = 10**9      # phi/tau/sigma refuse inputs bigger than this (avoid hangs)
_FACT_CAP   = 2000
GUARD_RE = re.compile(r"^\s*(n\s*(?:>=|>|<=|<)\s*-?\d+)\s+and\s+(.+)$", re.S)



# ───────────────────────── number-theory helpers ────────────────────────────

def gcd(a, b):
    return math.gcd(int(a), int(b))

def is_prime(x):
    """Deterministic Miller-Rabin (correct for all x < 3.3e24)."""
    n = int(x)
    if n < 2: return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2; s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        y = pow(a, d, n)
        if y in (1, n - 1):
            continue
        for _ in range(s - 1):
            y = y * y % n
            if y == n - 1:
                break
        else:
            return False
    return True

def _cap(x):
    if abs(int(x)) > _HELPER_CAP:
        raise ValueError("argument too large to factor — claim left uncheckable")

def phi(x):
    k = abs(int(x)); _cap(k)
    if k == 0: return 0
    res, nn, p = k, k, 2
    while p * p <= nn:
        if nn % p == 0:
            while nn % p == 0: nn //= p
            res -= res // p
        p += 1
    if nn > 1: res -= res // nn
    return res

def tau(x):
    k = abs(int(x)); _cap(k)
    if k == 0: return 0
    cnt, d = 0, 1
    while d * d <= k:
        if k % d == 0:
            cnt += 1 if d * d == k else 2
        d += 1
    return cnt

def sigma(x):
    k = abs(int(x)); _cap(k)
    if k == 0: return 0
    tot, d = 0, 1
    while d * d <= k:
        if k % d == 0:
            tot += d
            if d * d != k: tot += k // d
        d += 1
    return tot

def vp(x, p):
    """p-adic valuation: exponent of prime p in x (v_p(0) -> large sentinel)."""
    x = int(x); p = int(p)
    if p < 2: raise ValueError("vp requires base >= 2")
    if x == 0: return _FACT_CAP
    x = abs(x); e = 0
    while x % p == 0:
        x //= p; e += 1
    return e

def factorial(k):
    k = int(k)
    if k < 0 or k > _FACT_CAP:
        raise ValueError("factorial out of supported range")
    return math.factorial(k)

def lcm(a, b):
    a, b = int(a), int(b)
    return 0 if (a == 0 or b == 0) else abs(a * b) // math.gcd(a, b)
def is_square(x):
    x = int(x)
    return x >= 0 and math.isqrt(x)**2 == x
def is_cube(x):
    x = int(x); r = round(abs(x) ** (1/3))
    return any((s*r**3 == x) for s in (1, -1) for r in (r-1, r, r+1))
def divisors(x):
    x = abs(int(x))
    if x == 0: return []
    return [d for d in range(1, x+1) if x % d == 0]  # cap x first if slow

SAFE_GLOBALS = {
    "__builtins__": {}, "abs": abs, "pow": pow, "range": range,
    "all": all, "any": any, "sum": sum, "min": min, "max": max, "len": len,
    "set": set, "list": list, "map": map, "int": int, "round": round, "math": math,
    "gcd": gcd, "is_prime": is_prime, "isprime": is_prime,
    "phi": phi, "totient": phi, "tau": tau, "sigma": sigma, "vp": vp,
    "lcm": lcm, "is_square": is_square, "is_cube": is_cube, "divisors": divisors,
    "factorial": factorial,
}
FORBIDDEN = re.compile(r"(__|import|open|eval|exec|os\.|sys\.|subprocess|lambda)")

# ───────────────────────── single-variable verification ─────────────────────
def normalize_predicate(pred):
    if not isinstance(pred, str):
        return pred
    p = pred.replace("^", "**")

    # strip a leading "lambda n:" wrapper some models emit. Only a LEADING
    # wrapper is removed, so an embedded lambda still trips FORBIDDEN below.
    p = re.sub(r"^\s*lambda\s+\w+\s*:\s*", "", p)

    # word-operators models use instead of Python:
    p = re.sub(r"(?<![\w.])prod\s*\(", "math.prod(", p)          # prod(...) -> math.prod(...)
    p = re.sub(r"([\w.]+\([^()]*\)|[\w.]+)\s+divides\s+(\([^()]*\)|[\w.]+)",
               r"(\2) % (\1) == 0", p)                           # "X divides Y" -> "(Y) % (X) == 0"
    p = re.sub(r"\bimplies\b", "==>", p)                         # "A implies B" -> handled by ==> below

    p = re.sub(r"([A-Za-z)\]])(\d+)", r"\1**\2", p)             # bare exponent: n2 -> n**2
    p = re.sub(r"(\d)\s*\(", r"\1*(", p)                        # 2(n+1) -> 2*(n+1)
    p = re.sub(r"\)\s*\(", r")*(", p)                           # (a)(b) -> (a)*(b)
    p = re.sub(r"(?<![A-Za-z0-9_])([a-z])\s*\(", r"\1*(", p)    # n(...) -> n*(...)
    p = re.sub(
        r"(?<![\w).])([A-Za-z_]\w*(?:\*\*\w+)?)\s*-\s*(\w+(?:\*\*\w+)?)\s*%\s*(\w+)",
        r"(\1 - \2) % \3", p,
    )
    if "==>" not in p:
        m = GUARD_RE.match(p)
        if m:
            p = m.group(1) + " ==> " + m.group(2)

    # logical implication (from "==>" or converted "implies"): "A ==> B" -> "(not (A)) or (B)"
    if "==>" in p:
        a, _, b = p.partition("==>")
        p = f"(not ({a.strip()})) or ({b.strip()})"
    return p

def check_predicate_v(pred, dom=N_RANGE):
    """Verify a predicate and report WHY.

    Returns (label, reason):
      label  in {True, False, None}
      reason in {holds, counterexample, no-checkable-n, forbidden,
                 build:<ErrType>, runtime:<ErrType>}

    Semantics are identical to the original check_predicate: a ZeroDivisionError
    or ValueError at a particular n SKIPS that n (helper refused a too-large
    input, or division by zero at an edge); any other in-loop error makes the
    whole predicate uncheckable. A predicate that is never successfully evaluated
    at any n is uncheckable (no-checkable-n)."""
    if not isinstance(pred, str):
        return None, "forbidden"
    pred = normalize_predicate(pred)
    if FORBIDDEN.search(pred):
        return None, "forbidden"
    try:
        fn = eval("lambda n: (" + pred + ")", SAFE_GLOBALS)
    except Exception as e:
        return None, "build:" + type(e).__name__
    saw = False
    for n in dom:
        try:
            ok = fn(n)
        except ZeroDivisionError:
            continue
        except ValueError:
            continue           # helper refused a too-large input -> skip this n
        except Exception as e:
            return None, "runtime:" + type(e).__name__
        saw = True
        if not ok:
            return False, "counterexample"
    return (True, "holds") if saw else (None, "no-checkable-n")

def check_predicate(pred, dom=N_RANGE):
    """Back-compatible single-value wrapper (label only)."""
    return check_predicate_v(pred, dom)[0]

# ── timeout-proof verification ───────────────────────────────────────────────
# A model can emit a predicate that is finite but enormously expensive (e.g. a
# double-loop sum-of-two-squares scan, ~O(n^2) per n). Such code holds Python's
# GIL, so a thread-based timer cannot interrupt it — only a separate PROCESS that
# we can kill. To stay fast we verify a whole file in ONE child; only if that
# child stalls do we fall back to per-claim timeouts for that file.
FILE_TIMEOUT  = 20     # seconds for a whole file before falling back
CLAIM_TIMEOUT = 4      # seconds per individual predicate in the fallback path

def _verify_all_worker(preds, q):
    q.put([check_predicate_v(p) for p in preds])

def _verify_one_worker(pred, q):
    q.put(check_predicate_v(pred))

def _run_proc(target, arg, timeout):
    """Run target(arg, queue) in a killable child. Returns its result or None."""
    q = mp.Queue()
    p = mp.Process(target=target, args=(arg, q), daemon=True)
    p.start(); p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join()
        return None                     # timed out -> uncheckable
    try:
        return q.get(timeout=2)
    except Exception:
        return None

def verify(claims):
    """Set c['truth'] and c['truth_reason'] for each claim, never hanging on a
    pathological predicate. truth_reason makes a wall of uncheckable nodes
    diagnosable (missing helper, timeout, bad formalization, ...)."""
    preds = [c.get("predicate", "") for c in claims]
    results = _run_proc(_verify_all_worker, preds, FILE_TIMEOUT)
    if results is None or len(results) != len(preds):
        # a predicate stalled the whole-file pass: isolate it per claim
        results = []
        for p in preds:
            r = _run_proc(_verify_one_worker, p, CLAIM_TIMEOUT)
            results.append(r if r is not None else (None, "timeout"))
    for c, r in zip(claims, results):
        if isinstance(r, tuple):
            c["truth"], c["truth_reason"] = r
        else:                            # defensive: legacy bare label
            c["truth"], c["truth_reason"] = r, ("holds" if r is True else
                                                "counterexample" if r is False else "?")
    return claims

# ───────────────────────── reporting (adds reason histogram) ─────────────────
def report(claims):
    """Original calibration_harness report, plus a breakdown of WHY nodes were
    uncheckable so a high gray rate is immediately diagnosable."""
    m = ch.report(claims)
    why = Counter(c.get("truth_reason", "?") for c in claims if c.get("truth") is None)
    if why:
        print("  uncheckable breakdown: " +
              ", ".join(f"{k}={v}" for k, v in why.most_common()))
    return m

# ───────────────────────── the seed bank ────────────────────────────────────
# Each entry: the root CLAIM the model is given as node 0, plus a reference
# predicate + ground truth (used by --list-seeds / sanity, NOT shown to the model).
SEED_BANK = [
    {"subfield": "divisibility",
     "claim": "For all integers n, n^3 - n is divisible by 6.",
     "ref_pred": "(n**3 - n) % 6 == 0", "truth": True},
    {"subfield": "congruence",
     "claim": "For all integers n, n^2 is congruent to 0 or 1 modulo 4.",
     "ref_pred": "(n*n) % 4 in (0, 1)", "truth": True},
    {"subfield": "gcd",
     "claim": "For all integers n, gcd(n^2 + 1, n + 1) divides 2.",
     "ref_pred": "gcd(n*n + 1, n + 1) in (1, 2)", "truth": True},
    {"subfield": "finite sum (Nicomachus)",
     "claim": "For all integers n >= 1, (1 + 2 + ... + n)^2 equals 1^3 + 2^3 + ... + n^3.",
     "ref_pred": "(n < 1) or (sum(k for k in range(1, n + 1))**2 == sum(k**3 for k in range(1, n + 1)))",
     "truth": True},
    {"subfield": "divisor-sum identity (Gauss)",
     "claim": "For all integers n >= 1, the sum of phi(d) over all positive divisors d of n equals n.",
     "ref_pred": "(n < 1) or (sum(phi(d) for d in divisors(n)) == n)", "truth": True},
    {"subfield": "divisor-function inequality",
     "claim": "For all integers n >= 1, sigma(n) * phi(n) is at most n^2.",
     "ref_pred": "(n < 1) or (sigma(n) * phi(n) <= n*n)", "truth": True},
    {"subfield": "primality x congruence",
     "claim": "For all integers n, if n is prime and n > 3 then n^2 is congruent to 1 modulo 24.",
     "ref_pred": "(not is_prime(n)) or (n <= 3) or ((n*n - 1) % 24 == 0)", "truth": True},
    {"subfield": "primality-dependent (Fermat)",
     "claim": "For all integers n > 1, if n is prime then 2^n - 2 is divisible by n.",
     "ref_pred": "(n < 2) or (not is_prime(n)) or ((pow(2, n, n) - 2) % n == 0)", "truth": True},
    {"subfield": "divisor function (tau)",
     "claim": "For all integers n >= 1, the number of divisors of n is odd if and only if n is a perfect square.",
     "ref_pred": "(n < 1) or ((tau(n) % 2 == 1) == (math.isqrt(n)**2 == n))", "truth": True},
    {"subfield": "divisor function (sigma)",
     "claim": "For all integers n > 1, sigma(n) is at least n + 1, with equality exactly when n is prime.",
     "ref_pred": "(n <= 1) or ((sigma(n) >= n + 1) and ((sigma(n) == n + 1) == is_prime(n)))",
     "truth": True},
    {"subfield": "totient",
     "claim": "For all integers n > 2, Euler's totient phi(n) is even.",
     "ref_pred": "(n <= 2) or (phi(n) % 2 == 0)", "truth": True},
    {"subfield": "congruence",
     "claim": "For all integers n, n^5 is congruent to n modulo 10.",
     "ref_pred": "(n**5 - n) % 10 == 0", "truth": True},
    {"subfield": "p-adic valuation",
     "claim": "For all integers n >= 1, the exponent of 2 in (2n)! is at least n.",
     "ref_pred": "(n < 1) or (vp(factorial(2*n), 2) >= n)", "truth": True},
    {"subfield": "primality iff factorial (Wilson)",
     "claim": "For all integers n > 1, n is prime if and only if (n-1)! + 1 is divisible by n.",
     "ref_pred": "(n <= 1) or (((factorial(n - 1) % n) == (n - 1)) == is_prime(n))",
     "truth": True},
    {"subfield": "primality (DELIBERATELY FALSE SEED)",
     "claim": "For all integers n >= 0, n^2 + n + 41 is prime.",
     "ref_pred": "(n < 0) or is_prime(n*n + n + 41)", "truth": False},
]

# ───────────────────────── generation prompt ────────────────────────────────
GEN_SYSTEM = (
    "You are a research mathematician exploring elementary number theory over the "
    "integers. From a single seed statement you grow a large directed acyclic graph "
    "of dependent claims, reasoning on structural intuition, analogy, and pattern, hunting for "
    "deep, non-obvious generalizations and consequences. You are encouraged to be "
    "thorough and to include non-trivial conjectures you are less than certain of. "
    "You build CONVERGENT graphs where claims frequently combine two or more earlier results; you avoid linear chains. "
)

GEN_USER = """Build a directed acyclic graph of exactly {n} dependent number-theory claims rooted at:

  node 0: "{seed}"

Grow MANY distinct claims. Make each claim its OWN
node — do NOT compress several cases into one general claim with extra variables;
prefer many concrete nodes over a few sweeping ones (e.g. give n^5 - n, n^7 - n,
n^11 - n as separate nodes rather than one parameterized family). As you branch,
move through different subfields: divisibility, congruences, parity,
factorization identities, gcd statements, primality-dependent statements, finite
sums or products, and divisor functions (phi, tau, sigma).
Many claims should depend on MULTIPLE earlier claims — list every node it genuinely 
builds on in "parents", not just one. A claim that combines two prior facts 
(e.g. "divisible by 8" AND "divisible by 3" ⟹ "divisible by 24") must list both parents. 
REQUIREMENT (mandatory): at LEAST 5 in every 10 non-seed nodes MUST have two or
more parents. Do NOT produce linear chains where each node has a single parent.

Every claim is a UNIVERSAL statement about a single integer variable n, tested
over n in -500..500. Encode any hypothesis as an implication inside the
predicate, e.g. "if n is prime then P" becomes "(not is_prime(n)) or (P)", and
"for odd n, P" becomes "(n % 2 == 0) or (P)".

Helpers usable in predicates:
  gcd(a,b)  is_prime(x)  phi(x)  tau(x)  sigma(x)  vp(x,p)
  abs pow range all any sum min max len int round math factorial
(phi = Euler totient, tau = number of divisors, sigma = sum of divisors,
 vp(x,p) = exponent of prime p in x.)

CRITICAL RULE: Do not generate "strawman" claims you instantly recognize as
trivially false (e.g. divisibility you can break at n=2). Every claim must be a
plausible, deeply considered conjecture an expert would find reasonable to
investigate. If a claim turns out false, it must be from a subtle, non-trivial
breakdown — not an obvious oversight.

For EACH claim output an object with exactly these fields:
  "node_id"    : integer, 0 for the seed, unique and increasing
  "parents"    : list of earlier node_ids it depends on ([] only for the seed)
  "claim"      : the precise mathematical statement, in words
  "predicate"  : a single Python expression in the one integer variable n, True
                 exactly when the claim holds for that n. It faithfully encodes
                 the CLAIM whether or not the claim is correct.
  "confidence" : your probability 0.0-1.0 that the claim is true, from structural
                 intuition BEFORE any mechanical check. Never output 0.0; omit any
                 claim you have zero confidence in.
  "is_terminal": true if it should not be expanded further.

Output ONLY a JSON array of these objects. No prose, no markdown, no code fences."""

# ───────────────────────── selftest mock ────────────────────────────────────
MOCK = [
 {"node_id":0,"parents":[],"claim":"n^3-n div by 6","predicate":"(n**3-n)%6==0","confidence":0.97,"is_terminal":False},
 {"node_id":1,"parents":[0],"claim":"tau(n) odd iff square","predicate":"(n<1) or ((tau(n)%2==1)==(math.isqrt(n)**2==n))","confidence":0.85,"is_terminal":True},
 {"node_id":2,"parents":[0],"claim":"sigma(n) is always even (FALSE: n=1)","predicate":"(n<1) or (sigma(n)%2==0)","confidence":0.5,"is_terminal":True},
 {"node_id":3,"parents":[0],"claim":"gcd(n,n+2) divides 2","predicate":"gcd(n,n+2) in (1,2)","confidence":0.9,"is_terminal":True},
 {"node_id":4,"parents":[0],"claim":"phi(n) even for n>2","predicate":"(n<=2) or (phi(n)%2==0)","confidence":0.8,"is_terminal":True},
 {"node_id":5,"parents":[0],"claim":"Euler poly always prime (FALSE)","predicate":"(n<0) or is_prime(n*n+n+41)","confidence":0.6,"is_terminal":True},
]

def _seed(a, default=0):
    return int(a[a.index("--seed") + 1]) if "--seed" in a else default

def main():
    a = sys.argv[1:]
    if "--list-seeds" in a:
        print("SEED BANK  (use --seed <index>):\n")
        for i, s in enumerate(SEED_BANK):
            print(f"  [{i:>2}] ({s['subfield']})  truth={s['truth']}")
            print(f"        {s['claim']}")
        return
    if "--print-prompt" in a:
        i = _seed(a)
        seed = SEED_BANK[i]["claim"]
        print(f"# seed index {i}  ({SEED_BANK[i]['subfield']})\n")
        print("SYSTEM:\n" + GEN_SYSTEM + "\n\nUSER:\n" + GEN_USER.format(n=50, seed=seed))
        print(f"\n[Paste into a FRESH chat (web search OFF). Save the JSON as e.g. "
              f"claude_seed{i}_run1.json, then: python seed_harness.py --from-json <file>]")
        return
    if "--selftest" in a:
        claims = verify([dict(c) for c in MOCK])
        m = report(claims); draw(claims, m, "selftest_seed"); return
    if "--from-json" in a:
        path = a[a.index("--from-json") + 1]
        prefix = os.path.splitext(os.path.basename(path))[0]
        claims = verify(parse_claims(open(path, encoding="utf-8").read()))
        json.dump(claims, open(prefix + "_verified.json", "w"), indent=2)
        print(f"input: {path}  ->  outputs prefixed '{prefix}_'")
        m = report(claims); draw(claims, m, save_prefix=prefix); return
    if "--generate" in a:
        i = _seed(a)
        seed = SEED_BANK[i]["claim"]
        model = a[a.index("--model") + 1] if "--model" in a else ch.MODEL_DEFAULT
        n = int(a[a.index("--n") + 1]) if "--n" in a else 50
        prefix = a[a.index("--name") + 1] if "--name" in a else \
            re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_") + f"_seed{i}"
        print(f"generating >= {n} claims with {model} from seed {i} (tools off)...")
        raw = call_model(GEN_SYSTEM, GEN_USER.format(n=n, seed=seed), model)
        claims = verify(parse_claims(raw))
        json.dump(claims, open(prefix + "_verified.json", "w"), indent=2)
        print(f"saved {prefix}_verified.json")
        m = report(claims); draw(claims, m, save_prefix=prefix); return
    print(__doc__)

if __name__ == "__main__":
    main()