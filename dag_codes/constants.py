"""
Provenance
----------
BASE_ERROR_RATE = 0.1758
    Empirical base error rate from running the PARC dataset
    (github.com/SagnikMukherjee/PARC): P(claim wrong | parent correct).
    Cross-checked against three papers reporting 0.15 / 0.19 / 0.21
    (mean 0.18) — see Clinic I brief.

RHO = 0.7504
    Propagation strength derived from PARC:
    P(wrong | parent wrong) - P(wrong | parent correct)
    = 0.9262 - 0.1758.

BRANCHING_LAMBDA_DAG = 2.2
    Historical working value for DAG experiments. Flagged in Clinic II as
    needing literature grounding — the geometry/topology work (post
    mvp-v1) is the fix; until then this is the documented status quo, not
    a claim.

BRANCHING_LAMBDA_TREE = 1.7
    Same status, tree experiments. calibrate_branching_from_simulations()
    in tree_model gives the empirical realized branching for any chosen
    lambda (Poisson thinning by depth/node caps makes realized < nominal).

MAX_DEPTH = 6
    Standard experiment depth cap (proposal §3.3 suggested d_max = 6).

EXTRA_EDGE_PROB = 0.15
    P(node gains a second parent) in DAG generation — the tree->DAG knob.
    0.20 appears as ClaimDAG's constructor default; experiments have used
    0.15 throughout (run_experiment, run_adaptive_simulation defaults).

Offline-vs-adaptive wrinkle (pre-existing, documented not fixed):
dag_main's offline strategy comparison runs ClaimDAG's constructor
defaults (0.15 / 0.85), while every adaptive experiment uses the
PARC-derived pair below. Both are frozen behavior under mvp-v1.
"""

BASE_ERROR_RATE = 0.1758
RHO = 0.7504
PROPAGATION_RATE = BASE_ERROR_RATE + RHO      # 0.9262, PARC P(wrong|parent wrong)

BRANCHING_LAMBDA_DAG = 2.2
BRANCHING_LAMBDA_TREE = 1.7

MAX_DEPTH = 6
EXTRA_EDGE_PROB = 0.15

# Derived verification threshold (see derived_sigma_main RULE D):
# opportunity cost of a check = one displaced expansion's expected yield.
RULE_D_THRESHOLD_DAG = BRANCHING_LAMBDA_DAG * (1.0 - BASE_ERROR_RATE)  # ~1.81
