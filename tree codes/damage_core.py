"""
damage_core.py: the damage / propagation math, in one place.

tree_model.py, dag_model.py and adaptive_dag_model.py used to each carry
their own copy of the geometric subtree-size estimate and their own
expected-damage formula, and the copies had drifted (rho multiplied in
some, folded into p_false in others; a budget cap in some, none in
others). Clinic feedback called this out, so the three functions below
are now the only definition, and the model files just wrap them.

Every existing import still works (`from tree_model import
expected_damage`, etc.) and returns the same numbers as before, since the
wrappers just call into here.

A couple of differences between the models are real and worth keeping
straight, not bugs:

  * tree_model uses the union error model, so rho is already baked into
    p_false through the propagation chain. Its damage call passes
    rho=1.0, otherwise contamination gets counted twice.
  * dag_model and adaptive_dag_model use the additive three-state model
    instead, so they pass rho in explicitly as a multiplier.
  * adaptive_dag_model has never capped the geometric series by remaining
    node budget (cap=None), while tree_model and dag_model do. Left as-is
    so old sigma-sweep numbers don't shift under us; worth revisiting once
    sigma becomes a derived per-node threshold.
"""

from typing import Optional


def geometric_descendants(
    branching: float,
    remaining_depth: int,
    cap: Optional[float] = None,
) -> float:
    """
    Expected number of future descendants of a node, given `remaining_depth`
    levels left to grow under a constant branching factor:

        E[descendants] = sum_{i=1}^{r} b^i = (b^(r+1) - b) / (b - 1)

    (equals r exactly when b == 1). Pass `cap`, usually the remaining node
    budget, to clip the estimate.

    Static tree, offline DAG, adaptive fog-of-war loop: they all call this
    for their branching/expansion estimate.
    """
    if remaining_depth is None or remaining_depth <= 0:
        return 0.0

    if abs(branching - 1.0) < 1e-9:
        structural = float(remaining_depth)
    else:
        structural = (branching ** (remaining_depth + 1) - branching) / (
            branching - 1.0
        )

    if cap is not None:
        structural = min(structural, float(cap))
    return float(structural)


def expected_damage_score(p_false: float, d_hat: float, rho: float = 1.0) -> float:
    """
    ExpectedDamage(v) = P(v false) * rho * E[future descendants of v]

    Check which rho you actually want before calling this:
      * tree_model's union error model already chains propagation into
        p_false via p_union(), so pass rho=1.0 here, or contamination gets
        counted twice.
      * dag_model and adaptive_dag_model's p_false is a per-node estimate
        with no downstream propagation baked in, so rho has to be passed
        in explicitly as the contamination multiplier.
    """
    return p_false * rho * d_hat


def p_union(p_ind: float, p_prop: float) -> float:
    """
    Probabilistic OR of two independent failure channels:

        p_error = 1 - (1 - p_ind)(1 - p_prop)

    p_ind is the independent per-node error rate (epsilon); p_prop is the
    propagated channel, rho * parent.p_false. The tree simulation's error
    model is built on this.
    """
    return float(1.0 - (1.0 - p_ind) * (1.0 - p_prop))


def additive_propagation(epsilon: float, rho: float, parent_p_false: float) -> float:
    """
    The DAG model's clipped additive propagation update:

        child.p_false = clip(epsilon + rho * parent_p_false, 0, 1)

    This stays around next to p_union() because the DAG code, and its
    published numbers, were built on this rule. Union is the better model
    going forward, but don't swap it in under existing experiments without
    checking what that does to their results.
    """
    v = epsilon + rho * parent_p_false
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)
