"""Right-censored Weibull life-data reliability analysis.

Answers three questions a design-verification life test actually needs
after the samples come off test: given N units on test per failure mode,
each observed either to fail or to survive to the end of the test window
(right-censored), what are the true Weibull shape (beta) and scale (eta)
parameters behind each failure mode, what is the B10 life (the time by
which 10% of the population is expected to have failed) with a
confidence bound on that estimate, and which failure modes should get
engineering attention first.

Right-censored Weibull maximum likelihood
------------------------------------------
For a unit observed at time t, the contribution to the likelihood is the
Weibull density f(t; beta, eta) if the unit failed at t, or the
survival function S(t; beta, eta) = exp(-(t/eta)^beta) if the unit was
still alive when the test was stopped at t (right-censored). The joint
log-likelihood over all units is

    ln L(beta, eta) = sum_{failures}   [ ln(beta/eta) + (beta-1) ln(t_i/eta) - (t_i/eta)^beta ]
                     + sum_{censored}  [ -(t_i/eta)^beta ]

This is the standard censored-data Weibull likelihood (see e.g. Meeker &
Escobar, "Statistical Methods for Reliability Data", chapter 8, or
NIST/SEMATECH e-Handbook of Statistical Methods, section 8.1.7). Fitting
by MLE on only the failed units' times (throwing the censored units away
entirely) ignores real information, every censored unit tells the
likelihood "this unit survived at least until its censoring time", and
produces a biased fit; see `fit_weibull_uncensored_naive` below, kept
specifically for the Findings narrative in the README documenting that
this was the first (wrong) implementation attempted here, and
`tests/test_reliability.py::test_naive_uncensored_fit_is_more_biased_than_censored_mle`,
which pins the bug down numerically so it cannot silently regress back in.

Optimization is done in (log(beta), log(eta)) space so the search is
unconstrained (beta and eta must both be positive) and handed to
`scipy.optimize.minimize` with the derivative-free Nelder-Mead method,
which needs no analytic gradient and is robust on this well-behaved 2-D
surface. The asymptotic covariance of the MLE is obtained from a
finite-difference Hessian of the negative log-likelihood evaluated at the
fitted optimum (the observed information matrix), inverted; this is used
directly for the B10 confidence interval below rather than trusting
whatever quasi-Newton inverse-Hessian approximation a gradient-based
optimizer happens to have accumulated.

B10 life and its 90% confidence interval
-----------------------------------------
B10(beta, eta) solves F(B10) = 0.10, i.e.

    B10 = eta * (-ln(0.9))^(1/beta)

The confidence interval is built by the delta method entirely in
log-space, which is the standard approach for Weibull life quantiles
because it respects B10 > 0 for any finite-sample interval (a
symmetric interval built directly on B10 could go negative at small
sample sizes). Writing u = ln(beta), v = ln(eta) (the same parametrization
the optimizer already searches over):

    ln(B10) = v + ln(-ln(0.9)) * exp(-u)

so the gradient of ln(B10) with respect to (u, v) is
(-ln(-ln(0.9)) * exp(-u), 1), and Var(ln(B10)) = grad^T Cov(u, v) grad,
where Cov(u, v) is the inverted finite-difference Hessian described
above. The 90% interval is exp(ln(B10) +/- 1.645 * sqrt(Var(ln(B10)))).

Failure-mode Pareto
--------------------
Ranked by ascending B10 life: the failure mode whose fitted Weibull
predicts 10% cumulative failures soonest is the one an engineering team
should act on first. This was chosen over an "expected failures within a
stated design life at a stated fleet size" ranking because that
alternative requires two additional assumed inputs (a design life and a
fleet size) that are not given anywhere else in this project and would
have been invented rather than measured; ranking by B10 life uses only
numbers this module already fits from data, is monotonic with the more
elaborate ranking whenever all six modes share a comparable shape
parameter, and is a standard reliability-engineering practice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import optimize
from scipy.stats import norm, weibull_min

Z_90 = 1.6448536269514722  # norm.ppf(0.95), the two-sided 90% CI multiplier
B10_QUANTILE = 0.10
_LN_B10_CONST = math.log(-math.log(1.0 - B10_QUANTILE))  # ln(-ln(0.9))


@dataclass(frozen=True)
class FailureModeSpec:
    name: str
    true_shape: float  # true injected beta
    true_scale: float  # true injected eta, in test hours
    n_units: int
    censor_time: float  # test duration in hours; units still alive at this time are right-censored
    seed: int


# Six distinct failure modes, each with its own true injected Weibull
# shape/scale, a physically-motivated name, and its own RNG seed. The
# starting point was 40 units/mode at a 5.0-hour censoring time (240
# units, 1,200 planned unit-hours total, matching the original
# design-of-experiment target); that first attempt did not recover all 12
# injected parameters within 6% relative error (max error 52.9%, driven
# by failure modes with as few as 11 observed failures out of 40 units).
# A second attempt at 200 units/mode (6,000 planned unit-hours) improved
# the max error to 13.6% but still missed. This third and final attempt
# uses 1,000 units/mode (5.0-hour censoring, 30,000 planned unit-hours
# total across all six modes), a genuine increase in simulated data
# volume to reduce small-sample MLE variance, not a change to the 6%
# bar itself. See the README Findings entry for the full account,
# including that this third attempt still narrowly misses the 6% bar on
# two of the six modes and is reported honestly as such.
FAILURE_MODES = [
    FailureModeSpec("bearing_wear", true_shape=3.2, true_scale=6.0, n_units=1000, censor_time=5.0, seed=201),
    FailureModeSpec("seal_degradation", true_shape=2.0, true_scale=8.0, n_units=1000, censor_time=5.0, seed=202),
    FailureModeSpec("solder_joint_fatigue", true_shape=1.2, true_scale=5.0, n_units=1000, censor_time=5.0, seed=203),
    FailureModeSpec("electronics_random_failure", true_shape=1.0, true_scale=10.0, n_units=1000, censor_time=5.0, seed=204),
    FailureModeSpec("corrosion", true_shape=2.5, true_scale=7.0, n_units=1000, censor_time=5.0, seed=205),
    FailureModeSpec("infant_mortality_defect", true_shape=0.6, true_scale=3.0, n_units=1000, censor_time=5.0, seed=206),
]

RECOVERY_TOLERANCE = 0.06  # the claim: every injected shape and scale recovered within 6% relative error


@dataclass(frozen=True)
class LifeTestData:
    mode: str
    times: np.ndarray  # observed time per unit: failure time if failed, censor_time if censored
    event: np.ndarray  # 1 = failure observed, 0 = right-censored
    n_units: int
    n_failures: int
    n_censored: int
    total_unit_hours: float


def simulate_life_test(spec: FailureModeSpec) -> LifeTestData:
    """Simulate spec.n_units units on test, each with a true Weibull(true_shape,
    true_scale) failure time; a unit is right-censored if its true failure
    time exceeds spec.censor_time. numpy's rng.weibull(a) draws from a
    standard Weibull with shape a and scale 1; scaling by true_scale gives
    the Weibull(true_shape, true_scale) family used throughout this module.
    """
    rng = np.random.default_rng(spec.seed)
    true_failure_times = rng.weibull(spec.true_shape, size=spec.n_units) * spec.true_scale
    event = (true_failure_times <= spec.censor_time).astype(int)
    observed_times = np.minimum(true_failure_times, spec.censor_time)
    return LifeTestData(
        mode=spec.name,
        times=observed_times,
        event=event,
        n_units=spec.n_units,
        n_failures=int(event.sum()),
        n_censored=int(spec.n_units - event.sum()),
        total_unit_hours=float(observed_times.sum()),
    )


def _neg_log_likelihood(log_params: np.ndarray, times: np.ndarray, event: np.ndarray) -> float:
    log_beta, log_eta = log_params
    beta = math.exp(log_beta)
    eta = math.exp(log_eta)
    # Guard against log(0) for any unit observed at exactly t=0 (probability
    # zero in the simulation, but a real life-test data set could have one).
    z = np.maximum(times, 1e-9) / eta
    ll_failures = np.sum(event * (math.log(beta) - math.log(eta) + (beta - 1.0) * np.log(z) - z ** beta))
    ll_censored = np.sum((1 - event) * (-(z ** beta)))
    return -(ll_failures + ll_censored)


def fit_weibull_censored_mle(times: np.ndarray, event: np.ndarray,
                              beta0: float = 1.0, eta0: float | None = None) -> tuple[float, float]:
    """Right-censored Weibull MLE via direct optimization of the censored
    log-likelihood defined above (module docstring). Returns (beta_hat,
    eta_hat). This is the correct method; see `fit_weibull_uncensored_naive`
    for the bug this replaced."""
    if eta0 is None:
        eta0 = float(np.mean(times[event == 1])) if event.sum() > 0 else float(np.mean(times))
    x0 = np.array([math.log(beta0), math.log(max(eta0, 1e-6))])
    result = optimize.minimize(_neg_log_likelihood, x0, args=(times, event), method="Nelder-Mead",
                                options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000, "maxfev": 20000})
    if not result.success:
        raise RuntimeError(f"censored Weibull MLE failed to converge: {result.message}")
    beta_hat = math.exp(result.x[0])
    eta_hat = math.exp(result.x[1])
    return beta_hat, eta_hat


def fit_weibull_uncensored_naive(times: np.ndarray, event: np.ndarray) -> tuple[float, float]:
    """The first (wrong) approach tried while building this module: fit a
    Weibull MLE using only the failed units' observed times via
    `scipy.stats.weibull_min.fit`, discarding every right-censored unit as
    if it had never been on test. Kept here, not deleted, specifically so
    the bias it introduces (documented in the README Findings section and
    pinned down numerically in tests/test_reliability.py) stays visible
    and cannot silently reappear as the "real" fit path."""
    failed_times = times[event == 1]
    if len(failed_times) < 2:
        raise RuntimeError("need at least 2 observed failures for the naive uncensored fit")
    beta_hat, _loc, eta_hat = weibull_min.fit(failed_times, floc=0)
    return float(beta_hat), float(eta_hat)


def _hessian_fd(f, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Finite-difference Hessian of scalar function f at x (central
    differences on the mixed second partials). Used to get the observed
    information matrix of the censored Weibull negative log-likelihood at
    its fitted optimum, independent of any optimizer-internal Hessian
    approximation."""
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_pp, x_pm = x.copy(), x.copy()
            x_mp, x_mm = x.copy(), x.copy()
            x_pp[i] += eps; x_pp[j] += eps
            x_pm[i] += eps; x_pm[j] -= eps
            x_mp[i] -= eps; x_mp[j] += eps
            x_mm[i] -= eps; x_mm[j] -= eps
            hess[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4.0 * eps * eps)
    return hess


@dataclass(frozen=True)
class RecoveryResult:
    mode: str
    true_shape: float
    true_scale: float
    fit_shape: float
    fit_scale: float
    shape_rel_error: float
    scale_rel_error: float


def check_recovery(spec: FailureModeSpec, data: LifeTestData) -> RecoveryResult:
    beta_hat, eta_hat = fit_weibull_censored_mle(data.times, data.event)
    shape_rel_error = abs(beta_hat - spec.true_shape) / spec.true_shape
    scale_rel_error = abs(eta_hat - spec.true_scale) / spec.true_scale
    return RecoveryResult(
        mode=spec.name, true_shape=spec.true_shape, true_scale=spec.true_scale,
        fit_shape=beta_hat, fit_scale=eta_hat,
        shape_rel_error=shape_rel_error, scale_rel_error=scale_rel_error,
    )


@dataclass(frozen=True)
class B10Result:
    mode: str
    beta_hat: float
    eta_hat: float
    b10: float
    ci_low: float
    ci_high: float
    confidence: float


def b10_life_with_ci(times: np.ndarray, event: np.ndarray, beta_hat: float, eta_hat: float,
                      mode_name: str = "") -> B10Result:
    """B10 life and its 90% confidence interval via the delta method in
    (log(beta), log(eta)) space, as derived in the module docstring."""
    log_params = np.array([math.log(beta_hat), math.log(eta_hat)])
    hess = _hessian_fd(lambda p: _neg_log_likelihood(p, times, event), log_params)
    cov = np.linalg.inv(hess)  # inverse observed information = asymptotic covariance of (log beta, log eta)

    u, v = log_params  # u = ln(beta), v = ln(eta)
    ln_b10 = v + _LN_B10_CONST * math.exp(-u)
    grad = np.array([-_LN_B10_CONST * math.exp(-u), 1.0])
    var_ln_b10 = float(grad @ cov @ grad)
    if var_ln_b10 < 0:
        raise RuntimeError(f"negative variance estimate for ln(B10) in mode {mode_name}; Hessian is not positive definite")
    se_ln_b10 = math.sqrt(var_ln_b10)

    b10 = math.exp(ln_b10)
    ci_low = math.exp(ln_b10 - Z_90 * se_ln_b10)
    ci_high = math.exp(ln_b10 + Z_90 * se_ln_b10)
    return B10Result(mode=mode_name, beta_hat=beta_hat, eta_hat=eta_hat, b10=b10,
                      ci_low=ci_low, ci_high=ci_high, confidence=0.90)


@dataclass(frozen=True)
class ReliabilityStudyResult:
    recoveries: list
    b10_results: list
    max_rel_error: float
    meets_claim: bool
    pareto_order: list  # mode names, ascending B10 (highest priority first)


def run_reliability_study() -> ReliabilityStudyResult:
    """Runs the full study: simulate all six failure modes, fit each with
    the censored MLE, check recovery, compute B10 with CI, and rank the
    Pareto order by ascending B10 life."""
    recoveries = []
    b10_results = []
    for spec in FAILURE_MODES:
        data = simulate_life_test(spec)
        recovery = check_recovery(spec, data)
        recoveries.append(recovery)
        b10 = b10_life_with_ci(data.times, data.event, recovery.fit_shape, recovery.fit_scale, mode_name=spec.name)
        b10_results.append(b10)

    max_rel_error = max(
        max(r.shape_rel_error, r.scale_rel_error) for r in recoveries
    )
    meets_claim = max_rel_error <= RECOVERY_TOLERANCE
    pareto_order = [b.mode for b in sorted(b10_results, key=lambda b: b.b10)]

    return ReliabilityStudyResult(
        recoveries=recoveries, b10_results=b10_results,
        max_rel_error=max_rel_error, meets_claim=meets_claim, pareto_order=pareto_order,
    )
