"""Seven locked toy checks. Seed 0. One PASS or FAIL line each.

Constructions are fixed witnesses of the statements in the prompt. They are
not retuned if an assert fails.
"""

import os
import random
import sys

import numpy as np
import torch

os.environ["PYTHONHASHSEED"] = "0"
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from radial import (  # noqa: E402
    append_gate,
    auroc_inlier,
    certificate,
    evaluate_mapped_split,
    fit_id_covariance,
    mahalanobis_radius,
    radius_inversion_mw,
)

N = 2000
_FAILED = False


def _line(ok: bool, name: str) -> None:
    global _FAILED
    if not ok:
        _FAILED = True
    print(("PASS " if ok else "FAIL ") + name)


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def check_equivalence() -> None:
    rng = _rng()
    x_id = rng.normal(loc=0.0, scale=1.0, size=(N, 2))
    x_new = rng.normal(loc=0.0, scale=0.5, size=(N, 2)) + np.array([1.5, -0.5])
    mu = x_id.mean(axis=0)
    mu_new = x_new.mean(axis=0)
    sigma, precision = fit_id_covariance(x_id)
    row = evaluate_mapped_split(x_id, x_new, mu, mu_new, sigma, precision)
    ok = (
        abs(row["auroc_mahalanobis"] - row["auroc_gaussian"]) <= 1e-6
        and row["auroc_mahalanobis"] > 0.5
        and row["auroc_gaussian"] > 0.5
        and abs(row["premap_mw"] - row["auroc_mahalanobis"]) <= 1e-6
    )
    _line(ok, "equivalence")


def check_quantile_not_necessary() -> None:
    rng = _rng()
    d_id = rng.uniform(0.0, 1.0, size=N)
    d_new = rng.uniform(0.0, 0.9, size=N)
    mw = radius_inversion_mw(d_new, d_id)
    _cert, eps = certificate(d_new, d_id)
    ok = mw > 0.5 and eps == "none"
    _line(ok, "quantile_not_necessary")


def check_certificate_fires() -> None:
    rng = _rng()
    x_id = rng.normal(loc=0.0, scale=1.0, size=(N, 2))
    near = rng.normal(loc=0.0, scale=1e-3, size=(1600, 2))
    far_pos = rng.normal(loc=(40.0, 0.0), scale=1e-3, size=(200, 2))
    far_neg = rng.normal(loc=(-40.0, 0.0), scale=1e-3, size=(200, 2))
    x_new = np.vstack([near, far_pos, far_neg])
    if x_new.shape != (N, 2):
        _line(False, "certificate_fires")
        return
    mu = x_id.mean(axis=0)
    mu_new = x_new.mean(axis=0)
    _sigma, precision = fit_id_covariance(x_id)
    d_id = mahalanobis_radius(x_id, mu, precision)
    d_new = mahalanobis_radius(x_new, mu_new, precision)
    # The far fifth of the cloud stays in the sample.
    if not (np.count_nonzero(np.linalg.norm(x_new, axis=1) > 10.0) == 400):
        _line(False, "certificate_fires")
        return
    mw = radius_inversion_mw(d_new, d_id)
    _cert, eps = certificate(d_new, d_id)
    ok = mw > 0.5 and eps in {"0.05", "0.10", "0.20"}
    _line(ok, "certificate_fires")


def check_control() -> None:
    rng = _rng()
    x_id = rng.normal(loc=0.0, scale=1.0, size=(N, 2))
    x_new = rng.normal(loc=0.0, scale=2.0, size=(N, 2))
    mu = x_id.mean(axis=0)
    mu_new = x_new.mean(axis=0)
    sigma, precision = fit_id_covariance(x_id)
    d_id = mahalanobis_radius(x_id, mu, precision)
    d_new = mahalanobis_radius(x_new, mu_new, precision)
    premap = radius_inversion_mw(d_new, d_id)
    row = evaluate_mapped_split(x_id, x_new, mu, mu_new, sigma, precision)
    ok = premap < 0.5 and row["auroc_mahalanobis"] < 0.5
    _line(ok, "control")


def _sigma_orthogonal(sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    eigenvalues, evecs = np.linalg.eigh(np.asarray(sigma, dtype=np.float64))
    if np.any(eigenvalues <= 0.0):
        raise RuntimeError("covariance is not positive definite")
    sqrt = (evecs * np.sqrt(eigenvalues)) @ evecs.T
    inv_sqrt = (evecs * (1.0 / np.sqrt(eigenvalues))) @ evecs.T
    gaussian = rng.normal(size=sigma.shape)
    q, r = np.linalg.qr(gaussian)
    q = q @ np.diag(np.sign(np.diag(r)))
    q[:, np.diag(r) == 0.0] *= 1.0
    return sqrt @ q @ inv_sqrt


def check_rotation_invisibility() -> None:
    rng = _rng()
    x_id = rng.normal(size=(N, 2))
    x_new = rng.normal(size=(N, 2)) + np.array([0.3, -0.2])
    mu = x_id.mean(axis=0)
    mu_new = x_new.mean(axis=0)
    sigma, precision = fit_id_covariance(x_id)
    a = _sigma_orthogonal(sigma, rng)
    if not np.allclose(a @ sigma @ a.T, sigma, rtol=0.0, atol=1e-6):
        _line(False, "rotation_invisibility")
        return
    centered = x_new - mu_new
    t_i = centered + mu
    t_a = centered @ a.T + mu
    d_i = np.sort(mahalanobis_radius(t_i, mu, precision))
    d_a = np.sort(mahalanobis_radius(t_a, mu, precision))
    ok = d_i.shape == d_a.shape and np.max(np.abs(d_i - d_a)) <= 1e-6
    _line(ok, "rotation_invisibility")


def check_constant_phi() -> None:
    def phi(t: np.ndarray) -> np.ndarray:
        return np.zeros_like(np.asarray(t, dtype=np.float64), dtype=np.float64)

    grid = np.array([-2.0, -1.0, 0.0])
    images = phi(grid)
    strictly = bool(np.all(np.diff(images) > 0.0))
    s_new = phi(np.zeros(128))
    s_id = phi(np.ones(128))
    auc, _mw = auroc_inlier(s_new, s_id)
    ok = (not strictly) and (not (auc > 0.5))
    _line(ok, "constant_phi")


def check_variance_counterexample() -> None:
    eps = 1e-3
    n = 400
    s_correct = np.zeros(n, dtype=np.float64)
    s_incorrect = np.full(n, eps, dtype=np.float64)
    point_masses = np.all(s_correct == 0.0) and np.all(s_incorrect == eps)
    auc, mw = auroc_inlier(s_incorrect, s_correct)
    ok = point_masses and auc == 1.0 and mw == 1.0
    _line(ok, "variance_counterexample")


def main() -> None:
    checks = (
        check_equivalence,
        check_quantile_not_necessary,
        check_certificate_fires,
        check_control,
        check_rotation_invisibility,
        check_constant_phi,
        check_variance_counterexample,
    )
    code = 0
    try:
        for check in checks:
            check()
        if _FAILED:
            code = 1
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 1
        append_gate("toy", code)
        raise
    append_gate("toy", code)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
