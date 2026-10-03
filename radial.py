"""Radial-order quantities shared by the toy and the embedding run.

Population claims live in THEORY.md. This module only evaluates the pinned
finite-sample counterparts. It does not search epsilon, refit Ledoit-Wolf,
or add a ridge.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

EPSILONS = (0.05, 0.10, 0.20)
AUROC_ATOL = 1e-6
ROOT = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_gate(name: str, code: int) -> None:
    path = ROOT / "GATES.md"
    if not path.exists():
        path.write_text("# Gates\n\n| gate | utc_timestamp | exit_code |\n| --- | --- | --- |\n")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"| {name} | {utc_now()} | {code} |\n")


def empirical_quantile(samples: np.ndarray, p: float) -> float:
    """Generalized inverse of the empirical CDF.

    F_n^{-1}(p) = inf{t : F_n(t) >= p}, with F_n(t) = #{X_i <= t} / n.
    This is the empirical counterpart of the quantile in the certificate.
    """
    xs = np.sort(np.asarray(samples, dtype=np.float64).reshape(-1))
    n = xs.shape[0]
    if n == 0 or not (0.0 < p <= 1.0):
        raise RuntimeError("empirical quantile is undefined for this sample")
    k = int(np.ceil(p * n) - 1)
    k = min(max(k, 0), n - 1)
    return float(xs[k])


def certificate(d_new: np.ndarray, d_id: np.ndarray) -> tuple[str, str]:
    """Return (certificate, epsilon).

    epsilon is the smallest value in {0.05, 0.10, 0.20} that fires, or none.
    """
    for eps in EPSILONS:
        q_new = empirical_quantile(d_new, 1.0 - eps)
        q_id = empirical_quantile(d_id, eps)
        if q_new <= q_id:
            return "yes", f"{eps:.2f}"
    return "no", "none"


def fit_id_covariance(x_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit Ledoit-Wolf once on raw ID rows. Stop if Sigma is not usable."""
    x = np.asarray(x_id, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise RuntimeError("ID matrix is not usable for Ledoit-Wolf")
    if not np.isfinite(x).all():
        raise RuntimeError("ID matrix has non-finite entries")
    fitted = LedoitWolf(assume_centered=False).fit(x)
    sigma = np.asarray(fitted.covariance_, dtype=np.float64)
    if sigma.shape != (x.shape[1], x.shape[1]):
        raise RuntimeError("Ledoit-Wolf covariance has the wrong shape")
    if not np.isfinite(sigma).all():
        raise RuntimeError("Ledoit-Wolf covariance is not finite")
    if not np.allclose(sigma, sigma.T, rtol=0.0, atol=1e-8):
        raise RuntimeError("Ledoit-Wolf covariance is not symmetric")
    eigenvalues = np.linalg.eigvalsh(sigma)
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0.0):
        raise RuntimeError("Ledoit-Wolf covariance is not positive definite")
    try:
        precision = np.linalg.inv(sigma)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Ledoit-Wolf covariance is not invertible") from exc
    if not np.isfinite(precision).all():
        raise RuntimeError("precision matrix is not finite")
    return sigma, precision


def mahalanobis_radius(x: np.ndarray, mu: np.ndarray, precision: np.ndarray) -> np.ndarray:
    diff = np.asarray(x, dtype=np.float64) - np.asarray(mu, dtype=np.float64)
    quad = np.einsum("ij,jk,ik->i", diff, precision, diff)
    if not np.isfinite(quad).all():
        raise RuntimeError("Mahalanobis quadratic form is not finite")
    # Negative values at this scale are linear-algebra roundoff, not a ridge.
    if np.any(quad < -1e-8):
        raise RuntimeError("Mahalanobis quadratic form is substantially negative")
    quad = np.maximum(quad, 0.0)
    return np.sqrt(quad)


def gaussian_loglik(radius: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Log-density of N(mu, Sigma). Depends on the point only through radius."""
    sign, logdet = np.linalg.slogdet(np.asarray(sigma, dtype=np.float64))
    if sign <= 0 or not np.isfinite(logdet):
        raise RuntimeError("Gaussian log-likelihood requires a usable covariance")
    dim = sigma.shape[0]
    r2 = np.asarray(radius, dtype=np.float64) ** 2
    return -0.5 * (r2 + logdet + dim * np.log(2.0 * np.pi))


def auroc_inlier(s_new: np.ndarray, s_id: np.ndarray) -> tuple[float, float]:
    """Inlier AUROC. y=1 on new-site rows. Exit if sklearn and Mann-Whitney disagree."""
    s_new = np.asarray(s_new, dtype=np.float64).reshape(-1)
    s_id = np.asarray(s_id, dtype=np.float64).reshape(-1)
    if s_new.size == 0 or s_id.size == 0:
        raise RuntimeError("AUROC received an empty score vector")
    if not np.isfinite(s_new).all() or not np.isfinite(s_id).all():
        raise RuntimeError("AUROC received a non-finite score")
    y = np.concatenate([np.ones(s_new.size), np.zeros(s_id.size)])
    s = np.concatenate([s_new, s_id])
    auc = float(roc_auc_score(y, s))
    statistic = mannwhitneyu(s_new, s_id, alternative="greater", method="auto").statistic
    mw = float(statistic) / float(s_new.size * s_id.size)
    if abs(auc - mw) > AUROC_ATOL:
        print(
            f"AUROC cross-check failed: sklearn={auc:.12g} mannwhitney={mw:.12g}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return auc, mw


def radius_inversion_mw(d_new: np.ndarray, d_id: np.ndarray) -> float:
    """Tie-averaged Mann-Whitney estimate of P(d_new < d_id)."""
    _auc, mw = auroc_inlier(-np.asarray(d_new, dtype=np.float64), -np.asarray(d_id, dtype=np.float64))
    return mw


def oracle_translate(z: np.ndarray, mu_new: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """T(z) = z - mu_new + mu. Only A = I."""
    return np.asarray(z, dtype=np.float64) - np.asarray(mu_new, dtype=np.float64) + np.asarray(mu, dtype=np.float64)


def arithmetic_mean(x: np.ndarray) -> np.ndarray:
    mu = np.asarray(x, dtype=np.float64).mean(axis=0)
    if not np.isfinite(mu).all():
        raise RuntimeError("arithmetic mean is not finite")
    return mu


def euclidean_inlier(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Inlier score for Sigma proportional to I: phi(-||z-mu||_2) with phi the identity."""
    diff = np.asarray(x, dtype=np.float64) - np.asarray(mu, dtype=np.float64)
    return -np.linalg.norm(diff, axis=1)


def sign_half(value: float) -> float:
    return float(np.sign(value - 0.5))


def split_passes(premap_mw: float, auroc_mahalanobis: float, gap: float) -> str:
    signs_match = sign_half(auroc_mahalanobis) == sign_half(premap_mw)
    gap_ok = abs(gap) <= AUROC_ATOL
    return "yes" if signs_match and gap_ok else "no"


def evaluate_mapped_split(
    x_id: np.ndarray,
    x_new: np.ndarray,
    mu: np.ndarray,
    mu_new: np.ndarray,
    sigma: np.ndarray,
    precision: np.ndarray,
) -> dict[str, float | str | int]:
    """Pre-map radii, then T, then scores. Does not fit a covariance."""
    d_id = mahalanobis_radius(x_id, mu, precision)
    d_new = mahalanobis_radius(x_new, mu_new, precision)
    premap_mw = radius_inversion_mw(d_new, d_id)
    cert, eps = certificate(d_new, d_id)

    mapped = oracle_translate(x_new, mu_new, mu)
    d_mapped = mahalanobis_radius(mapped, mu, precision)
    d_id_post = mahalanobis_radius(x_id, mu, precision)
    auroc_m, _mw_m = auroc_inlier(-d_mapped, -d_id_post)
    auroc_g, _mw_g = auroc_inlier(
        gaussian_loglik(d_mapped, sigma),
        gaussian_loglik(d_id_post, sigma),
    )
    gap = float(auroc_m - auroc_g)
    auroc_e, _mw_e = auroc_inlier(euclidean_inlier(mapped, mu), euclidean_inlier(x_id, mu))
    return {
        "n_id": int(np.asarray(x_id).shape[0]),
        "n_new": int(np.asarray(x_new).shape[0]),
        "premap_mw": premap_mw,
        "certificate": cert,
        "epsilon": eps,
        "auroc_mahalanobis": auroc_m,
        "auroc_gaussian": auroc_g,
        "gap": gap,
        "auroc_euclidean": auroc_e,
        "pass": split_passes(premap_mw, auroc_m, gap),
    }
