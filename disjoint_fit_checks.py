"""Disjoint-fit radii on the five saved clouds. CPU only. No re-embed.

Real cells: fit (μ, Σ) on ID half A; score ID radii on disjoint half B;
keep X_new unchanged. With n_id=2000 this is 1000/1000.

Matched null: same RandomState(0) permutation of X_id; A = first 1000 (fit),
B = next 500 (ID eval), C = last 500 (pseudo-new). All three disjoint.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

ROOT = Path("/data2/cmdir/home/toandq/radial-order")
OUT = ROOT / "aistats-checks"
ATOL = 1e-6

CELL_ORDER = ("dino_derm", "dino_path", "clip_derm", "clip_path", "dino_iwild")
PUBLISHED_LW = {
    "dino_derm": 0.0155,
    "dino_path": 0.2235,
    "clip_derm": 0.00504,
    "clip_path": 0.1108,
    "dino_iwild": 0.1427,
}
RELATIVE = {
    "dino_derm": Path("reembed/dermatology"),
    "dino_path": Path("reembed/pathology"),
    "clip_derm": Path("clip/dermatology"),
    "clip_path": Path("clip/pathology"),
    "dino_iwild": Path("iwildcam"),
}


def stop(message: str) -> None:
    print(message)
    raise SystemExit(1)


def load_pair(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    x_id = np.asarray(np.load(directory / "X_id.npy"), dtype=np.float64)
    x_new = np.asarray(np.load(directory / "X_new.npy"), dtype=np.float64)
    if x_id.ndim != 2 or x_new.ndim != 2 or x_id.shape[1] != x_new.shape[1]:
        stop(f"{directory}: shapes disagree")
    return x_id, x_new


def fit_lw(x: np.ndarray) -> np.ndarray | None:
    sigma = np.asarray(LedoitWolf(assume_centered=False).fit(x).covariance_, dtype=np.float64)
    if not np.isfinite(sigma).all():
        return None
    if not np.allclose(sigma, sigma.T, rtol=0.0, atol=1e-8):
        return None
    eigenvalues = np.linalg.eigvalsh(sigma)
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0.0):
        return None
    return sigma


def radii(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray | None) -> np.ndarray | None:
    diff = np.asarray(x, dtype=np.float64) - np.asarray(mu, dtype=np.float64)
    if sigma is None:
        quad = np.sum(diff * diff, axis=1)
    else:
        try:
            solved = np.linalg.solve(sigma, diff.T).T
        except np.linalg.LinAlgError:
            return None
        quad = np.einsum("ij,ij->i", diff, solved)
    if not np.isfinite(quad).all() or np.any(quad < -1e-8):
        return None
    return np.sqrt(np.maximum(quad, 0.0))


def tie_gamma(d_new: np.ndarray, d_id: np.ndarray) -> tuple[float, float, float] | str:
    s_new = -np.asarray(d_new, dtype=np.float64)
    s_id = -np.asarray(d_id, dtype=np.float64)
    n = s_new.shape[0]
    m = s_id.shape[0]
    diff = s_new[:, None] - s_id[None, :]
    psi = np.ones(diff.shape, dtype=np.float64)
    psi[diff < 0.0] = 0.0
    psi[diff == 0.0] = 0.5
    v10 = psi.mean(axis=1)
    v01 = psi.mean(axis=0)
    gamma = float(v10.mean())
    se = float(np.sqrt(v10.var(ddof=1) / n + v01.var(ddof=1) / m))
    y = np.concatenate([np.ones(n), np.zeros(m)])
    auc = float(roc_auc_score(y, np.concatenate([s_new, s_id])))
    if abs(auc - gamma) > ATOL:
        return "auroc_bug"
    return gamma, se, auc


def fmt(value) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.16g}"


def write_csv(name: str, columns: tuple[str, ...], rows: list[dict]) -> None:
    path = OUT / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key, "")) for key in columns})


def score_pair(
    x_id_eval: np.ndarray,
    x_new: np.ndarray,
    mu_fit: np.ndarray,
    mu_new: np.ndarray,
    sigma: np.ndarray | None,
) -> dict:
    d_id = radii(x_id_eval, mu_fit, sigma)
    d_new = radii(x_new, mu_new, sigma)
    if d_id is None or d_new is None:
        return {"flag": "sigma_not_pd"}
    got = tie_gamma(d_new, d_id)
    if isinstance(got, str):
        return {"flag": got}
    gamma, se, _auc = got
    mapped = x_new - mu_new + mu_fit
    d_post = radii(mapped, mu_fit, sigma)
    d_id_post = radii(x_id_eval, mu_fit, sigma)
    if d_post is None or d_id_post is None:
        return {"flag": "sigma_not_pd"}
    got_post = tie_gamma(d_post, d_id_post)
    if isinstance(got_post, str):
        return {"flag": got_post}
    gamma_post, _se_post, _ = got_post
    g_new = -0.5 * d_new**2
    g_id = -0.5 * d_id**2
    # Scores are already inlier; feed as radii through the same -d map.
    got_g = tie_gamma(-g_new, -g_id)
    if isinstance(got_g, str):
        return {"flag": got_g}
    g_gamma, _, _ = got_g
    gaussian_gap = abs(g_gamma - gamma)
    flag = ""
    if gaussian_gap > ATOL:
        flag = "identity_bug"
    ci_low = gamma - 1.96 * se
    ci_high = gamma + 1.96 * se
    return {
        "gamma": gamma,
        "delong_se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_contains_half": ci_low <= 0.5 <= ci_high,
        "abs_auroc_minus_gamma": abs(gamma_post - gamma),
        "gaussian_gap": gaussian_gap,
        "flag": flag,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    real_rows: list[dict] = []
    null_rows: list[dict] = []
    notes: list[str] = [
        "# Disjoint-fit radii",
        "",
        "CPU only. Saved matrices only. No ridge. No re-embed.",
        "",
        "Real protocol: RandomState(0) permutation of X_id; A=first half, B=second half.",
        "Fit (μ, Σ_LW) on A. μ_new = column mean of full X_new. ID radii on B; new radii on X_new.",
        "Euclidean uses Σ=I with the same centers.",
        "",
        "Matched null: same RandomState(0) permutation of X_id;",
        "A=first 1000 (fit), B=next 500 (ID eval), C=last 500 (pseudo-new). All disjoint.",
        "This matches the real-cell fit size (1000). Eval sizes are smaller than the real new cloud.",
        "",
        "The earlier half/half null in null_split.csv fits and scores on overlapping roles;",
        "it is not overwritten. These files are the comparable pair.",
        "",
    ]

    for cell in CELL_ORDER:
        directory = ROOT / RELATIVE[cell]
        if not (directory / "X_id.npy").is_file() or not (directory / "X_new.npy").is_file():
            for metric in ("ledoit_wolf", "euclidean"):
                real_rows.append(
                    {
                        "cell": cell,
                        "metric": metric,
                        "n_fit": "",
                        "n_id_eval": "",
                        "n_new": "",
                        "d": "",
                        "d_over_n_fit": "",
                        "published_lw": PUBLISHED_LW[cell],
                        "gamma": "",
                        "delong_se": "",
                        "ci_low": "",
                        "ci_high": "",
                        "ci_contains_half": "",
                        "abs_auroc_minus_gamma": "",
                        "gaussian_gap": "",
                        "flag": "matrices_absent",
                    }
                )
                null_rows.append(
                    {
                        "cell": cell,
                        "metric": metric,
                        "n_fit": "",
                        "n_id_eval": "",
                        "n_new": "",
                        "d": "",
                        "d_over_n_fit": "",
                        "gamma": "",
                        "delong_se": "",
                        "ci_low": "",
                        "ci_high": "",
                        "ci_contains_half": "",
                        "flag": "matrices_absent",
                    }
                )
            notes.append(f"## {cell}")
            notes.append("")
            notes.append("flags: matrices_absent")
            notes.append("")
            continue

        x_id, x_new = load_pair(directory)
        n = x_id.shape[0]
        d = x_id.shape[1]
        if n < 4:
            stop(f"{cell}: n_id={n} is too small for A/B/C")
        order = np.random.RandomState(0).permutation(n)
        # Real: 1000/1000 when n=2000.
        if n % 2 == 1:
            order_ab = order[:-1]
        else:
            order_ab = order
        half = order_ab.shape[0] // 2
        idx_a = order_ab[:half]
        idx_b = order_ab[half:]
        x_a = x_id[idx_a]
        x_b = x_id[idx_b]
        mu_a = x_a.mean(axis=0)
        mu_new = x_new.mean(axis=0)
        sigma = fit_lw(x_a)
        n_fit = int(x_a.shape[0])
        n_id_eval = int(x_b.shape[0])
        n_new = int(x_new.shape[0])
        d_over = float(d) / float(n_fit)

        cell_flags: list[str] = []
        real_lw_gamma = None
        for metric, sigma_use in (("ledoit_wolf", sigma), ("euclidean", None)):
            row = {
                "cell": cell,
                "metric": metric,
                "n_fit": n_fit,
                "n_id_eval": n_id_eval,
                "n_new": n_new,
                "d": d,
                "d_over_n_fit": d_over,
                "published_lw": PUBLISHED_LW[cell],
            }
            if metric == "ledoit_wolf" and sigma is None:
                row.update(
                    {
                        "gamma": "",
                        "delong_se": "",
                        "ci_low": "",
                        "ci_high": "",
                        "ci_contains_half": "",
                        "abs_auroc_minus_gamma": "",
                        "gaussian_gap": "",
                        "flag": "sigma_not_pd",
                    }
                )
                cell_flags.append("sigma_not_pd")
            else:
                scored = score_pair(x_b, x_new, mu_a, mu_new, sigma_use)
                if "flag" in scored and scored["flag"] and "gamma" not in scored:
                    row.update(
                        {
                            "gamma": "",
                            "delong_se": "",
                            "ci_low": "",
                            "ci_high": "",
                            "ci_contains_half": "",
                            "abs_auroc_minus_gamma": "",
                            "gaussian_gap": "",
                            "flag": scored["flag"],
                        }
                    )
                    cell_flags.append(scored["flag"])
                else:
                    row.update(scored)
                    if metric == "ledoit_wolf":
                        real_lw_gamma = float(scored["gamma"])
                    if scored.get("flag"):
                        cell_flags.append(scored["flag"])
            real_rows.append(row)

        # Matched null: A=1000, B=500, C=500 from the same permutation.
        if n < 2000:
            stop(f"{cell}: n_id={n} is below 2000; refusing a different null split")
        idx_fit = order[:1000]
        idx_id = order[1000:1500]
        idx_new = order[1500:2000]
        null_a = x_id[idx_fit]
        null_b = x_id[idx_id]
        null_c = x_id[idx_new]
        mu_fit = null_a.mean(axis=0)
        mu_c = null_c.mean(axis=0)
        sigma_null = fit_lw(null_a)
        null_lw_gamma = None
        for metric, sigma_use in (("ledoit_wolf", sigma_null), ("euclidean", None)):
            row = {
                "cell": cell,
                "metric": metric,
                "n_fit": 1000,
                "n_id_eval": 500,
                "n_new": 500,
                "d": d,
                "d_over_n_fit": float(d) / 1000.0,
            }
            if metric == "ledoit_wolf" and sigma_null is None:
                row.update(
                    {
                        "gamma": "",
                        "delong_se": "",
                        "ci_low": "",
                        "ci_high": "",
                        "ci_contains_half": "",
                        "flag": "sigma_not_pd",
                    }
                )
                cell_flags.append("null:sigma_not_pd")
            else:
                scored = score_pair(null_b, null_c, mu_fit, mu_c, sigma_use)
                if "gamma" not in scored:
                    row.update(
                        {
                            "gamma": "",
                            "delong_se": "",
                            "ci_low": "",
                            "ci_high": "",
                            "ci_contains_half": "",
                            "flag": scored.get("flag", ""),
                        }
                    )
                    cell_flags.append("null:" + scored.get("flag", "failed"))
                else:
                    row.update(
                        {
                            "gamma": scored["gamma"],
                            "delong_se": scored["delong_se"],
                            "ci_low": scored["ci_low"],
                            "ci_high": scored["ci_high"],
                            "ci_contains_half": scored["ci_contains_half"],
                            "flag": scored.get("flag", ""),
                        }
                    )
                    if metric == "ledoit_wolf":
                        null_lw_gamma = float(scored["gamma"])
                    if scored.get("flag"):
                        cell_flags.append("null:" + scored["flag"])
            null_rows.append(row)

        notes.append(f"## {cell}")
        notes.append("")
        notes.append(f"published_lw: {PUBLISHED_LW[cell]}")
        notes.append(f"disjoint_real_lw: {real_lw_gamma}")
        notes.append(f"matched_null_lw: {null_lw_gamma}")
        notes.append(f"n_fit: {n_fit}")
        notes.append(f"d_over_n_fit: {d_over:.6g}")
        if real_lw_gamma is not None and null_lw_gamma is not None:
            notes.append(
                "relation_to_this_cell_null: "
                + (
                    "below"
                    if real_lw_gamma < null_lw_gamma
                    else "above"
                    if real_lw_gamma > null_lw_gamma
                    else "equal"
                )
            )
        if cell_flags:
            notes.append("flags: " + ", ".join(dict.fromkeys(cell_flags)))
        else:
            notes.append("flags: none")
        notes.append("")
        print(
            f"{cell} real_lw={real_lw_gamma} null_lw={null_lw_gamma}",
            flush=True,
        )

    write_csv(
        "disjoint_eval.csv",
        (
            "cell",
            "metric",
            "n_fit",
            "n_id_eval",
            "n_new",
            "d",
            "d_over_n_fit",
            "published_lw",
            "gamma",
            "delong_se",
            "ci_low",
            "ci_high",
            "ci_contains_half",
            "abs_auroc_minus_gamma",
            "gaussian_gap",
            "flag",
        ),
        real_rows,
    )
    write_csv(
        "null_disjoint.csv",
        (
            "cell",
            "metric",
            "n_fit",
            "n_id_eval",
            "n_new",
            "d",
            "d_over_n_fit",
            "gamma",
            "delong_se",
            "ci_low",
            "ci_high",
            "ci_contains_half",
            "flag",
        ),
        null_rows,
    )

    # Summary interval over matched null LW gammas.
    null_lw = [float(r["gamma"]) for r in null_rows if r["metric"] == "ledoit_wolf" and r.get("gamma") != ""]
    real_lw = [float(r["gamma"]) for r in real_rows if r["metric"] == "ledoit_wolf" and r.get("gamma") != ""]
    notes.append("## Summary")
    notes.append("")
    if null_lw:
        notes.append(f"matched_null_lw_min: {min(null_lw):.16g}")
        notes.append(f"matched_null_lw_max: {max(null_lw):.16g}")
    if real_lw:
        notes.append(f"disjoint_real_lw_min: {min(real_lw):.16g}")
        notes.append(f"disjoint_real_lw_max: {max(real_lw):.16g}")
    notes.append("")
    notes.append("No ranking of metrics. Theorem 1 is untouched.")
    notes.append("Published LW numbers are left as published.")
    notes.append("")
    (OUT / "DISJOINT.md").write_text("\n".join(notes), encoding="utf-8")
    print("wrote", OUT / "disjoint_eval.csv", OUT / "null_disjoint.csv", OUT / "DISJOINT.md")


if __name__ == "__main__":
    main()
