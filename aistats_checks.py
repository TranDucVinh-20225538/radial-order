"""CPU checks on the five saved clouds. Does not rewrite those npy files."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, solve, sqrtm
from scipy.stats import norm
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

ROOT = Path("/data2/cmdir/home/toandq/radial-order")
OUT = ROOT / "aistats-checks"
ATOL = 1e-6
REPLAY_ATOL = 1e-3

CELL_ORDER = ("dino_derm", "dino_path", "clip_derm", "clip_path", "dino_iwild")
METRICS = ("ledoit_wolf", "euclidean")
PUBLISHED = {
    "dino_derm": {"ledoit_wolf": 0.0155, "euclidean": 0.661},
    "dino_path": {"ledoit_wolf": 0.2235, "euclidean": 0.583},
    "clip_derm": {"ledoit_wolf": 0.00504, "euclidean": 0.5014},
    "clip_path": {"ledoit_wolf": 0.1108, "euclidean": 0.7568},
    "dino_iwild": {"ledoit_wolf": 0.1427, "euclidean": 0.4314},
}
EXPECTED_D = {
    "dino_derm": 768,
    "dino_path": 768,
    "clip_derm": 512,
    "clip_path": 512,
    "dino_iwild": 768,
}
RELATIVE = {
    "dino_derm": Path("reembed/dermatology"),
    "dino_path": Path("reembed/pathology"),
    "clip_derm": Path("clip/dermatology"),
    "clip_path": Path("clip/pathology"),
    "dino_iwild": Path("iwildcam"),
}


def write_stop(message: str, paths: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# Checks", "", message, ""]
    for path in paths:
        lines.append(path)
    lines.append("")
    (OUT / "CHECKS.md").write_text("\n".join(lines), encoding="utf-8")
    raise SystemExit(message)


def find_pairs() -> list[Path]:
    found = []
    for path in ROOT.rglob("X_id.npy"):
        if ".venv" in path.parts or "aistats-checks" in path.parts:
            continue
        if (path.parent / "X_new.npy").is_file():
            found.append(path.parent.resolve())
    return sorted(found)


def sidecar_ok() -> bool:
    reembed = (ROOT / "REEMBED.md").read_text(encoding="utf-8")
    clip = (ROOT / "clip" / "CLIP.md").read_text(encoding="utf-8")
    iwild = (ROOT / "iwildcam" / "IWILDCAM.md").read_text(encoding="utf-8")
    return (
        "dinov2_vitb14" in reembed
        and "## dermatology" in reembed
        and "## pathology" in reembed
        and "ViT-B/16" in clip
        and "## dermatology" in clip
        and "## pathology" in clip
        and "dinov2_vitb14" in iwild
        and 'get_subset("train")' in iwild
        and 'get_subset("test")' in iwild
    )


def match_cells() -> dict[str, Path]:
    found = find_pairs()
    if not sidecar_ok():
        write_stop("sidecar does not name the encoder and the cloud", [str(p) for p in found])
    expected = {cell: (ROOT / rel).resolve() for cell, rel in RELATIVE.items()}
    if set(found) != set(expected.values()):
        write_stop(
            "saved pairs are not the five uniquely named cells",
            [str(p) for p in found],
        )
    return expected


def load_pair(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    x_id = np.load(directory / "X_id.npy", mmap_mode="r")
    x_new = np.load(directory / "X_new.npy", mmap_mode="r")
    x_id = np.array(x_id, dtype=np.float64, copy=True)
    x_new = np.array(x_new, dtype=np.float64, copy=True)
    return x_id, x_new


def fit_lw(x_id: np.ndarray) -> np.ndarray:
    fitted = LedoitWolf(assume_centered=False).fit(x_id)
    return np.asarray(fitted.covariance_, dtype=np.float64)


def is_pd(sigma: np.ndarray) -> bool:
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        return False
    if not np.isfinite(sigma).all():
        return False
    evals = np.linalg.eigvalsh(sigma)
    return bool(np.isfinite(evals).all() and np.all(evals > 0.0))


def radii(x: np.ndarray, center: np.ndarray, sigma: np.ndarray) -> np.ndarray | None:
    diff = np.asarray(x, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    if sigma is None:
        quad = np.sum(diff * diff, axis=1)
    else:
        try:
            solved = solve(sigma, diff.T, assume_a="pos")
        except np.linalg.LinAlgError:
            return None
        quad = np.einsum("ij,ji->i", diff, solved)
    if not np.isfinite(quad).all() or np.any(quad < -1e-8):
        return None
    return np.sqrt(np.maximum(quad, 0.0))


def tie_gamma(d_new: np.ndarray, d_id: np.ndarray) -> tuple[float, float, float]:
    """Return kernel Γ, DeLong SE, and sklearn AUROC of s=-d."""
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
    return gamma, se, auc


def qr_positive(g: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(g)
    sign = np.sign(np.diag(r))
    sign[sign == 0.0] = 1.0
    return q * sign


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


def metric_sigma(kind: str, sigma_lw: np.ndarray) -> np.ndarray | None:
    if kind == "euclidean":
        return None
    return sigma_lw


def pair_radii(x_id, x_new, mu, mu_new, sigma) -> tuple[np.ndarray, np.ndarray] | None:
    d_id = radii(x_id, mu, sigma)
    d_new = radii(x_new, mu_new, sigma)
    if d_id is None or d_new is None:
        return None
    return d_id, d_new


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    directories = match_cells()
    gen = np.random.Generator(np.random.PCG64(0))
    rs = np.random.RandomState(0)

    loaded = {}
    for cell in CELL_ORDER:
        x_id, x_new = load_pair(directories[cell])
        if x_id.ndim != 2 or x_new.ndim != 2 or x_id.shape[1] != x_new.shape[1]:
            write_stop(f"{cell} matrix shapes disagree", [str(directories[cell])])
        if x_id.shape[1] != EXPECTED_D[cell]:
            write_stop(
                f"{cell} width {x_id.shape[1]} failed the encoder width check",
                [str(directories[cell])],
            )
        loaded[cell] = (x_id, x_new, x_id.mean(axis=0), x_new.mean(axis=0), fit_lw(x_id))
        print(f"loaded {cell} {x_id.shape}", flush=True)

    cell_rows: list[dict] = []
    null_rows: list[dict] = []
    phi_rows: list[dict] = []
    path_rows: list[dict] = []
    extra_rows: list[dict] = []
    flags: dict[str, list[str]] = {cell: [] for cell in CELL_ORDER}
    blocked: set[str] = set()
    stored = {}

    for cell in CELL_ORDER:
        x_id, x_new, mu, mu_new, sigma_lw = loaded[cell]
        if not is_pd(sigma_lw):
            for kind in METRICS:
                cell_rows.append(blank_cell(cell, kind, x_id, x_new, "sigma_not_pd"))
            flags[cell].append("sigma_not_pd")
            blocked.add(cell)
            continue
        identity_bad = False
        metric_pack = {}
        for kind in METRICS:
            sigma = metric_sigma(kind, sigma_lw)
            pair = pair_radii(x_id, x_new, mu, mu_new, sigma)
            if pair is None:
                cell_rows.append(blank_cell(cell, kind, x_id, x_new, "sigma_not_pd"))
                flags[cell].append(f"{kind}:sigma_not_pd")
                identity_bad = True
                continue
            d_id, d_new = pair
            gamma, se, auc = tie_gamma(d_new, d_id)
            row_flags = []
            if abs(auc - gamma) > ATOL:
                row_flags.append("auroc_bug")
                identity_bad = True
            mapped = x_new - mu_new + mu
            d_post = radii(mapped, mu, sigma)
            d_id_post = radii(x_id, mu, sigma)
            if d_post is None or d_id_post is None:
                row_flags.append("sigma_not_pd")
                identity_bad = True
                gamma_post = None
            else:
                gamma_post, _, auc_post = tie_gamma(d_post, d_id_post)
                if abs(auc_post - gamma_post) > ATOL:
                    row_flags.append("auroc_bug")
                    identity_bad = True
            abs_gap = None if gamma_post is None else abs(gamma_post - gamma)
            g_new = -0.5 * d_new ** 2
            g_id = -0.5 * d_id ** 2
            # Gaussian score is already an inlier score, so compare it directly.
            g_gamma, _, g_auc = tie_gamma(-g_new, -g_id)
            if abs(g_auc - g_gamma) > ATOL:
                row_flags.append("auroc_bug")
                identity_bad = True
            gaussian_gap = abs(g_gamma - gamma)
            if gaussian_gap > ATOL:
                row_flags.append("identity_bug")
                identity_bad = True
            published = PUBLISHED[cell][kind]
            if abs(gamma - published) > REPLAY_ATOL:
                row_flags.append("replay_mismatch")
            for name in row_flags:
                flags[cell].append(f"{kind}:{name}")
            ci_low = gamma - 1.96 * se
            ci_high = gamma + 1.96 * se
            record = {
                "cell": cell,
                "metric": kind,
                "n_id": x_id.shape[0],
                "n_new": x_new.shape[0],
                "d": x_id.shape[1],
                "published": published,
                "replay": gamma,
                "delong_se": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_contains_half": ci_low <= 0.5 <= ci_high,
                "abs_auroc_minus_gamma": abs_gap,
                "gaussian_gap": gaussian_gap,
                "orth_residual": "",
                "orth_gap": "",
                "flag": ";".join(row_flags),
                "d_id": d_id,
                "d_new": d_new,
                "sigma": sigma,
            }
            metric_pack[kind] = record
        if identity_bad:
            blocked.add(cell)
        stored[cell] = metric_pack
        print(f"replay {cell}", flush=True)

    # Section 2 consumes the generator once per cell, including blocked cells'
    # draws only when the cell is run. Blocked cells are skipped.
    orth = {}
    for cell in CELL_ORDER:
        x_id, x_new, mu, mu_new, sigma_lw = loaded[cell]
        d = x_id.shape[1]
        if cell in blocked:
            continue
        g = gen.standard_normal((d, d))
        b = qr_positive(g)
        try:
            root = np.asarray(sqrtm(sigma_lw))
        except Exception as exc:
            flags[cell].append(f"ledoit_wolf:{type(exc).__name__}: {exc}")
            orth[cell] = None
            continue
        if np.iscomplexobj(root):
            root = np.real(root)
        try:
            a_map = root @ b @ np.linalg.inv(root)
        except np.linalg.LinAlgError as exc:
            flags[cell].append(f"ledoit_wolf:{type(exc).__name__}: {exc}")
            orth[cell] = None
            continue
        residual = float(np.max(np.abs(a_map @ sigma_lw @ a_map.T - sigma_lw)))
        mapped = (x_new - mu_new) @ a_map.T + mu
        d_post = radii(mapped, mu, sigma_lw)
        d_id = stored[cell]["ledoit_wolf"]["d_id"]
        gap = ""
        row_flag = []
        if d_post is None:
            row_flag.append("sigma_not_pd")
        else:
            gamma_a, _, _ = tie_gamma(d_post, d_id)
            gap = abs(gamma_a - stored[cell]["ledoit_wolf"]["replay"])
        if residual > ATOL:
            row_flag.append("orth_residual")
        if row_flag:
            flags[cell].append("ledoit_wolf:" + ";".join(row_flag))
        orth[cell] = (residual, gap, row_flag)
        print(f"orth {cell} residual {residual:.3g}", flush=True)

    for cell in CELL_ORDER:
        pack = stored.get(cell, {})
        for kind in METRICS:
            record = pack.get(kind)
            if record is None:
                continue
            if kind == "ledoit_wolf" and cell in orth and orth[cell] is not None:
                residual, gap, row_flag = orth[cell]
                record["orth_residual"] = residual
                record["orth_gap"] = gap
                if row_flag:
                    record["flag"] = ";".join(filter(None, [record["flag"], *row_flag]))
            public = {k: v for k, v in record.items() if k not in {"d_id", "d_new", "sigma"}}
            cell_rows.append(public)

    # Section 3.
    for cell in CELL_ORDER:
        if cell in blocked:
            continue
        x_id = loaded[cell][0]
        n = x_id.shape[0]
        order = np.random.RandomState(0).permutation(n)
        if n % 2 == 1:
            order = order[:-1]
        half = order.shape[0] // 2
        null_id = x_id[order[:half]]
        null_new = x_id[order[half:]]
        mu_id = null_id.mean(axis=0)
        mu_new = null_new.mean(axis=0)
        sigma = fit_lw(null_id)
        spec = None
        if not is_pd(sigma):
            flags[cell].append("null:sigma_not_pd")
        else:
            centered = null_new - mu_new
            s_new = (centered.T @ centered) / float(null_new.shape[0])
            try:
                lambdas = np.sort(np.asarray(eigh(s_new, sigma, eigvals_only=True, check_finite=True), dtype=np.float64))
            except Exception as exc:
                flags[cell].append(f"null:{type(exc).__name__}: {exc}")
                lambdas = None
            if lambdas is not None:
                spec = {
                    "n_lt_1": int(np.sum(lambdas < 1.0)),
                    "n_eq_1": int(np.sum(lambdas == 1.0)),
                    "n_gt_1": int(np.sum(lambdas > 1.0)),
                    "lambda_min": float(lambdas[0]),
                    "lambda_max": float(lambdas[-1]),
                }
        for kind in METRICS:
            row = {"cell": cell, "metric": kind, "n_half": int(null_id.shape[0])}
            if not is_pd(sigma):
                row.update({"gamma": "", "delong_se": "", "flag": "sigma_not_pd"})
            else:
                pair = pair_radii(null_id, null_new, mu_id, mu_new, metric_sigma(kind, sigma))
                if pair is None:
                    row.update({"gamma": "", "delong_se": ""})
                    flags[cell].append(f"null:{kind}:sigma_not_pd")
                else:
                    d_id, d_new = pair
                    gamma, se, auc = tie_gamma(d_new, d_id)
                    row.update({"gamma": gamma, "delong_se": se})
                    if abs(auc - gamma) > ATOL:
                        flags[cell].append(f"null:{kind}:auroc_bug")
            if kind == "ledoit_wolf" and spec is not None:
                row.update(spec)
            else:
                row.update({"n_lt_1": "", "n_eq_1": "", "n_gt_1": "", "lambda_min": "", "lambda_max": ""})
            null_rows.append(row)
        print(f"null {cell}", flush=True)

    # Section 4. Generator continues after the orthogonal draws.
    for cell in CELL_ORDER:
        if cell in blocked:
            continue
        x_id, x_new, mu, mu_new, sigma_lw = loaded[cell]
        d = x_id.shape[1]
        n_new = x_new.shape[0]
        n_id = x_id.shape[0]
        centered = x_new - mu_new
        s_new = (centered.T @ centered) / float(n_new)
        for kind in METRICS:
            record = stored[cell][kind]
            sigma = metric_sigma(kind, sigma_lw)
            if sigma is None:
                lambdas = np.sort(np.linalg.eigvalsh(s_new))
                trace = float(np.trace(s_new))
            else:
                try:
                    lambdas = np.sort(np.asarray(eigh(s_new, sigma, eigvals_only=True, check_finite=True), dtype=np.float64))
                    trace = float(np.trace(solve(sigma, s_new, assume_a="pos")))
                except Exception as exc:
                    flags[cell].append(f"phi:{kind}:{type(exc).__name__}: {exc}")
                    phi_rows.append({"cell": cell, "metric": kind, "flag": type(exc).__name__})
                    # Keep the generator aligned: this row would have drawn.
                    gen.standard_normal((n_new, d))
                    gen.standard_normal((n_id, d))
                    continue
            sum_lambda = float(np.sum(lambdas))
            sum_sq = float(np.sum(lambdas ** 2))
            row = {
                "cell": cell,
                "metric": kind,
                "trace": trace,
                "sum_lambda_sq": sum_sq,
            }
            if abs(trace - sum_lambda) > ATOL:
                row["flag"] = "trace_mismatch"
                flags[cell].append(f"phi:{kind}:trace_mismatch")
                gen.standard_normal((n_new, d))
                gen.standard_normal((n_id, d))
                phi_rows.append(row)
                continue
            denom = np.sqrt(2.0 * d + 2.0 * sum_sq)
            phi = float(norm.cdf((d - sum_lambda) / denom))
            z_draw = gen.standard_normal((n_new, d))
            w_draw = gen.standard_normal((n_id, d))
            r2_new = (z_draw ** 2) @ lambdas
            r2_id = np.sum(w_draw ** 2, axis=1)
            gamma_s, se_s, auc_s = tie_gamma(r2_new, r2_id)
            if abs(auc_s - gamma_s) > ATOL:
                flags[cell].append(f"phi:{kind}:auroc_bug")
            se_real = record["delong_se"]
            gamma_real = record["replay"]
            separated = abs(gamma_s - gamma_real) > 1.96 * np.sqrt(se_real ** 2 + se_s ** 2)
            row.update(
                {
                    "phi": phi,
                    "gamma_real": gamma_real,
                    "gamma_surrogate": gamma_s,
                    "se_real": se_real,
                    "se_surrogate": se_s,
                    "phi_minus_surrogate": phi - gamma_s,
                    "surrogate_minus_real": gamma_s - gamma_real,
                    "surrogate_separated": bool(separated),
                    "flag": "",
                }
            )
            phi_rows.append(row)
        print(f"phi {cell}", flush=True)

    # Section 5.
    for cell in CELL_ORDER:
        if cell in blocked:
            continue
        x_id, x_new, mu, mu_new, sigma_lw = loaded[cell]
        d = sigma_lw.shape[0]
        eye = np.eye(d)
        scale = float(np.trace(sigma_lw) / d)
        gammas = []
        path_flag = []
        for k in range(21):
            t = k / 20.0
            sigma_t = (1.0 - t) * sigma_lw + t * scale * eye
            pair = pair_radii(x_id, x_new, mu, mu_new, sigma_t)
            if pair is None:
                path_flag.append("sigma_not_pd")
                gammas.append(None)
            else:
                d_id, d_new = pair
                gamma, _, auc = tie_gamma(d_new, d_id)
                if abs(auc - gamma) > ATOL:
                    path_flag.append("auroc_bug")
                gammas.append(gamma)
        lw = stored[cell]["ledoit_wolf"]["replay"]
        eu = stored[cell]["euclidean"]["replay"]
        g0 = gammas[0]
        g1 = gammas[-1]
        if g0 is None or g1 is None or abs(g0 - lw) > ATOL or abs(g1 - eu) > ATOL:
            path_flag.append("path_endpoint_bug")
        if path_flag:
            flags[cell].append("path:" + ";".join(dict.fromkeys(path_flag)))
        finite = [g for g in gammas if g is not None]
        sides = [0.0 if g is None or g == 0.5 else np.sign(g - 0.5) for g in gammas]
        crossings = 0
        for left, right in zip(sides, sides[1:]):
            if left != 0.0 and right != 0.0 and left != right:
                crossings += 1
        half_lw = bool(stored[cell]["ledoit_wolf"]["ci_contains_half"])
        half_eu = bool(stored[cell]["euclidean"]["ci_contains_half"])
        side_change = (not half_lw) and (not half_eu) and crossings > 0
        for k, gamma in enumerate(gammas):
            path_rows.append(
                {
                    "cell": cell,
                    "t": k / 20.0,
                    "gamma": "" if gamma is None else gamma,
                    "endpoint_lw": "" if g0 is None else abs(g0 - lw),
                    "endpoint_euclid": "" if g1 is None else abs(g1 - eu),
                    "n_crossings": crossings,
                    "crossing_is_a_side_change": side_change,
                    "flag": ";".join(dict.fromkeys(path_flag)),
                }
            )
        del finite
        print(f"path {cell} crossings {crossings}", flush=True)

    # Section 6.
    for cell in CELL_ORDER:
        x_id, x_new, mu, mu_new, sigma_lw = loaded[cell]
        diag = np.diag(sigma_lw).copy()
        if not np.all(diag > 0.0):
            extra_rows.append(blank_extra(cell, "diagonal", "sigma_not_pd"))
            flags[cell].append("diagonal:sigma_not_pd")
        else:
            sigma_d = np.diag(diag)
            extra_rows.append(extra_metric(cell, "diagonal", x_id, x_new, mu, mu_new, sigma_d, flags))
        nrm_id = np.linalg.norm(x_id, axis=1)
        nrm_new = np.linalg.norm(x_new, axis=1)
        if np.any(nrm_id == 0.0) or np.any(nrm_new == 0.0):
            extra_rows.append(blank_extra(cell, "sphere_ledoit_wolf", "zero_row"))
            flags[cell].append("sphere_ledoit_wolf:zero_row")
        else:
            z_id = x_id / nrm_id[:, None]
            z_new = x_new / nrm_new[:, None]
            sigma_s = fit_lw(z_id)
            if not is_pd(sigma_s):
                extra_rows.append(blank_extra(cell, "sphere_ledoit_wolf", "sigma_not_pd"))
                flags[cell].append("sphere_ledoit_wolf:sigma_not_pd")
            else:
                extra_rows.append(
                    extra_metric(
                        cell,
                        "sphere_ledoit_wolf",
                        z_id,
                        z_new,
                        z_id.mean(axis=0),
                        z_new.mean(axis=0),
                        sigma_s,
                        flags,
                    )
                )
        print(f"extra {cell}", flush=True)

    write_csv(
        "cells.csv",
        (
            "cell",
            "metric",
            "n_id",
            "n_new",
            "d",
            "published",
            "replay",
            "delong_se",
            "ci_low",
            "ci_high",
            "ci_contains_half",
            "abs_auroc_minus_gamma",
            "gaussian_gap",
            "orth_residual",
            "orth_gap",
            "flag",
        ),
        cell_rows,
    )
    write_csv(
        "null_split.csv",
        (
            "cell",
            "metric",
            "n_half",
            "gamma",
            "delong_se",
            "n_lt_1",
            "n_eq_1",
            "n_gt_1",
            "lambda_min",
            "lambda_max",
        ),
        null_rows,
    )
    write_csv(
        "phi.csv",
        (
            "cell",
            "metric",
            "trace",
            "sum_lambda_sq",
            "phi",
            "gamma_real",
            "gamma_surrogate",
            "se_real",
            "se_surrogate",
            "phi_minus_surrogate",
            "surrogate_minus_real",
            "surrogate_separated",
        ),
        phi_rows,
    )
    write_csv(
        "path.csv",
        (
            "cell",
            "t",
            "gamma",
            "endpoint_lw",
            "endpoint_euclid",
            "n_crossings",
            "crossing_is_a_side_change",
            "flag",
        ),
        path_rows,
    )
    write_csv(
        "extra_metrics.csv",
        (
            "cell",
            "metric",
            "gamma",
            "delong_se",
            "ci_contains_half",
            "abs_auroc_minus_gamma",
            "flag",
        ),
        extra_rows,
    )
    lines = ["# Checks", ""]
    for cell in CELL_ORDER:
        fired = flags[cell]
        lines.append(f"## {cell}")
        lines.append("")
        lines.append("flags: none" if not fired else "flags: " + ", ".join(fired))
        lines.append("")
    (OUT / "CHECKS.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT, flush=True)


def blank_cell(cell, kind, x_id, x_new, flag) -> dict:
    return {
        "cell": cell,
        "metric": kind,
        "n_id": x_id.shape[0],
        "n_new": x_new.shape[0],
        "d": x_id.shape[1],
        "published": PUBLISHED[cell][kind],
        "replay": "",
        "delong_se": "",
        "ci_low": "",
        "ci_high": "",
        "ci_contains_half": "",
        "abs_auroc_minus_gamma": "",
        "gaussian_gap": "",
        "orth_residual": "",
        "orth_gap": "",
        "flag": flag,
    }


def blank_extra(cell, metric, flag) -> dict:
    return {
        "cell": cell,
        "metric": metric,
        "gamma": "",
        "delong_se": "",
        "ci_contains_half": "",
        "abs_auroc_minus_gamma": "",
        "flag": flag,
    }


def extra_metric(cell, name, x_id, x_new, mu, mu_new, sigma, flags) -> dict:
    pair = pair_radii(x_id, x_new, mu, mu_new, sigma)
    if pair is None:
        flags[cell].append(f"{name}:sigma_not_pd")
        return blank_extra(cell, name, "sigma_not_pd")
    d_id, d_new = pair
    gamma, se, auc = tie_gamma(d_new, d_id)
    row_flags = []
    if abs(auc - gamma) > ATOL:
        row_flags.append("auroc_bug")
    mapped = x_new - mu_new + mu
    d_post = radii(mapped, mu, sigma)
    d_id_post = radii(x_id, mu, sigma)
    abs_gap = ""
    if d_post is None or d_id_post is None:
        row_flags.append("sigma_not_pd")
    else:
        gamma_post, _, auc_post = tie_gamma(d_post, d_id_post)
        if abs(auc_post - gamma_post) > ATOL:
            row_flags.append("auroc_bug")
        abs_gap = abs(gamma_post - gamma)
    if row_flags:
        flags[cell].append(name + ":" + ";".join(row_flags))
    ci_low = gamma - 1.96 * se
    ci_high = gamma + 1.96 * se
    return {
        "cell": cell,
        "metric": name,
        "gamma": gamma,
        "delong_se": se,
        "ci_contains_half": ci_low <= 0.5 <= ci_high,
        "abs_auroc_minus_gamma": abs_gap,
        "flag": ";".join(row_flags),
    }


if __name__ == "__main__":
    main()
