"""AISTATS round-3 analyses. Writes only under aistats-checks/round3/ and export_embeddings/clip_iwild_*.npy."""

from __future__ import annotations

import csv
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh, solve
from scipy.stats import kurtosis, norm
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

ROOT = Path("/data2/cmdir/home/toandq/radial-order")
EMB = ROOT / "export_embeddings"
OUT = ROOT / "aistats-checks" / "round3"
WILDS_ROOT = Path("/data2/cmdir/home/toandq/DST-Skin/data/raw/wilds")
IWILD_IDX = ROOT / "iwildcam"

CELL_ORDER = ("dino_derm", "dino_path", "clip_derm", "clip_path", "dino_iwild")
SANITY_LW = (0.151962, 0.371211, 0.044637, 0.23225, 0.3464)
ATOL = 1e-6
SEEDS = tuple(range(20))
HYBRID_RNG = tuple(range(5))
Q_VALUES = (0.10, 0.25, 0.50)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str) -> None:
    print(msg, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(msg + "\n")


def load_emb(cell: str) -> tuple[np.ndarray, np.ndarray]:
    x_id = np.asarray(np.load(EMB / f"{cell}_id.npy"), dtype=np.float64)
    x_new = np.asarray(np.load(EMB / f"{cell}_new.npy"), dtype=np.float64)
    return x_id, x_new


def tie_gamma(d_new: np.ndarray, d_id: np.ndarray) -> tuple[float, float, float]:
    s_new = -np.asarray(d_new, dtype=np.float64)
    s_id = -np.asarray(d_id, dtype=np.float64)
    n, m = s_new.shape[0], s_id.shape[0]
    s_id_sorted = np.sort(s_id)
    s_new_sorted = np.sort(s_new)
    left_id = np.searchsorted(s_id_sorted, s_new, side="left")
    right_id = np.searchsorted(s_id_sorted, s_new, side="right")
    v10 = (left_id + 0.5 * (right_id - left_id)) / m
    left_new = np.searchsorted(s_new_sorted, s_id, side="left")
    right_new = np.searchsorted(s_new_sorted, s_id, side="right")
    v01 = (n - right_new + 0.5 * (right_new - left_new)) / n
    gamma = float(v10.mean())
    se = float(np.sqrt(v10.var(ddof=1) / n + v01.var(ddof=1) / m))
    y = np.concatenate([np.ones(n), np.zeros(m)])
    auc = float(roc_auc_score(y, np.concatenate([s_new, s_id])))
    return gamma, se, auc


def radii(
    x: np.ndarray,
    center: np.ndarray,
    sigma: np.ndarray | None,
    *,
    chol: tuple[np.ndarray, bool] | None = None,
) -> np.ndarray | None:
    diff = np.asarray(x, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    if sigma is None:
        quad = np.sum(diff * diff, axis=1)
    else:
        try:
            if chol is None:
                chol = cho_factor(sigma, lower=True, check_finite=True)
            solved = cho_solve(chol, diff.T)
        except np.linalg.LinAlgError:
            return None
        quad = np.einsum("ij,ji->i", diff, solved)
    if not np.isfinite(quad).all() or np.any(quad < -1e-8):
        return None
    return np.sqrt(np.maximum(quad, 0.0))


def fit_lw(a: np.ndarray) -> np.ndarray:
    return np.asarray(LedoitWolf(assume_centered=False).fit(a).covariance_, dtype=np.float64)


def l2_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(n == 0.0):
        raise ValueError("zero-norm row in l2 normalization")
    return x / n


def metric_transform(x_id: np.ndarray, x_new: np.ndarray, metric: str) -> tuple[np.ndarray, np.ndarray]:
    if metric == "sphere_LW":
        return l2_rows(x_id), l2_rows(x_new)
    return x_id, x_new


def sigma_from_a(a: np.ndarray, metric: str) -> np.ndarray | None:
    if metric == "euclidean":
        return None
    lw = fit_lw(a)
    if metric == "diag":
        return np.diag(np.diag(lw))
    return lw


def score_from_split(
    z_id: np.ndarray,
    z_new: np.ndarray,
    p: np.ndarray,
    metric: str,
    *,
    real: bool,
    mu: np.ndarray,
    sigma: np.ndarray | None,
) -> tuple[float, float, float] | None:
    if real:
        b = z_id[p[1000:]]
        mu_new = z_new.mean(axis=0)
        eval_new = z_new
    else:
        b = z_id[p[1000:1500]]
        c = z_id[p[1500:2000]]
        mu_new = c.mean(axis=0)
        eval_new = c
    chol = None if sigma is None else cho_factor(sigma, lower=True, check_finite=True)
    d_id = radii(b, mu, sigma, chol=chol)
    d_new = radii(eval_new, mu_new, sigma, chol=chol)
    if d_id is None or d_new is None:
        return None
    gamma, se, auc = tie_gamma(d_new, d_id)
    if abs(auc - gamma) > ATOL:
        return None
    mapped = eval_new - mu_new + mu
    d_post = radii(mapped, mu, sigma, chol=chol)
    d_id_post = radii(b, mu, sigma, chol=chol)
    if d_post is None or d_id_post is None:
        return None
    g_post, _, auc_p = tie_gamma(d_post, d_id_post)
    if abs(auc_p - g_post) > ATOL:
        return None
    return gamma, se, abs(g_post - gamma)


def score_disjoint(
    x_id: np.ndarray,
    x_new: np.ndarray,
    seed: int,
    metric: str,
    *,
    real: bool,
) -> tuple[float, float, float] | None:
    try:
        z_id, z_new = metric_transform(x_id, x_new, metric)
    except ValueError:
        return None
    p = np.random.RandomState(seed).permutation(2000)
    a = z_id[p[:1000]]
    mu = a.mean(axis=0)
    sigma = sigma_from_a(a, metric)
    if sigma is not None and not np.all(np.linalg.eigvalsh(sigma) > 0):
        return None
    return score_from_split(z_id, z_new, p, metric, real=real, mu=mu, sigma=sigma)


def sanity_gate() -> None:
    for cell, target in zip(CELL_ORDER, SANITY_LW):
        x_id, x_new = load_emb(cell)
        got = score_disjoint(x_id, x_new, 0, "ledoit_wolf", real=True)
        if got is None:
            raise SystemExit(f"SANITY FAIL {cell}: could not score ledoit_wolf seed 0")
        gamma, _, _ = got
        if abs(gamma - target) > ATOL:
            raise SystemExit(
                f"SANITY FAIL {cell}: disjoint LW seed 0 gamma={gamma:.16g} expected {target:.16g}"
            )
    log("sanity gate PASS")


def write_csv(name: str, columns: tuple[str, ...], rows: list[dict]) -> None:
    path = OUT / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in columns})


def fmt3(x: float) -> str:
    return f"{x:.3f}"


def fmt4(x: float) -> str:
    return f"{x:.4f}"


def _part1_seed(cell: str, metric: str, seed: int) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    x_id, x_new = load_emb(cell)
    z_id, z_new = metric_transform(x_id, x_new, metric)
    p = np.random.RandomState(seed).permutation(2000)
    a = z_id[p[:1000]]
    mu = a.mean(axis=0)
    sigma = sigma_from_a(a, metric)
    if sigma is not None and not np.all(np.linalg.eigvalsh(sigma) > 0):
        raise RuntimeError(f"part1 sigma {cell} {metric} seed={seed}")
    real = score_from_split(z_id, z_new, p, metric, real=True, mu=mu, sigma=sigma)
    nul = score_from_split(z_id, z_new, p, metric, real=False, mu=mu, sigma=sigma)
    if real is None or nul is None:
        raise RuntimeError(f"part1 failed {cell} {metric} seed={seed}")
    g, se, ig = real
    ng, nse, _ = nul
    return {
        "cell": cell,
        "metric": metric,
        "seed": seed,
        "gamma": g,
        "delong_se": se,
        "null_gamma": ng,
        "null_se": nse,
        "identity_gap": ig,
    }


def part1() -> tuple[list[dict], dict]:
    summary: dict = {}
    tasks = [(cell, metric, seed) for cell in CELL_ORDER for metric in ("diag", "sphere_LW") for seed in SEEDS]
    rows: list[dict] = []
    workers = min(32, len(tasks))
    log(f"part1 parallel workers={workers} tasks={len(tasks)}")
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_part1_seed, cell, metric, seed) for cell, metric, seed in tasks]
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % 50 == 0 or done == len(tasks):
                log(f"part1 progress {done}/{len(tasks)}")
    metric_order = {"diag": 0, "sphere_LW": 1}
    rows.sort(key=lambda r: (CELL_ORDER.index(r["cell"]), metric_order[r["metric"]], r["seed"]))
    for cell in CELL_ORDER:
        for metric in ("diag", "sphere_LW"):
            block = [r for r in rows if r["cell"] == cell and r["metric"] == metric]
            gammas = [r["gamma"] for r in block]
            nulls = [r["null_gamma"] for r in block]
            id_gaps = [r["identity_gap"] for r in block]
            g_arr = np.array(gammas)
            vs_half = np.sign(g_arr - 0.5)
            vs_half = vs_half[vs_half != 0]
            if len(vs_half) == 0:
                half_class = "mixed"
            elif np.all(vs_half > 0):
                half_class = "above_1/2"
            elif np.all(vs_half < 0):
                half_class = "below_1/2"
            else:
                half_class = "mixed"
            vs_null = [g < n for g, n in zip(gammas, nulls)]
            if all(vs_null):
                null_class = "below_own_null"
            elif not any(vs_null):
                null_class = "above_own_null"
            else:
                null_class = "mixed_vs_null"
            summary[(cell, metric)] = {
                "gamma_median": float(np.median(g_arr)),
                "gamma_min": float(np.min(g_arr)),
                "gamma_max": float(np.max(g_arr)),
                "half_class": half_class,
                "null_class": null_class,
                "identity_gap_max": float(np.max(id_gaps)),
            }
            log(f"part1 {cell} {metric}")
    write_csv(
        "extra_metrics_disjoint.csv",
        ("cell", "metric", "seed", "gamma", "delong_se", "null_gamma", "null_se", "identity_gap"),
        rows,
    )
    return rows, summary


def band_slices(d: int) -> dict[str, slice]:
    i10 = int(np.ceil(0.10 * d))
    i25 = int(np.ceil(0.25 * d))
    i50 = int(np.ceil(0.50 * d))
    return {
        "bottom_10": slice(0, i10),
        "10_25": slice(i10, i25),
        "25_50": slice(i25, i50),
        "top_50": slice(i50, d),
    }


def whiten(x: np.ndarray, mu: np.ndarray, u: np.ndarray, s: np.ndarray) -> np.ndarray:
    return ((x - mu) @ u) / np.sqrt(s)


def unwhiten(w: np.ndarray, mu: np.ndarray, u: np.ndarray, s: np.ndarray) -> np.ndarray:
    return mu + (w * np.sqrt(s)) @ u.T


def disjoint_seed0(x_id: np.ndarray, x_new: np.ndarray):
    p = np.random.RandomState(0).permutation(2000)
    a = x_id[p[:1000]]
    b = x_id[p[1000:]]
    mu = a.mean(axis=0)
    mu_new = x_new.mean(axis=0)
    sigma = fit_lw(a)
    return a, b, x_new, mu, mu_new, sigma, p


def part2() -> tuple[list[dict], list[dict], dict]:
    band_rows: list[dict] = []
    hybrid_rows: list[dict] = []
    narrative: dict[str, str] = {}
    for cell in CELL_ORDER:
        x_id, x_new = load_emb(cell)
        _a, b, xn, mu, mu_new, sigma, _p = disjoint_seed0(x_id, x_new)
        evals, u = eigh(sigma)
        if np.any(evals <= 0):
            raise SystemExit(f"part2 {cell}: sigma not PD")
        slices = band_slices(len(evals))
        wb = whiten(b, mu, u, evals)
        wn = whiten(xn, mu_new, u, evals)
        for cloud_name, w in (("B", wb), ("X_new", wn)):
            d2 = np.sum(w * w, axis=1)
            mean_d2 = float(np.mean(d2))
            for band_name, sl in slices.items():
                contrib = np.sum(w[:, sl] ** 2, axis=1)
                share = float(np.mean(contrib) / mean_d2) if mean_d2 > 0 else 0.0
                kvals = kurtosis(w[:, sl], axis=0, fisher=True, nan_policy="raise")
                band_rows.append(
                    {
                        "cell": cell,
                        "cloud": cloud_name,
                        "band": band_name,
                        "mean_d2_share": share,
                        "kurtosis_median": float(np.median(kvals)),
                        "kurtosis_p90": float(np.percentile(kvals, 90)),
                    }
                )

        d_id = radii(b, mu, sigma)
        d_new = radii(xn, mu_new, sigma)
        gamma_real, _, _ = tie_gamma(d_new, d_id)

        sb = (b - mu).T @ (b - mu) / b.shape[0]
        sn = (xn - mu_new).T @ (xn - mu_new) / xn.shape[0]
        rng = np.random.RandomState(0)
        fb = rng.multivariate_normal(mu, sb, size=b.shape[0])
        fn = rng.multivariate_normal(mu_new, sn, size=xn.shape[0])
        g_full, _, _ = tie_gamma(radii(fn, mu_new, sigma), radii(fb, mu, sigma))
        hybrid_rows.append(
            {"cell": cell, "q": "", "method": "real", "gamma_mean": gamma_real, "gamma_sd": 0.0}
        )
        hybrid_rows.append(
            {"cell": cell, "q": "", "method": "full_gaussian", "gamma_mean": g_full, "gamma_sd": 0.0}
        )

        def bottom_idx(q: float) -> np.ndarray:
            k = int(np.ceil(q * len(evals)))
            return np.arange(k)

        def apply_surrogate(wb0, wn0, band: np.ndarray, mode: str, rng_seed: int) -> tuple[np.ndarray, np.ndarray]:
            rngl = np.random.RandomState(rng_seed)
            wb1 = wb0.copy()
            wn1 = wn0.copy()
            if mode == "G_low":
                for cloud_w in (wb1, wn1):
                    sub = cloud_w[:, band]
                    if sub.shape[1] == 1:
                        draws = rngl.normal(0.0, np.std(sub[:, 0], ddof=0), size=sub.shape[0])
                        cloud_w[:, band] = draws[:, None]
                    else:
                        cov = np.cov(sub.T, bias=True)
                        draws = rngl.multivariate_normal(np.zeros(sub.shape[1]), cov, size=sub.shape[0])
                        cloud_w[:, band] = draws
            elif mode == "G_high":
                hi = np.setdiff1d(np.arange(wb0.shape[1]), band)
                for cloud_w in (wb1, wn1):
                    sub = cloud_w[:, hi]
                    if sub.shape[1] == 1:
                        draws = rngl.normal(0.0, np.std(sub[:, 0], ddof=0), size=sub.shape[0])
                        cloud_w[:, hi] = draws[:, None]
                    else:
                        cov = np.cov(sub.T, bias=True)
                        draws = rngl.multivariate_normal(np.zeros(sub.shape[1]), cov, size=sub.shape[0])
                        cloud_w[:, hi] = draws
            elif mode == "P_low":
                for cloud_w in (wb1, wn1):
                    for j in band:
                        perm = rngl.permutation(cloud_w.shape[0])
                        cloud_w[:, j] = cloud_w[perm, j]
            else:
                raise ValueError(mode)
            return wb1, wn1

        moves: list[str] = []
        for q in Q_VALUES:
            band = bottom_idx(q)
            for mode in ("G_low", "G_high", "P_low"):
                vals = []
                for rs in HYBRID_RNG:
                    wb1, wn1 = apply_surrogate(wb, wn, band, mode, rs)
                    xb = unwhiten(wb1, mu, u, evals)
                    xnn = unwhiten(wn1, mu_new, u, evals)
                    g, _, _ = tie_gamma(radii(xnn, mu_new, sigma), radii(xb, mu, sigma))
                    vals.append(g)
                mean_g = float(np.mean(vals))
                sd_g = float(np.std(vals, ddof=0))
                hybrid_rows.append(
                    {
                        "cell": cell,
                        "q": q,
                        "method": mode,
                        "gamma_mean": mean_g,
                        "gamma_sd": sd_g,
                    }
                )
                if abs(mean_g - g_full) < abs(gamma_real - g_full):
                    moves.append(
                        f"q={q} {mode} mean={mean_g:.3f} (|Gamma-full|={abs(mean_g - g_full):.3f} "
                        f"vs real={abs(gamma_real - g_full):.3f})"
                    )
        narrative[cell] = "; ".join(moves) if moves else "none moved mean Gamma toward full surrogate vs real"
        log(f"part2 {cell}")

    write_csv(
        "surrogate_bands.csv",
        ("cell", "cloud", "band", "mean_d2_share", "kurtosis_median", "kurtosis_p90"),
        band_rows,
    )
    write_csv(
        "surrogate_hybrid.csv",
        ("cell", "q", "method", "gamma_mean", "gamma_sd"),
        hybrid_rows,
    )
    return band_rows, hybrid_rows, narrative


def embed_clip_iwild() -> tuple[Path, Path]:
    clip_path = ROOT / "clip_clouds.py"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import torch

    from clip_clouds import BATCH, load_clip, sha256_file as clip_sha  # noqa: WPS433

    id_path = EMB / "clip_iwild_id.npy"
    new_path = EMB / "clip_iwild_new.npy"
    if id_path.is_file() and new_path.is_file():
        print("clip_iwild matrices already present", flush=True)
        return id_path, new_path

    id_idx = np.load(IWILD_IDX / "indices_id.npy")
    new_idx = np.load(IWILD_IDX / "indices_new.npy")
    if id_idx.shape[0] != 2000 or new_idx.shape[0] != 2000:
        raise SystemExit("iwild indices are not 2000 rows each")

    from wilds import get_dataset

    dataset = get_dataset(dataset="iwildcam", download=False, root_dir=str(WILDS_ROOT))
    train = dataset.get_subset("train")
    test = dataset.get_subset("test")
    train_set = set(int(i) for i in train.indices)
    test_set = set(int(i) for i in test.indices)
    if not all(int(i) in train_set for i in id_idx):
        raise SystemExit("indices_id contains non-train indices")
    if not all(int(i) in test_set for i in new_idx):
        raise SystemExit("indices_new contains non-test indices")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, _ = load_clip(device)

    from PIL import Image

    def embed_indices(indices: np.ndarray) -> np.ndarray:
        rows = []
        with torch.no_grad():
            for start in range(0, len(indices), BATCH):
                batch = []
                for idx in indices[start : start + BATCH]:
                    image = dataset.get_input(int(idx))
                    if not isinstance(image, Image.Image):
                        image = Image.fromarray(np.asarray(image))
                    batch.append(preprocess(image.convert("RGB")))
                tensor = torch.stack(batch, dim=0).to(device)
                features = model.encode_image(tensor)
                rows.append(features.detach().to(dtype=torch.float32).cpu().numpy())
                if start % (BATCH * 20) == 0:
                    print(f"clip_iwild embedded {start}/{len(indices)}", flush=True)
        return np.concatenate(rows, axis=0).astype(np.float32, copy=False)

    x_id = embed_indices(id_idx)
    x_new = embed_indices(new_idx)
    if x_id.shape != (2000, 512) or x_new.shape != (2000, 512):
        raise SystemExit(f"clip_iwild shapes {x_id.shape} {x_new.shape}")

    tmp_id = id_path.with_suffix(".tmp.npy")
    tmp_new = new_path.with_suffix(".tmp.npy")
    np.save(tmp_id, np.ascontiguousarray(x_id))
    np.save(tmp_new, np.ascontiguousarray(x_new))
    os.replace(tmp_id, id_path)
    os.replace(tmp_new, new_path)
    print(f"sha256 clip_iwild_id {clip_sha(id_path)}", flush=True)
    print(f"sha256 clip_iwild_new {clip_sha(new_path)}", flush=True)
    return id_path, new_path


def phi_disjoint(x_id: np.ndarray, x_new: np.ndarray, sigma: np.ndarray, mu: np.ndarray, mu_new: np.ndarray, d: int):
    n_new = x_new.shape[0]
    centered = x_new - mu_new
    s_new = (centered.T @ centered) / float(n_new)
    lambdas = np.sort(np.asarray(eigh(s_new, sigma, eigvals_only=True), dtype=np.float64))
    sum_lambda = float(np.sum(lambdas))
    sum_sq = float(np.sum(lambdas**2))
    denom = np.sqrt(2.0 * d + 2.0 * sum_sq)
    phi = float(norm.cdf((d - sum_lambda) / denom))
    gen = np.random.Generator(np.random.PCG64(0))
    z_draw = gen.standard_normal((n_new, d))
    w_draw = gen.standard_normal((x_id.shape[0], d))
    r2_new = (z_draw**2) @ lambdas
    r2_id = np.sum(w_draw**2, axis=1)
    gamma_s, se_s, _ = tie_gamma(r2_new, r2_id)
    return phi, gamma_s, se_s, lambdas


def part3() -> tuple[list[dict], list[dict], list[dict], dict]:
    id_path, new_path = embed_clip_iwild()
    x_id = np.asarray(np.load(id_path), dtype=np.float64)
    x_new = np.asarray(np.load(new_path), dtype=np.float64)
    meta = {
        "sha256_id": sha256_file(id_path),
        "sha256_new": sha256_file(new_path),
        "indices_id_sha": sha256_file(IWILD_IDX / "indices_id.npy"),
        "indices_new_sha": sha256_file(IWILD_IDX / "indices_new.npy"),
    }

    summary_rows: list[dict] = []
    path_rows: list[dict] = []
    perm_rows: list[dict] = []

    # full fit
    sigma_full = fit_lw(x_id)
    mu_full = x_id.mean(axis=0)
    mu_new_full = x_new.mean(axis=0)
    for metric, sigma in (("ledoit_wolf", sigma_full), ("euclidean", None)):
        d_id = radii(x_id, mu_full, sigma)
        d_new = radii(x_new, mu_new_full, sigma)
        g, se, _ = tie_gamma(d_new, d_id)
        mapped = x_new - mu_new_full + mu_full
        g_post, _, _ = tie_gamma(radii(mapped, mu_full, sigma), d_id)
        summary_rows.append(
            {
                "analysis": "full_fit",
                "metric": metric,
                "seed": "",
                "gamma": g,
                "delong_se": se,
                "null_gamma": "",
                "identity_gap": abs(g_post - g),
                "phi": "",
                "gamma_surrogate": "",
                "n_lt_1": "",
                "n_eq_1": "",
                "n_gt_1": "",
            }
        )

    p0 = np.random.RandomState(0).permutation(2000)
    a = x_id[p0[:1000]]
    b = x_id[p0[1000:]]
    mu = a.mean(axis=0)
    mu_new = x_new.mean(axis=0)
    sigma = fit_lw(a)
    d = sigma.shape[0]

    for metric in ("ledoit_wolf", "euclidean"):
        sig = sigma if metric == "ledoit_wolf" else None
        d_id = radii(b, mu, sig)
        d_new = radii(x_new, mu_new, sig)
        g, se, _ = tie_gamma(d_new, d_id)
        mapped = x_new - mu_new + mu
        g_post, _, _ = tie_gamma(radii(mapped, mu, sig), d_id)
        nul = score_disjoint(x_id, x_new, 0, metric, real=False)
        if nul is None:
            raise SystemExit(f"clip_iwild null failed for {metric}")
        nul_g, nul_se, _ = nul
        phi, gamma_s, se_s, lambdas = (
            phi_disjoint(b, x_new, sigma, mu, mu_new, d) if metric == "ledoit_wolf" else ("", "", "", np.array([]))
        )
        row = {
            "analysis": "disjoint_seed0",
            "metric": metric,
            "seed": 0,
            "gamma": g,
            "delong_se": se,
            "null_gamma": nul_g,
            "identity_gap": abs(g_post - g),
            "phi": phi,
            "gamma_surrogate": gamma_s if metric == "ledoit_wolf" else "",
            "n_lt_1": int(np.sum(lambdas < 1.0)) if metric == "ledoit_wolf" else "",
            "n_eq_1": int(np.sum(lambdas == 1.0)) if metric == "ledoit_wolf" else "",
            "n_gt_1": int(np.sum(lambdas > 1.0)) if metric == "ledoit_wolf" else "",
        }
        summary_rows.append(row)

    for seed in SEEDS:
        got = score_disjoint(x_id, x_new, seed, "ledoit_wolf", real=True)
        if got is None:
            raise SystemExit(f"clip_iwild perm failed seed {seed}")
        g, se, ig = got
        perm_rows.append({"seed": seed, "gamma": g, "delong_se": se, "identity_gap": ig})

    scale = float(np.trace(sigma) / d)
    eye = np.eye(d)
    for k in range(21):
        t = k / 20.0
        sigma_t = (1.0 - t) * sigma + t * scale * eye
        d_id = radii(b, mu, sigma_t)
        d_new = radii(x_new, mu_new, sigma_t)
        g, _, _ = tie_gamma(d_new, d_id)
        path_rows.append({"t": t, "gamma": g})

    write_csv(
        "clip_iwild.csv",
        (
            "analysis",
            "metric",
            "seed",
            "gamma",
            "delong_se",
            "null_gamma",
            "identity_gap",
            "phi",
            "gamma_surrogate",
            "n_lt_1",
            "n_eq_1",
            "n_gt_1",
        ),
        summary_rows,
    )
    write_csv("clip_iwild_path.csv", ("t", "gamma"), path_rows)
    write_csv("clip_iwild_perm.csv", ("seed", "gamma", "delong_se", "identity_gap"), perm_rows)
    log("part3 done")
    return summary_rows, path_rows, perm_rows, meta


def summary_from_part1_csv() -> dict:
    path = OUT / "extra_metrics_disjoint.csv"
    if not path.is_file():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    summary: dict = {}
    for cell in CELL_ORDER:
        for metric in ("diag", "sphere_LW"):
            block = [r for r in rows if r["cell"] == cell and r["metric"] == metric]
            if not block:
                continue
            gammas = np.array([float(r["gamma"]) for r in block])
            nulls = [float(r["null_gamma"]) for r in block]
            id_gaps = [float(r["identity_gap"]) for r in block]
            vs_half = np.sign(gammas - 0.5)
            vs_half = vs_half[vs_half != 0]
            if len(vs_half) == 0:
                half_class = "mixed"
            elif np.all(vs_half > 0):
                half_class = "above_1/2"
            elif np.all(vs_half < 0):
                half_class = "below_1/2"
            else:
                half_class = "mixed"
            vs_null = [g < n for g, n in zip(gammas, nulls)]
            if all(vs_null):
                null_class = "below_own_null"
            elif not any(vs_null):
                null_class = "above_own_null"
            else:
                null_class = "mixed_vs_null"
            summary[(cell, metric)] = {
                "gamma_median": float(np.median(gammas)),
                "gamma_min": float(np.min(gammas)),
                "gamma_max": float(np.max(gammas)),
                "half_class": half_class,
                "null_class": null_class,
                "identity_gap_max": float(np.max(id_gaps)),
            }
    return summary


def narrative_from_hybrid_csv() -> dict:
    path = OUT / "surrogate_hybrid.csv"
    if not path.is_file():
        return {}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    narrative: dict[str, str] = {}
    for cell in CELL_ORDER:
        real_row = next((r for r in rows if r["cell"] == cell and r["method"] == "real"), None)
        full_row = next((r for r in rows if r["cell"] == cell and r["method"] == "full_gaussian"), None)
        if not real_row or not full_row:
            continue
        gamma_real = float(real_row["gamma_mean"])
        g_full = float(full_row["gamma_mean"])
        moves: list[str] = []
        for r in rows:
            if r["cell"] != cell or r["method"] not in ("G_low", "G_high", "P_low"):
                continue
            mean_g = float(r["gamma_mean"])
            if abs(mean_g - g_full) < abs(gamma_real - g_full):
                moves.append(
                    f"q={r['q']} {r['method']} mean={mean_g:.3f} "
                    f"(|Gamma-full|={abs(mean_g - g_full):.3f} vs |real-full|={abs(gamma_real - g_full):.3f})"
                )
        narrative[cell] = "; ".join(moves) if moves else "none moved mean Gamma toward full surrogate vs real"
    return narrative


def build_report(
    t0: float,
    p1_summary: dict,
    p2_narrative: dict,
    p3_meta: dict | None,
    sha256_new: list[tuple[str, str]],
) -> None:
    lines = ["# Round 3 report", "", f"Runtime: {time.time() - t0:.1f} s", ""]
    lines += ["## Part 1 — extra disjoint metrics", ""]
    for cell in CELL_ORDER:
        for metric in ("diag", "sphere_LW"):
            s = p1_summary[(cell, metric)]
            lines.append(
                f"- **{cell} / {metric}**: median Gamma [{s['gamma_min']:.3f}, {s['gamma_max']:.3f}] "
                f"median={s['gamma_median']:.3f}; vs 1/2: {s['half_class']}; vs own null: {s['null_class']}; "
                f"max identity_gap={s['identity_gap_max']:.4f}"
            )
    lines += ["", "Output: `extra_metrics_disjoint.csv`.", ""]

    lines += ["## Part 2 — surrogate bands (seed 0 disjoint LW)", ""]
    for cell in CELL_ORDER:
        lines.append(f"- **{cell}**: {p2_narrative.get(cell, '')}")
    lines += ["", "Outputs: `surrogate_bands.csv`, `surrogate_hybrid.csv`.", ""]

    lines += ["## Part 3 — CLIP ViT-B/16 iWildCam", ""]
    if p3_meta is None:
        lines.append("STOP: part 3 did not complete.")
    else:
        lines.append(
            "Recovered subsample: `iwildcam/indices_id.npy` and `indices_new.npy` "
            f"(sha256 {p3_meta['indices_id_sha']}, {p3_meta['indices_new_sha']}); "
            "train vs official test per `IWILDCAM.md` (cap: sort, RandomState(0).shuffle, first 2000)."
        )
        lines.append(
            f"New embeddings: `export_embeddings/clip_iwild_id.npy` sha256={p3_meta['sha256_id']}; "
            f"`clip_iwild_new.npy` sha256={p3_meta['sha256_new']}."
        )
        lines.append("Outputs: `clip_iwild.csv`, `clip_iwild_path.csv`, `clip_iwild_perm.csv`.")
    lines += ["", "## New .npy sha256", ""]
    for name, digest in sha256_new:
        lines.append(f"- {name}: `{digest}`")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run.log").write_text("", encoding="utf-8")
    t0 = time.time()
    sanity_gate()
    p3_meta = None
    sha256_new: list[tuple[str, str]] = []
    if os.environ.get("ROUND3_PART3_ONLY") == "1":
        log("PART3 only mode")
        p1_summary = summary_from_part1_csv()
        p2_narrative = narrative_from_hybrid_csv()
        try:
            _, _, _, p3_meta = part3()
            for name in ("clip_iwild_id.npy", "clip_iwild_new.npy"):
                path = EMB / name
                if path.is_file():
                    sha256_new.append((str(path.relative_to(ROOT)), sha256_file(path)))
        except Exception as exc:
            (OUT / "PART3_STOP.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            log(f"PART3 STOP: {exc}")
            p3_meta = None
        build_report(t0, p1_summary, p2_narrative, p3_meta, sha256_new)
        print("wrote", OUT, flush=True)
        return
    _, p1_summary = part1()
    _, _, p2_narrative = part2()
    if os.environ.get("ROUND3_SKIP_PART3") == "1":
        log("PART3 skipped (ROUND3_SKIP_PART3=1)")
        build_report(t0, p1_summary, p2_narrative, None, sha256_new)
        print("wrote", OUT, flush=True)
        return
    try:
        _, _, _, p3_meta = part3()
        for name in ("clip_iwild_id.npy", "clip_iwild_new.npy"):
            path = EMB / name
            if path.is_file():
                sha256_new.append((str(path.relative_to(ROOT)), sha256_file(path)))
    except Exception as exc:
        stop_path = OUT / "PART3_STOP.txt"
        stop_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        print(f"PART3 STOP: {exc}", flush=True)
    build_report(t0, p1_summary, p2_narrative, p3_meta, sha256_new)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
