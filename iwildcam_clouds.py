"""DINOv2 on WILDS iWildCam train versus official test. One encoder. No training.

Cap: sort subset indices, shuffle with RandomState(0), take 2000.
Matrices are written before T, before Ledoit-Wolf, and before any AUROC.
"""

import hashlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

os.environ["PYTHONHASHSEED"] = "0"
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from radial import (  # noqa: E402
    AUROC_ATOL,
    ROOT,
    arithmetic_mean,
    auroc_inlier,
    gaussian_loglik,
    mahalanobis_radius,
    oracle_translate,
    radius_inversion_mw,
)
from embed.__main__ import BATCH, load_encoder  # noqa: E402
from scipy.linalg import eigh  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402
import scipy  # noqa: E402

WILDS_ROOT = Path("/data2/cmdir/home/toandq/DST-Skin/data/raw/wilds")
OUT = ROOT / "iwildcam"
CAP = 2000
SMOKE_N = 64


def stop(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_f32(path: Path, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 768:
        stop(f"{path.name} shape {array.shape} is not (n, 768)")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_i64(path: Path, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array, dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_note(text: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "IWILDCAM.md").write_text(text, encoding="utf-8")


def load_dataset():
    try:
        from wilds import get_dataset
    except Exception as exc:
        write_note(f"# iWildCam\n\nexception: {type(exc).__name__}: {exc}\n")
        stop(f"wilds import failed: {type(exc).__name__}: {exc}")
    try:
        dataset = get_dataset(dataset="iwildcam", download=True, root_dir=str(WILDS_ROOT))
    except Exception as exc:
        write_note(
            "\n".join(
                [
                    "# iWildCam",
                    "",
                    "download_or_load: failed",
                    f"exception: {type(exc).__name__}: {exc}",
                    "",
                ]
            )
        )
        stop(f"iwildcam get_dataset failed: {type(exc).__name__}: {exc}")
    split_dict = getattr(dataset, "split_dict", None)
    print("split_dict", split_dict, flush=True)
    if not isinstance(split_dict, dict) or "train" not in split_dict or "test" not in split_dict:
        print(split_dict)
        write_note(f"# iWildCam\n\nsplit_dict: {split_dict}\n")
        stop("train or test is absent from split_dict")
    try:
        train = dataset.get_subset("train")
        test = dataset.get_subset("test")
    except Exception as exc:
        print(split_dict)
        write_note(f"# iWildCam\n\nsplit_dict: {split_dict}\nexception: {type(exc).__name__}: {exc}\n")
        stop(f"get_subset failed: {type(exc).__name__}: {exc}")
    if len(train) == 0 or len(test) == 0:
        print(split_dict)
        write_note(f"# iWildCam\n\nsplit_dict: {split_dict}\ntrain_or_test: empty\n")
        stop("train or test subset is empty")
    return dataset, train, test


def cap_indices(subset) -> np.ndarray:
    indices = np.array(sorted(int(i) for i in subset.indices), dtype=np.int64)
    order = indices.copy()
    np.random.RandomState(0).shuffle(order)
    if order.shape[0] > CAP:
        order = order[:CAP]
    return np.ascontiguousarray(order, dtype=np.int64)


def embed_indices(dataset, indices: np.ndarray, model, transform, device) -> np.ndarray:
    from PIL import Image

    rows = []
    with torch.no_grad():
        for start in range(0, len(indices), BATCH):
            batch = []
            for idx in indices[start : start + BATCH]:
                image = dataset.get_input(int(idx))
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(np.asarray(image))
                batch.append(transform(image.convert("RGB")))
            tensor = torch.stack(batch, dim=0).to(device)
            features = model.forward_features(tensor)
            if "x_norm_clstoken" not in features:
                stop("forward_features did not return x_norm_clstoken")
            token = features["x_norm_clstoken"]
            if token.ndim != 2 or token.shape[0] != tensor.shape[0]:
                stop(f"CLS token shape {tuple(token.shape)} is not a batch of vectors")
            rows.append(token.detach().to(dtype=torch.float32).cpu().numpy())
            if start % (BATCH * 25) == 0:
                print(f"embedded {start + len(batch)}/{len(indices)}", flush=True)
    if not rows:
        stop("no images were embedded")
    return np.concatenate(rows, axis=0).astype(np.float32, copy=False)


def matrices_ready() -> bool:
    has_id = (OUT / "X_id.npy").is_file()
    has_new = (OUT / "X_new.npy").is_file()
    if has_id != has_new:
        stop("only one iWildCam matrix is on disk; leaving it in place")
    return has_id and has_new


def fit_covariance(x_id: np.ndarray) -> np.ndarray:
    x = np.asarray(x_id, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or not np.isfinite(x).all():
        stop("ID matrix is not usable for Ledoit-Wolf")
    fitted = LedoitWolf(assume_centered=False).fit(x)
    sigma = np.asarray(fitted.covariance_, dtype=np.float64)
    if sigma.shape != (x.shape[1], x.shape[1]) or not np.isfinite(sigma).all():
        stop("Ledoit-Wolf covariance_ is not usable")
    if not np.allclose(sigma, sigma.T, rtol=0.0, atol=1e-8):
        stop("Ledoit-Wolf covariance_ is not symmetric")
    return sigma


def same_sign(gamma: float, premap_mw: float) -> str:
    return "yes" if np.sign(gamma - 0.5) == np.sign(premap_mw - 0.5) else "no"


def score_pair(s_new: np.ndarray, s_id: np.ndarray) -> float:
    auc, _mw = auroc_inlier(s_new, s_id)
    return auc


def ledoit_report(x_id: np.ndarray, x_new: np.ndarray, sigma: np.ndarray) -> dict:
    mu = arithmetic_mean(x_id)
    mu_new = arithmetic_mean(x_new)
    precision = np.linalg.inv(sigma)
    d_id = mahalanobis_radius(x_id, mu, precision)
    d_new = mahalanobis_radius(x_new, mu_new, precision)
    premap = radius_inversion_mw(d_new, d_id)
    mapped = oracle_translate(x_new, mu_new, mu)
    d_mapped = mahalanobis_radius(mapped, mu, precision)
    d_id_post = mahalanobis_radius(x_id, mu, precision)
    gamma = score_pair(-d_mapped, -d_id_post)
    gaussian = score_pair(gaussian_loglik(d_mapped, sigma), gaussian_loglik(d_id_post, sigma))
    return {"gamma": gamma, "gaussian_gap": float(gamma - gaussian), "same_sign": same_sign(gamma, premap)}


def euclidean_report(x_id: np.ndarray, x_new: np.ndarray) -> dict:
    mu = arithmetic_mean(x_id)
    mu_new = arithmetic_mean(x_new)
    d_id = np.linalg.norm(np.asarray(x_id, dtype=np.float64) - mu, axis=1)
    d_new = np.linalg.norm(np.asarray(x_new, dtype=np.float64) - mu_new, axis=1)
    premap = radius_inversion_mw(d_new, d_id)
    mapped = oracle_translate(x_new, mu_new, mu)
    d_mapped = np.linalg.norm(mapped - mu, axis=1)
    d_id_post = np.linalg.norm(np.asarray(x_id, dtype=np.float64) - mu, axis=1)
    eye = np.eye(x_id.shape[1], dtype=np.float64)
    gamma = score_pair(-d_mapped, -d_id_post)
    gaussian = score_pair(gaussian_loglik(d_mapped, eye), gaussian_loglik(d_id_post, eye))
    return {"gamma": gamma, "gaussian_gap": float(gamma - gaussian), "same_sign": same_sign(gamma, premap)}


def sign_condition(lambdas: np.ndarray) -> str:
    if np.all(lambdas <= 1.0) and np.any(lambdas < 1.0):
        return "all_le_1"
    if np.all(lambdas >= 1.0) and np.any(lambdas > 1.0):
        return "all_ge_1"
    return "straddles"


def spectrum_of(x_new: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    x = np.asarray(x_new, dtype=np.float64)
    centered = x - x.mean(axis=0)
    s_new = (centered.T @ centered) / float(x.shape[0])
    try:
        lambdas = eigh(s_new, np.asarray(sigma, dtype=np.float64), eigvals_only=True, check_finite=True)
    except Exception as exc:
        write_note(
            "\n".join(
                [
                    "# iWildCam",
                    "",
                    "eigh: raised",
                    f"exception: {type(exc).__name__}: {exc}",
                    "",
                ]
            )
        )
        stop(f"eigh raised: {type(exc).__name__}: {exc}")
    lambdas = np.sort(np.asarray(lambdas, dtype=np.float64))
    if lambdas.shape != (x.shape[1],):
        stop(f"eigenvalue vector length {lambdas.shape[0]} is not d")
    return lambdas


def write_identity_bug(metric: str, gap: float) -> None:
    write_note(
        "\n".join(
            [
                "# iWildCam",
                "",
                "identity_bug",
                "",
                f"metric: {metric}",
                f"gaussian_gap: {gap:.16g}",
                "The Mahalanobis and one-center Gaussian scores disagree by more than 1e-6.",
                "",
            ]
        )
    )
    stop(f"identity_bug {metric} gap {gap:.16g}")


def smoke(dataset, id_idx: np.ndarray, new_idx: np.ndarray, model, transform, device) -> None:
    print("iWildCam smoke embed 64+64", flush=True)
    x_id = embed_indices(dataset, id_idx[:SMOKE_N], model, transform, device)
    x_new = embed_indices(dataset, new_idx[:SMOKE_N], model, transform, device)
    report = ledoit_report(x_id, x_new, fit_covariance(x_id))
    print(f"smoke gap {abs(report['gaussian_gap']):.12g}", flush=True)
    if abs(report["gaussian_gap"]) > AUROC_ATOL:
        stop("smoke Mahalanobis/Gaussian gap exceeded 1e-6")


def write_outputs(id_idx: np.ndarray, new_idx: np.ndarray, lw: dict, eu: dict, spec: dict, n_id: int, n_new: int, d: int) -> None:
    import csv

    columns = (
        "metric",
        "n_id",
        "n_new",
        "d",
        "gamma",
        "gaussian_gap",
        "same_sign",
        "lambda_min",
        "lambda_max",
        "n_lt_1",
        "n_gt_1",
        "sign_condition",
    )
    rows = [
        {
            "metric": "ledoit_wolf",
            "n_id": n_id,
            "n_new": n_new,
            "d": d,
            "gamma": f"{lw['gamma']:.16g}",
            "gaussian_gap": f"{lw['gaussian_gap']:.16g}",
            "same_sign": lw["same_sign"],
            "lambda_min": f"{spec['lambda_min']:.16g}",
            "lambda_max": f"{spec['lambda_max']:.16g}",
            "n_lt_1": spec["n_lt_1"],
            "n_gt_1": spec["n_gt_1"],
            "sign_condition": spec["sign_condition"],
        },
        {
            "metric": "euclidean",
            "n_id": n_id,
            "n_new": n_new,
            "d": d,
            "gamma": f"{eu['gamma']:.16g}",
            "gaussian_gap": f"{eu['gaussian_gap']:.16g}",
            "same_sign": eu["same_sign"],
            "lambda_min": "",
            "lambda_max": "",
            "n_lt_1": "",
            "n_gt_1": "",
            "sign_condition": "",
        },
    ]
    with (OUT / "iwildcam_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# iWildCam",
        "",
        "subsets: train, test",
        "id: get_subset(\"train\")",
        "new: get_subset(\"test\")",
        "cap: sort indices, RandomState(0).shuffle, first 2000",
        f"sha256_indices_id: {sha256_file(OUT / 'indices_id.npy')}",
        f"sha256_indices_new: {sha256_file(OUT / 'indices_new.npy')}",
        f"sha256_X_id: {sha256_file(OUT / 'X_id.npy')}",
        f"sha256_X_new: {sha256_file(OUT / 'X_new.npy')}",
        f"torch: {torch.__version__}",
        f"scipy: {scipy.__version__}",
        f"hostname: {os.uname().nodename}",
        "encoder: torch.hub facebookresearch/dinov2 dinov2_vitb14",
        "vector: x_norm_clstoken, float32, no later L2",
        "eigh: ran",
        f"n_id: {n_id}",
        f"n_new: {n_new}",
        f"d: {d}",
        f"ledoit_gamma: {lw['gamma']:.16g}",
        f"ledoit_gaussian_gap: {lw['gaussian_gap']:.16g}",
        f"ledoit_same_sign: {lw['same_sign']}",
        f"euclidean_gamma: {eu['gamma']:.16g}",
        f"euclidean_gaussian_gap: {eu['gaussian_gap']:.16g}",
        f"euclidean_same_sign: {eu['same_sign']}",
        f"sign_condition: {spec['sign_condition']}",
        f"n_indices_id: {id_idx.shape[0]}",
        f"n_indices_new: {new_idx.shape[0]}",
        "",
    ]
    (OUT / "IWILDCAM.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not torch.cuda.is_available():
        stop("CUDA is not available on this host.")
    dataset, train, test = load_dataset()
    id_idx = cap_indices(train)
    new_idx = cap_indices(test)
    print(f"cap train {len(train)} -> {id_idx.shape[0]} test {len(test)} -> {new_idx.shape[0]}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    save_i64(OUT / "indices_id.npy", id_idx)
    save_i64(OUT / "indices_new.npy", new_idx)

    model, transform, device = load_encoder()
    print(f"encoder on {torch.cuda.get_device_name(device)}", flush=True)
    if not matrices_ready():
        smoke(dataset, id_idx, new_idx, model, transform, device)
        print(f"embed iWildCam n_id={id_idx.shape[0]} n_new={new_idx.shape[0]}", flush=True)
        x_id = embed_indices(dataset, id_idx, model, transform, device)
        x_new = embed_indices(dataset, new_idx, model, transform, device)
        save_f32(OUT / "X_id.npy", x_id)
        save_f32(OUT / "X_new.npy", x_new)
        print("saved iWildCam matrices before Ledoit-Wolf", flush=True)
    x_id = np.load(OUT / "X_id.npy")
    x_new = np.load(OUT / "X_new.npy")
    if x_id.dtype != np.float32 or x_new.dtype != np.float32:
        stop(f"saved dtype {x_id.dtype} {x_new.dtype} is not float32")
    sigma = fit_covariance(x_id)
    np.save(OUT / "covariance.npy", sigma)
    lw = ledoit_report(x_id, x_new, sigma)
    eu = euclidean_report(x_id, x_new)
    for name, report in (("ledoit_wolf", lw), ("euclidean", eu)):
        if abs(report["gaussian_gap"]) > AUROC_ATOL:
            write_identity_bug(name, report["gaussian_gap"])
    lambdas = spectrum_of(x_new, sigma)
    np.save(OUT / "spectrum_iwildcam.npy", lambdas)
    spec = {
        "lambda_min": float(lambdas[0]),
        "lambda_max": float(lambdas[-1]),
        "n_lt_1": int(np.sum(lambdas < 1.0)),
        "n_gt_1": int(np.sum(lambdas > 1.0)),
        "sign_condition": sign_condition(lambdas),
    }
    write_outputs(id_idx, new_idx, lw, eu, spec, int(x_id.shape[0]), int(x_new.shape[0]), int(x_id.shape[1]))
    print(
        f"iWildCam LW {lw['gamma']:.6g} euclid {eu['gamma']:.6g} {spec['sign_condition']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
