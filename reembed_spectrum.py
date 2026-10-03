"""Re-embed the scored filename lists, save the clouds, then the spectrum.

One encoder. Saved lists are embedded in file order. Matrices are written
before Ledoit-Wolf, before T, and before any AUROC. A later restart loads
those files and does not delete them.
"""

import csv
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

from radial import ROOT, arithmetic_mean, evaluate_mapped_split  # noqa: E402
from embed.__main__ import embed_paths, load_encoder  # noqa: E402
from scipy.linalg import eigh  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402
import scipy  # noqa: E402

SCORED = {"dermatology": 0.0155, "pathology": 0.2235}
LISTS = {
    "dermatology": ("dermatology_id.txt", "dermatology_new.txt"),
    "pathology": ("pathology_id.txt", "pathology_new.txt"),
}
SPECTRUM_NPY = {"dermatology": "spectrum_derm.npy", "pathology": "spectrum_pathology.npy"}
COLUMNS = (
    "modality",
    "n_id",
    "n_new",
    "d",
    "lambda_min",
    "lambda_max",
    "n_lt_1",
    "n_gt_1",
    "trace_over_d",
    "sign_condition",
)


def stop(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_list(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    paths = [line for line in lines if line.strip()]
    if not paths:
        stop(f"{path} is empty")
    return paths


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


def save_text(path: Path, paths: list[str]) -> None:
    payload = ("\n".join(paths) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".txt.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sign_condition(lambdas: np.ndarray) -> str:
    if np.all(lambdas <= 1.0) and np.any(lambdas < 1.0):
        return "all_le_1"
    if np.all(lambdas >= 1.0) and np.any(lambdas > 1.0):
        return "all_ge_1"
    return "straddles"


def spectrum_of(x_new: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    x = np.asarray(x_new, dtype=np.float64)
    n_new = x.shape[0]
    mu_new = x.mean(axis=0)
    centered = x - mu_new
    s_new = (centered.T @ centered) / float(n_new)
    try:
        lambdas = eigh(s_new, np.asarray(sigma, dtype=np.float64), eigvals_only=True, check_finite=True)
    except Exception as exc:
        note = "\n".join(
            [
                "",
                "eigh: raised",
                f"hostname: {os.uname().nodename}",
                f"torch: {torch.__version__}",
                f"scipy: {scipy.__version__}",
                f"exception: {type(exc).__name__}: {exc}",
                "",
            ]
        )
        with (ROOT / "REEMBED.md").open("a", encoding="utf-8") as handle:
            handle.write(note)
        stop(f"eigh raised: {type(exc).__name__}: {exc}")
    lambdas = np.sort(np.asarray(lambdas, dtype=np.float64))
    if lambdas.shape != (x.shape[1],):
        stop(f"eigenvalue vector length {lambdas.shape} is not d")
    return lambdas


def row_from(modality: str, lambdas: np.ndarray, n_id: int, n_new: int) -> dict:
    d = int(lambdas.shape[0])
    return {
        "modality": modality,
        "n_id": n_id,
        "n_new": n_new,
        "d": d,
        "lambda_min": float(lambdas[0]),
        "lambda_max": float(lambdas[-1]),
        "n_lt_1": int(np.sum(lambdas < 1.0)),
        "n_gt_1": int(np.sum(lambdas > 1.0)),
        "trace_over_d": float(np.sum(lambdas) / d),
        "sign_condition": sign_condition(lambdas),
    }


def write_spectrum_csv(rows: list[dict]) -> None:
    path = ROOT / "spectrum.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def cloud_dir(modality: str) -> Path:
    return ROOT / "reembed" / modality


def matrices_ready(modality: str) -> bool:
    directory = cloud_dir(modality)
    has_id = (directory / "X_id.npy").is_file()
    has_new = (directory / "X_new.npy").is_file()
    if has_id != has_new:
        stop(f"{modality} has only one saved matrix; leaving it in place")
    return has_id and has_new


def load_saved(modality: str) -> tuple[np.ndarray, np.ndarray]:
    directory = cloud_dir(modality)
    x_id = np.load(directory / "X_id.npy")
    x_new = np.load(directory / "X_new.npy")
    for name, array in (("X_id", x_id), ("X_new", x_new)):
        if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != 768:
            stop(f"saved {modality} {name} is {array.dtype} {array.shape}")
    print(f"loaded saved {modality} matrices {x_id.shape} {x_new.shape}", flush=True)
    return x_id, x_new


def embed_and_save(modality: str, id_paths: list[str], new_paths: list[str], model, transform, device) -> None:
    print(f"embed {modality} n_id={len(id_paths)} n_new={len(new_paths)}", flush=True)
    x_id = embed_paths(id_paths, model, transform, device)
    x_new = embed_paths(new_paths, model, transform, device)
    directory = cloud_dir(modality)
    save_f32(directory / "X_id.npy", x_id)
    save_f32(directory / "X_new.npy", x_new)
    save_text(directory / "filenames_id.txt", id_paths)
    save_text(directory / "filenames_new.txt", new_paths)
    hashes = {
        "X_id.npy": sha256_file(directory / "X_id.npy"),
        "X_new.npy": sha256_file(directory / "X_new.npy"),
        "filenames_id.txt": sha256_file(directory / "filenames_id.txt"),
        "filenames_new.txt": sha256_file(directory / "filenames_new.txt"),
    }
    lines = [f"{digest}  {name}" for name, digest in hashes.items()]
    save_text(directory / "sha256.txt", lines)
    print(f"saved {modality} matrices before Ledoit-Wolf", flush=True)


def fit_covariance(x_id: np.ndarray) -> np.ndarray:
    """One Ledoit-Wolf fit. covariance_ only; no second fit and no precision_."""
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


def smoke(model, transform, device) -> None:
    id_paths = read_list(ROOT / "filename_lists" / "dermatology_smoke_id.txt")
    new_paths = read_list(ROOT / "filename_lists" / "dermatology_smoke_new.txt")
    if len(id_paths) != 64 or len(new_paths) != 64:
        stop(f"smoke lists are {len(id_paths)} and {len(new_paths)}, not 64")
    print("smoke embed 64+64", flush=True)
    x_id = embed_paths(id_paths, model, transform, device)
    x_new = embed_paths(new_paths, model, transform, device)
    sigma = fit_covariance(x_id)
    measured = evaluate_mapped_split(
        x_id, x_new, arithmetic_mean(x_id), arithmetic_mean(x_new), sigma, np.linalg.inv(sigma)
    )
    gap = abs(float(measured["gap"]))
    print(f"smoke gap {gap:.12g}", flush=True)
    if gap > 1e-6:
        stop("smoke Mahalanobis/Gaussian gap exceeded 1e-6")


def require_paths(paths: list[str], label: str) -> None:
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        print(f"{label} missing {len(missing)} files", file=sys.stderr)
        for path in missing[:10]:
            print(path, file=sys.stderr)
        stop(f"{label} filename list points at missing files")


def write_report(records: list[dict]) -> None:
    lines = [
        "# Re-embed",
        "",
        f"hostname: {os.uname().nodename}",
        f"torch: {torch.__version__}",
        f"scipy: {scipy.__version__}",
        "filename_list_source: saved list",
        "cap_redraw: no",
        "encoder: torch.hub facebookresearch/dinov2 dinov2_vitb14",
        "eigh: ran",
        "",
        "Scored AUROC values are the finished-run figures and are not overwritten.",
        "",
    ]
    for record in records:
        lines.append(f"## {record['modality']}")
        lines.append("")
        for key in (
            "list_id",
            "list_new",
            "sha256_X_id",
            "sha256_X_new",
            "sha256_filenames_id",
            "sha256_filenames_new",
            "n_id",
            "n_new",
            "d",
            "sign_condition",
            "lambda_min",
            "lambda_max",
            "trace_over_d",
            "auroc_mahalanobis",
            "premap_mw",
            "scored_auroc",
        ):
            lines.append(f"{key}: {record[key]}")
        lines.append("")
    (ROOT / "REEMBED.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not torch.cuda.is_available():
        stop("CUDA is not available on this host.")
    lists = {}
    for modality, (id_name, new_name) in LISTS.items():
        id_list = ROOT / "filename_lists" / id_name
        new_list = ROOT / "filename_lists" / new_name
        if not id_list.is_file() or not new_list.is_file():
            stop(f"{modality} filename list is absent; refusing to rebuild")
        id_paths = read_list(id_list)
        new_paths = read_list(new_list)
        require_paths(id_paths, f"{modality} id")
        require_paths(new_paths, f"{modality} new")
        lists[modality] = (id_list, new_list, id_paths, new_paths)
        print(f"{modality} saved list {len(id_paths)} {len(new_paths)}", flush=True)

    model, transform, device = load_encoder()
    print(f"encoder on {torch.cuda.get_device_name(device)}", flush=True)
    smoke(model, transform, device)

    rows = []
    records = []
    for modality in ("dermatology", "pathology"):
        id_list, new_list, id_paths, new_paths = lists[modality]
        if matrices_ready(modality):
            x_id, x_new = load_saved(modality)
        else:
            embed_and_save(modality, id_paths, new_paths, model, transform, device)
            x_id, x_new = load_saved(modality)
        directory = cloud_dir(modality)
        sigma = fit_covariance(x_id)
        np.save(directory / "covariance.npy", sigma)
        lambdas = spectrum_of(x_new, sigma)
        np.save(ROOT / SPECTRUM_NPY[modality], lambdas)
        row = row_from(modality, lambdas, int(x_id.shape[0]), int(x_new.shape[0]))
        rows.append(row)
        write_spectrum_csv(rows)
        mu = arithmetic_mean(x_id)
        mu_new = arithmetic_mean(x_new)
        measured = evaluate_mapped_split(x_id, x_new, mu, mu_new, sigma, np.linalg.inv(sigma))
        records.append(
            {
                "modality": modality,
                "list_id": str(id_list),
                "list_new": str(new_list),
                "sha256_X_id": sha256_file(directory / "X_id.npy"),
                "sha256_X_new": sha256_file(directory / "X_new.npy"),
                "sha256_filenames_id": sha256_file(directory / "filenames_id.txt"),
                "sha256_filenames_new": sha256_file(directory / "filenames_new.txt"),
                "n_id": row["n_id"],
                "n_new": row["n_new"],
                "d": row["d"],
                "sign_condition": row["sign_condition"],
                "lambda_min": f"{row['lambda_min']:.16g}",
                "lambda_max": f"{row['lambda_max']:.16g}",
                "trace_over_d": f"{row['trace_over_d']:.16g}",
                "auroc_mahalanobis": f"{float(measured['auroc_mahalanobis']):.16g}",
                "premap_mw": f"{float(measured['premap_mw']):.16g}",
                "scored_auroc": SCORED[modality],
            }
        )
        write_report(records)
        print(
            f"{modality} {row['sign_condition']} "
            f"lambda [{row['lambda_min']:.6g}, {row['lambda_max']:.6g}] "
            f"auroc {float(measured['auroc_mahalanobis']):.6g} scored {SCORED[modality]}",
            flush=True,
        )
        del x_id, x_new, sigma, lambdas
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
