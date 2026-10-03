"""CLIP ViT-B/16 on the job-61185 filename lists. One encoder. No training.

Matrices are written before T, before Ledoit-Wolf, and before any AUROC.
A later restart loads those files and does not delete them.
"""

import csv
import hashlib
import inspect
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
from scipy.linalg import eigh  # noqa: E402
from sklearn.covariance import LedoitWolf  # noqa: E402
import scipy  # noqa: E402

BATCH = 32
OUT = ROOT / "clip"
LIST_ROOT = ROOT / "reembed"
MODALITIES = ("dermatology", "pathology")
SPECTRUM_NPY = {"dermatology": "spectrum_derm.npy", "pathology": "spectrum_pathology.npy"}
COLUMNS = (
    "modality",
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
    paths = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not paths:
        stop(f"{path} is empty")
    return paths


def save_f32(path: Path, array: np.ndarray) -> None:
    array = np.ascontiguousarray(array, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        stop(f"{path.name} shape {getattr(array, 'shape', None)} is not a cloud")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_text(path: Path, lines: list[str]) -> None:
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def cloud_dir(modality: str) -> Path:
    return OUT / modality


def matrices_ready(modality: str) -> bool:
    directory = cloud_dir(modality)
    has_id = (directory / "X_id.npy").is_file()
    has_new = (directory / "X_new.npy").is_file()
    if has_id != has_new:
        stop(f"{modality} has only one saved CLIP matrix; leaving it in place")
    return has_id and has_new


def load_saved(modality: str) -> tuple[np.ndarray, np.ndarray]:
    directory = cloud_dir(modality)
    x_id = np.load(directory / "X_id.npy")
    x_new = np.load(directory / "X_new.npy")
    for name, array in (("X_id", x_id), ("X_new", x_new)):
        if array.dtype != np.float32 or array.ndim != 2:
            stop(f"saved CLIP {modality} {name} is {array.dtype} {array.shape}")
    print(f"loaded saved CLIP {modality} {x_id.shape} {x_new.shape}", flush=True)
    return x_id, x_new


def require_paths(paths: list[str], label: str) -> None:
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        print(f"{label} missing {len(missing)} files", file=sys.stderr)
        for path in missing[:10]:
            print(path, file=sys.stderr)
        stop(f"{label} filename list points at missing files")


def job_lists(modality: str) -> tuple[list[str], list[str], str, Path, Path]:
    id_list = LIST_ROOT / modality / "filenames_id.txt"
    new_list = LIST_ROOT / modality / "filenames_new.txt"
    if not id_list.is_file() or not new_list.is_file():
        stop(f"{modality} job-61185 filename list is absent")
    id_paths = read_list(id_list)
    new_paths = read_list(new_list)
    require_paths(id_paths, f"{modality} id")
    require_paths(new_paths, f"{modality} new")
    return id_paths, new_paths, "job 61185 saved list", id_list, new_list


def load_clip(device: torch.device):
    try:
        import clip
    except Exception as exc:
        stop(f"clip import failed: {type(exc).__name__}: {exc}")
    try:
        model, preprocess = clip.load("ViT-B/16", device=device)
    except Exception as exc:
        stop(f"ViT-B/16 checkpoint failed: {type(exc).__name__}: {exc}")
    model.eval()
    source = inspect.getsource(model.encode_image)
    if "normalize" in source or "/ image_features" in source or "norm(" in source:
        stop("encode_image is not the unnormalized image embedding")
    if not callable(preprocess):
        stop("clip.load did not return a preprocess")
    return model, preprocess, source


def embed_paths(paths: list[str], model, preprocess, device) -> np.ndarray:
    from PIL import Image

    rows = []
    with torch.no_grad():
        for start in range(0, len(paths), BATCH):
            batch = []
            for path in paths[start : start + BATCH]:
                with Image.open(path) as image:
                    batch.append(preprocess(image.convert("RGB")))
            tensor = torch.stack(batch, dim=0).to(device)
            features = model.encode_image(tensor)
            if features.ndim != 2 or features.shape[0] != tensor.shape[0]:
                stop(f"encode_image shape {tuple(features.shape)} is not a batch of vectors")
            rows.append(features.detach().to(dtype=torch.float32).cpu().numpy())
            if start % (BATCH * 20) == 0:
                print(f"embedded {start + len(batch)}/{len(paths)}", flush=True)
    if not rows:
        stop("no images were embedded")
    return np.concatenate(rows, axis=0).astype(np.float32, copy=False)


def embed_and_save(modality: str, id_paths: list[str], new_paths: list[str], model, preprocess, device) -> None:
    print(f"embed CLIP {modality} n_id={len(id_paths)} n_new={len(new_paths)}", flush=True)
    x_id = embed_paths(id_paths, model, preprocess, device)
    x_new = embed_paths(new_paths, model, preprocess, device)
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
    save_text(directory / "sha256.txt", [f"{digest}  {name}" for name, digest in hashes.items()])
    print(f"saved CLIP {modality} matrices before Ledoit-Wolf", flush=True)


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
    return {
        "metric": "ledoit_wolf",
        "gamma": gamma,
        "gaussian_gap": float(gamma - gaussian),
        "same_sign": same_sign(gamma, premap),
        "premap_mw": premap,
    }


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
    return {
        "metric": "euclidean",
        "gamma": gamma,
        "gaussian_gap": float(gamma - gaussian),
        "same_sign": same_sign(gamma, premap),
        "premap_mw": premap,
    }


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
        with (OUT / "CLIP.md").open("a", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
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


def blank_spectrum() -> dict:
    return {"lambda_min": "", "lambda_max": "", "n_lt_1": "", "n_gt_1": "", "sign_condition": ""}


def spectrum_fields(lambdas: np.ndarray) -> dict:
    return {
        "lambda_min": f"{float(lambdas[0]):.16g}",
        "lambda_max": f"{float(lambdas[-1]):.16g}",
        "n_lt_1": int(np.sum(lambdas < 1.0)),
        "n_gt_1": int(np.sum(lambdas > 1.0)),
        "sign_condition": sign_condition(lambdas),
    }


def write_csv(rows: list[dict]) -> None:
    path = OUT / "clip_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in COLUMNS})


def write_identity_bug(modality: str, metric: str, gap: float) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "# CLIP",
            "",
            "identity_bug",
            "",
            f"modality: {modality}",
            f"metric: {metric}",
            f"gaussian_gap: {gap:.16g}",
            "The Mahalanobis and one-center Gaussian scores disagree by more than 1e-6.",
            "",
        ]
    )
    (OUT / "CLIP.md").write_text(text, encoding="utf-8")
    stop(f"identity_bug {modality} {metric} gap {gap:.16g}")


def smoke(model, preprocess, device) -> None:
    id_list = ROOT / "filename_lists" / "dermatology_smoke_id.txt"
    new_list = ROOT / "filename_lists" / "dermatology_smoke_new.txt"
    if not id_list.is_file() or not new_list.is_file():
        stop("dermatology smoke filename lists are absent")
    id_paths = read_list(id_list)
    new_paths = read_list(new_list)
    if len(id_paths) != 64 or len(new_paths) != 64:
        stop(f"smoke lists are {len(id_paths)} and {len(new_paths)}, not 64")
    require_paths(id_paths, "smoke id")
    require_paths(new_paths, "smoke new")
    print("CLIP smoke embed 64+64", flush=True)
    x_id = embed_paths(id_paths, model, preprocess, device)
    x_new = embed_paths(new_paths, model, preprocess, device)
    report = ledoit_report(x_id, x_new, fit_covariance(x_id))
    print(f"smoke gap {abs(report['gaussian_gap']):.12g}", flush=True)
    if abs(report["gaussian_gap"]) > AUROC_ATOL:
        stop("smoke Mahalanobis/Gaussian gap exceeded 1e-6")


def preprocess_note(preprocess) -> str:
    parts = [repr(preprocess)]
    mean = getattr(preprocess, "transforms", None)
    if mean is not None:
        for step in preprocess.transforms:
            name = type(step).__name__
            extra = ""
            if hasattr(step, "mean") and hasattr(step, "std"):
                extra = f" mean={tuple(float(v) for v in step.mean)} std={tuple(float(v) for v in step.std)}"
            if hasattr(step, "size"):
                extra += f" size={step.size}"
            parts.append(f"{name}{extra}")
    return " | ".join(parts)


def write_clip_md(header: dict, records: list[dict]) -> None:
    lines = [
        "# CLIP",
        "",
        "checkpoint: ViT-B/16",
        "vector: model.encode_image, float32, no later L2",
        f"preprocess: {header['preprocess']}",
        f"filename_list_source: {header['list_source']}",
        f"torch: {torch.__version__}",
        f"scipy: {scipy.__version__}",
        f"hostname: {os.uname().nodename}",
        "eigh: ran" if records and all("sign_condition" in record for record in records) else "eigh: pending",
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
            "ledoit_gamma",
            "ledoit_gaussian_gap",
            "ledoit_same_sign",
            "euclidean_gamma",
            "euclidean_gaussian_gap",
            "euclidean_same_sign",
            "sign_condition",
        ):
            lines.append(f"{key}: {record[key]}")
        lines.append("")
    (OUT / "CLIP.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not torch.cuda.is_available():
        stop("CUDA is not available on this host.")
    device = torch.device("cuda")
    lists = {}
    list_source = None
    for modality in MODALITIES:
        id_paths, new_paths, source, id_list, new_list = job_lists(modality)
        if list_source is None:
            list_source = source
        elif source != list_source:
            stop("filename-list sources differ across modalities")
        lists[modality] = (id_paths, new_paths, id_list, new_list)
        print(f"{modality} {source} {len(id_paths)} {len(new_paths)}", flush=True)

    model, preprocess, _source = load_clip(device)
    print(f"CLIP ViT-B/16 on {torch.cuda.get_device_name(device)}", flush=True)
    header = {"preprocess": preprocess_note(preprocess), "list_source": list_source}
    smoke(model, preprocess, device)

    rows: list[dict] = []
    records: list[dict] = []
    for modality in MODALITIES:
        id_paths, new_paths, id_list, new_list = lists[modality]
        if modality == "pathology" and not matrices_ready("dermatology"):
            stop("pathology started before dermatology matrices were on disk")
        if matrices_ready(modality):
            x_id, x_new = load_saved(modality)
        else:
            embed_and_save(modality, id_paths, new_paths, model, preprocess, device)
            x_id, x_new = load_saved(modality)
        directory = cloud_dir(modality)
        sigma = fit_covariance(x_id)
        np.save(directory / "covariance.npy", sigma)
        lw = ledoit_report(x_id, x_new, sigma)
        eu = euclidean_report(x_id, x_new)
        for report in (lw, eu):
            if abs(report["gaussian_gap"]) > AUROC_ATOL:
                write_identity_bug(modality, report["metric"], report["gaussian_gap"])
        lambdas = spectrum_of(x_new, sigma)
        np.save(OUT / SPECTRUM_NPY[modality], lambdas)
        spec = spectrum_fields(lambdas)
        n_id = int(x_id.shape[0])
        n_new = int(x_new.shape[0])
        d = int(x_id.shape[1])
        rows.append(
            {
                "modality": modality,
                "metric": "ledoit_wolf",
                "n_id": n_id,
                "n_new": n_new,
                "d": d,
                "gamma": f"{lw['gamma']:.16g}",
                "gaussian_gap": f"{lw['gaussian_gap']:.16g}",
                "same_sign": lw["same_sign"],
                **spec,
            }
        )
        rows.append(
            {
                "modality": modality,
                "metric": "euclidean",
                "n_id": n_id,
                "n_new": n_new,
                "d": d,
                "gamma": f"{eu['gamma']:.16g}",
                "gaussian_gap": f"{eu['gaussian_gap']:.16g}",
                "same_sign": eu["same_sign"],
                **blank_spectrum(),
            }
        )
        write_csv(rows)
        records.append(
            {
                "modality": modality,
                "list_id": str(id_list),
                "list_new": str(new_list),
                "sha256_X_id": sha256_file(directory / "X_id.npy"),
                "sha256_X_new": sha256_file(directory / "X_new.npy"),
                "sha256_filenames_id": sha256_file(directory / "filenames_id.txt"),
                "sha256_filenames_new": sha256_file(directory / "filenames_new.txt"),
                "n_id": n_id,
                "n_new": n_new,
                "d": d,
                "ledoit_gamma": f"{lw['gamma']:.16g}",
                "ledoit_gaussian_gap": f"{lw['gaussian_gap']:.16g}",
                "ledoit_same_sign": lw["same_sign"],
                "euclidean_gamma": f"{eu['gamma']:.16g}",
                "euclidean_gaussian_gap": f"{eu['gaussian_gap']:.16g}",
                "euclidean_same_sign": eu["same_sign"],
                "sign_condition": spec["sign_condition"],
            }
        )
        write_clip_md(header, records)
        print(
            f"{modality} LW {lw['gamma']:.6g} euclid {eu['gamma']:.6g} {spec['sign_condition']}",
            flush=True,
        )
        del x_id, x_new, sigma, lambdas
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
