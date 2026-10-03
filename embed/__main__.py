"""Dermatology then pathology. No second preprocessing, seed, or epsilon."""

import csv
import hashlib
import os
import random
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

os.environ["PYTHONHASHSEED"] = "0"
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from radial import (  # noqa: E402
    ROOT,
    append_gate,
    arithmetic_mean,
    evaluate_mapped_split,
    fit_id_covariance,
    utc_now,
)

CAP = 2000
SMOKE_N = 64
BATCH = 8
PAD_ROOT = Path("/data2/cmdir/home/toandq/DST-Skin/data/raw/pad_ufes20")
WILDS_ROOT = Path("/data2/cmdir/home/toandq/DST-Skin/data/raw/wilds")
ISIC_URL = "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
COLUMNS = (
    "modality",
    "split",
    "n_id",
    "n_new",
    "premap_mw",
    "certificate",
    "epsilon",
    "auroc_mahalanobis",
    "auroc_gaussian",
    "gap",
    "auroc_euclidean",
    "pass",
)


def stop(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def toy_ok() -> bool:
    log = ROOT / "toy_log.txt"
    if not log.is_file():
        return False
    lines = [line.strip() for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    return len(lines) == 7 and all(line.startswith("PASS ") for line in lines)


def subsample(paths: list[str], k: int) -> list[str]:
    ordered = sorted(paths)
    if len(ordered) <= k:
        return ordered
    rng = np.random.default_rng(0)
    picked = rng.choice(len(ordered), size=k, replace=False)
    return [ordered[i] for i in sorted(int(i) for i in picked)]


def write_list(name: str, paths: list[str]) -> str:
    directory = ROOT / "filename_lists"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{name}.txt"
    payload = ("\n".join(paths) + "\n").encode("utf-8")
    target.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def write_results(rows: list[dict]) -> None:
    csv_path = ROOT / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in COLUMNS})
    lines = [
        "# Results",
        "",
        "Euclidean AUROC is reported and is not part of pass. The oracle map is an alignment, not a detector.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join("---" for _ in COLUMNS) + " |",
    ]
    for row in rows:
        cells = []
        for key in COLUMNS:
            value = row[key]
            if isinstance(value, float):
                cells.append(f"{value:.12g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    (ROOT / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def require_toy() -> None:
    if toy_ok():
        return
    stop("python -m toy has not exited 0. No data download and no network load.")


def isic_directory() -> Path | None:
    candidates = [
        ROOT / "data" / "ISIC_2019_Training_Input",
        Path("/data2/cmdir/home/toandq/DST-Skin/data/raw/ISIC_2019_Training_Input"),
        Path("/data2/cmdir/home/toandq/DST-Skin/data/raw/isic2019/ISIC_2019_Training_Input"),
        Path("/data2/cmdir/home/toandq/data/ISIC_2019_Training_Input"),
    ]
    found = [path for path in candidates if path.is_dir()]
    if not found:
        return None
    if len(found) != 1:
        print("multiple ISIC_2019_Training_Input directories:")
        for path in found:
            print(path)
        stop("archive location is ambiguous")
    return found[0]


def download_isic() -> Path:
    destination = ROOT / "data"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "ISIC_2019_Training_Input.zip"
    print(f"downloading {ISIC_URL}")
    proc = subprocess.run(
        ["curl", "-L", "--fail", "--retry", "0", "-C", "-", "-o", str(archive), ISIC_URL],
        check=False,
    )
    if proc.returncode != 0 or not archive.is_file():
        stop(f"ISIC download failed with exit {proc.returncode}")
    try:
        with zipfile.ZipFile(archive) as handle:
            names = handle.namelist()
            if not any(Path(name).name.startswith("ISIC_") and name.endswith(".jpg") for name in names[:50]):
                top = sorted({name.split("/")[0] for name in names[:200]})
                print("zip top-level entries:")
                for entry in top:
                    print(entry)
                stop("ISIC archive layout does not match ISIC_2019_Training_Input")
            handle.extractall(destination)
    except zipfile.BadZipFile as exc:
        stop(f"ISIC archive is not a readable zip: {exc}")
    located = isic_directory()
    if located is None:
        print("extracted top-level listing:")
        for entry in sorted(destination.iterdir()):
            print(entry.name)
        stop("ISIC_2019_Training_Input was not found after extract")
    return located


def isic_paths(directory: Path) -> list[str]:
    if directory.name != "ISIC_2019_Training_Input":
        print("top-level name:", directory.name)
        stop("ID directory name is not ISIC_2019_Training_Input")
    paths = sorted(str(path) for path in directory.glob("ISIC_*.jpg") if path.is_file())
    if not paths:
        print("directory listing:")
        for entry in sorted(directory.iterdir())[:40]:
            print(entry.name)
        stop("ISIC_2019_Training_Input has no ISIC_*.jpg files")
    return paths


def pad_paths() -> list[str]:
    metadata = PAD_ROOT / "metadata.csv"
    parts = [
        PAD_ROOT / "hf" / "all_images" / "imgs_part_1",
        PAD_ROOT / "hf" / "all_images" / "imgs_part_2",
        PAD_ROOT / "hf" / "all_images" / "imgs_part_3",
    ]
    if not metadata.is_file() or any(not part.is_dir() for part in parts):
        print("PAD-UFES top-level listing:")
        if PAD_ROOT.is_dir():
            for entry in sorted(PAD_ROOT.iterdir()):
                print(entry.name)
        else:
            print(f"missing {PAD_ROOT}")
        stop("PAD-UFES-20 official layout was not found")
    index: dict[str, Path] = {}
    for part in parts:
        for path in part.iterdir():
            if not path.is_file():
                continue
            if path.name in index:
                stop(f"duplicate PAD-UFES filename {path.name}")
            index[path.name] = path
    chosen = []
    with metadata.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "img_id" not in reader.fieldnames:
            stop(f"metadata.csv columns are {reader.fieldnames}")
        for row in reader:
            img_id = row["img_id"]
            path = index.get(img_id)
            if path is None:
                stop(f"img_id {img_id} is not in imgs_part_1, imgs_part_2, or imgs_part_3")
            chosen.append(str(path))
    if not chosen:
        stop("metadata.csv listed no img_id entries")
    return chosen


def load_encoder():
    require_toy()
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    except Exception as exc:
        append_gate("dinov2_hub", 1)
        stop(f"{type(exc).__name__}: {exc}")
    model.eval()
    import inspect

    source = inspect.getsource(model.forward_features)
    if "x_norm_clstoken" not in source:
        stop("loaded DINOv2 forward_features does not expose x_norm_clstoken")
    if "functional.normalize" in source or "F.normalize" in source:
        stop("loaded DINOv2 forward_features L2-normalizes the CLS token")
    try:
        from dinov2.data.transforms import make_classification_eval_transform
    except Exception as exc:
        stop(f"official DINOv2 preprocess import failed: {type(exc).__name__}: {exc}")
    transform = make_classification_eval_transform()
    from PIL import Image
    from torchvision.transforms import Normalize

    probe = transform(Image.fromarray(np.zeros((300, 280, 3), dtype=np.uint8)))
    if tuple(probe.shape) != (3, 224, 224):
        stop(f"official preprocess produced shape {tuple(probe.shape)}, not 3x224x224")
    normalizers = [step for step in getattr(transform, "transforms", []) if isinstance(step, Normalize)]
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    if len(normalizers) != 1:
        stop("official preprocess does not contain one ImageNet Normalize")
    mean = tuple(float(v) for v in normalizers[0].mean)
    std = tuple(float(v) for v in normalizers[0].std)
    if mean != imagenet_mean or std != imagenet_std:
        stop(f"official Normalize mean/std {mean} {std} is not ImageNet")
    if not torch.cuda.is_available():
        stop("CUDA is not available on this host. Refusing the login-node CPU path.")
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()
    return model, transform, device


def embed_paths(paths: list[str], model, transform, device) -> np.ndarray:
    from PIL import Image

    rows = []
    with torch.no_grad():
        for start in range(0, len(paths), BATCH):
            batch = []
            for path in paths[start : start + BATCH]:
                with Image.open(path) as image:
                    batch.append(transform(image.convert("RGB")))
            tensor = torch.stack(batch, dim=0).to(device)
            features = model.forward_features(tensor)
            if "x_norm_clstoken" not in features:
                stop("forward_features did not return x_norm_clstoken")
            token = features["x_norm_clstoken"]
            if token.ndim != 2 or token.shape[0] != tensor.shape[0]:
                stop(f"CLS token shape {tuple(token.shape)} is not a batch of vectors")
            rows.append(token.detach().to(dtype=torch.float32).cpu().numpy())
    if not rows:
        stop("no images were embedded")
    return np.concatenate(rows, axis=0).astype(np.float32, copy=False)


def row_for(modality: str, split: str, x_id: np.ndarray, x_new: np.ndarray, fitted=None) -> tuple[dict, object]:
    mu = arithmetic_mean(x_id)
    mu_new = arithmetic_mean(x_new)
    if fitted is None:
        sigma, precision = fit_id_covariance(x_id)
        artifact = ROOT / "artifacts"
        artifact.mkdir(parents=True, exist_ok=True)
        np.save(artifact / f"sigma_{modality}_{split}.npy", sigma)
    else:
        sigma, precision = fitted
    measured = evaluate_mapped_split(x_id, x_new, mu, mu_new, sigma, precision)
    record = {"modality": modality, "split": split, **measured}
    return record, (sigma, precision, mu)


def control_cloud(x_id: np.ndarray) -> np.ndarray:
    """Seed-0 copy of the capped ID cloud plus one fixed noise scale."""
    rng = np.random.default_rng(0)
    order = rng.choice(x_id.shape[0], size=x_id.shape[0], replace=False)
    copied = np.asarray(x_id, dtype=np.float64)[order]
    centered = np.asarray(x_id, dtype=np.float64) - arithmetic_mean(x_id)
    rms = float(np.sqrt(np.mean(centered ** 2)))
    if not np.isfinite(rms) or rms <= 0.0:
        stop("ID embedding RMS is not usable for the control scale")
    sigma = 10.0 * rms
    print(f"control noise scale {sigma:.12g}")
    noise = rng.normal(loc=0.0, scale=sigma, size=copied.shape)
    return copied + noise


def manifest(hashes: dict[str, str], hub: str) -> None:
    import scipy
    import sklearn
    import torchvision

    try:
        git_hash = subprocess.run(
            ["git", "-C", "/data2/cmdir/home/toandq", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError:
        git_hash = ""
    if not git_hash:
        git_hash = "repository exists at /data2/cmdir/home/toandq but has no commit hash"
    lines = [
        "# Manifest",
        "",
        f"utc_date: {utc_now()[:10]}",
        f"hostname: {os.uname().nodename}",
        f"python: {sys.version.split()[0]}",
        f"torch: {torch.__version__}",
        f"torchvision: {torchvision.__version__}",
        f"scikit-learn: {sklearn.__version__}",
        f"scipy: {scipy.__version__}",
        f"git_hash: {git_hash}",
        f"dinov2_hub: {hub}",
        f"cap: {CAP}",
        "filename_list_sha256:",
    ]
    for name, digest in hashes.items():
        lines.append(f"  {name}: {digest}")
    lines.append("")
    (ROOT / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")


def hub_resolution() -> str:
    repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if not repo.is_dir():
        matches = sorted(Path(torch.hub.get_dir()).glob("facebookresearch_dinov2*"))
        repo = matches[0] if matches else None
    if repo is None:
        return "torch.hub.load(facebookresearch/dinov2, dinov2_vitb14) returned a model; repo path not found"
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    digest = proc.stdout.strip() or "no git hash in hub checkout"
    return f"{repo} @ {digest}"


def dermatology(model, transform, device, rows: list[dict], hashes: dict[str, str]) -> tuple[np.ndarray, tuple]:
    directory = isic_directory() or download_isic()
    id_paths = isic_paths(directory)
    new_paths = pad_paths()
    smoke_id = subsample(id_paths, SMOKE_N)
    smoke_new = subsample(new_paths, SMOKE_N)
    hashes["dermatology_smoke_id"] = write_list("dermatology_smoke_id", smoke_id)
    hashes["dermatology_smoke_new"] = write_list("dermatology_smoke_new", smoke_new)
    x_smoke_id = embed_paths(smoke_id, model, transform, device)
    x_smoke_new = embed_paths(smoke_new, model, transform, device)
    smoke_row, _smoke_fit = row_for("dermatology", "smoke", x_smoke_id, x_smoke_new)
    rows.append(smoke_row)
    write_results(rows)
    gap_ok = abs(float(smoke_row["gap"])) <= 1e-6
    append_gate("dermatology_smoke", 0 if gap_ok else 1)
    if not gap_ok:
        stop("dermatology smoke Mahalanobis/Gaussian gap exceeded 1e-6")
    if smoke_row["pass"] != "yes":
        stop("dermatology smoke characterization failed")
    full_id = subsample(id_paths, CAP)
    full_new = subsample(new_paths, CAP)
    hashes["dermatology_id"] = write_list("dermatology_id", full_id)
    hashes["dermatology_new"] = write_list("dermatology_new", full_new)
    x_id = embed_paths(full_id, model, transform, device)
    x_new = embed_paths(full_new, model, transform, device)
    full_row, fitted = row_for("dermatology", "isic2019_padufes", x_id, x_new)
    rows.append(full_row)
    write_results(rows)
    append_gate("dermatology_full", 0 if full_row["pass"] == "yes" else 1)
    sigma, precision, mu = fitted
    noisy = control_cloud(x_id)
    mu_new = arithmetic_mean(noisy)
    control_row, _unused = row_for(
        "dermatology",
        "control",
        x_id,
        noisy,
        fitted=(sigma, precision),
    )
    # evaluate_mapped_split recomputes mu from x_id. mu_new is the noisy mean. fitted Sigma is not refit.
    del mu_new, mu
    rows.append(control_row)
    write_results(rows)
    inverted = float(control_row["auroc_mahalanobis"]) >= 0.5
    append_gate("dermatology_control", 1 if inverted or control_row["pass"] != "yes" else 0)
    if inverted:
        stop("control inverted: Mahalanobis AUROC is not below 1/2")
    if full_row["pass"] != "yes" or control_row["pass"] != "yes":
        stop("dermatology characterization failed")
    return x_id, (sigma, precision)


def camelyon_path(root: Path, patient, node, x_coord, y_coord) -> Path:
    patient_i = int(patient)
    node_i = int(node)
    x_i = int(float(x_coord))
    y_i = int(float(y_coord))
    folder = f"patient_{patient_i:03d}_node_{node_i}"
    name = f"patch_patient_{patient_i:03d}_node_{node_i}_x_{x_i}_y_{y_i}.png"
    return root / "patches" / folder / name


def pathology(model, transform, device, rows: list[dict], hashes: dict[str, str]) -> None:
    try:
        import inspect
        from wilds import get_dataset
    except Exception as exc:
        print(type(exc).__name__, exc)
        append_gate("pathology", 1)
        stop("wilds import failed")
    signature = inspect.signature(get_dataset)
    kwargs = {"dataset": "camelyon17", "download": True}
    accepts_var_keyword = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )
    if "root_dir" in signature.parameters or accepts_var_keyword:
        kwargs["root_dir"] = str(WILDS_ROOT)
    elif "root" in signature.parameters:
        kwargs["root"] = str(WILDS_ROOT)
    elif "data_dir" in signature.parameters:
        kwargs["data_dir"] = str(WILDS_ROOT)
    else:
        append_gate("pathology", 1)
        stop("get_dataset does not accept a data root")
    try:
        dataset = get_dataset(**kwargs)
    except Exception as exc:
        print(type(exc).__name__, exc)
        if WILDS_ROOT.is_dir():
            print("wilds top-level listing:")
            for entry in sorted(WILDS_ROOT.iterdir()):
                print(entry.name)
        append_gate("pathology", 1)
        stop("camelyon17 get_dataset failed")
    split_dict = getattr(dataset, "split_dict", None)
    print("split_dict", split_dict)
    if not isinstance(split_dict, dict) or "train" not in split_dict or "test" not in split_dict:
        print(split_dict)
        append_gate("pathology", 1)
        stop("train or test is absent from split_dict")
    try:
        train = dataset.get_subset("train")
        test = dataset.get_subset("test")
    except Exception as exc:
        print(split_dict)
        print(type(exc).__name__, exc)
        append_gate("pathology", 1)
        stop("get_subset train or test failed")
    if len(train) == 0 or len(test) == 0:
        print(split_dict)
        append_gate("pathology", 1)
        stop("train or test subset is empty")
    data_dir = Path(getattr(dataset, "data_dir", WILDS_ROOT / "camelyon17_v1.0"))
    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.is_file():
        print("camelyon top-level listing:")
        if data_dir.is_dir():
            for entry in sorted(data_dir.iterdir()):
                print(entry.name)
        append_gate("pathology", 1)
        stop("camelyon17 metadata.csv is absent")
    with metadata_path.open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    def paths_for(subset) -> list[str]:
        built = []
        for idx in list(subset.indices):
            row = table[int(idx)]
            path = camelyon_path(data_dir, row["patient"], row["node"], row["x_coord"], row["y_coord"])
            built.append(str(path))
        return built
    try:
        id_paths = paths_for(train)
        new_paths = paths_for(test)
    except Exception as exc:
        print(type(exc).__name__, exc)
        append_gate("pathology", 1)
        stop("could not build camelyon paths from the official subset indices")
    missing = [path for path in id_paths[:5] + new_paths[:5] if not Path(path).is_file()]
    if missing:
        print("missing patch files:")
        for path in missing:
            print(path)
        print("patches listing:")
        patch_root = data_dir / "patches"
        if patch_root.is_dir():
            for entry in sorted(patch_root.iterdir())[:20]:
                print(entry.name)
        append_gate("pathology", 1)
        stop("camelyon path pattern does not match the archive")
    id_keep = subsample(id_paths, CAP)
    new_keep = subsample(new_paths, CAP)
    hashes["pathology_id"] = write_list("pathology_id", id_keep)
    hashes["pathology_new"] = write_list("pathology_new", new_keep)
    x_id = embed_paths(id_keep, model, transform, device)
    x_new = embed_paths(new_keep, model, transform, device)
    record, _fitted = row_for("pathology", "camelyon17_train_test", x_id, x_new)
    rows.append(record)
    write_results(rows)
    append_gate("pathology", 0 if record["pass"] == "yes" else 1)
    if record["pass"] != "yes":
        stop("pathology characterization failed")


def main() -> None:
    require_toy()
    rows: list[dict] = []
    hashes: dict[str, str] = {}
    try:
        model, transform, device = load_encoder()
    except SystemExit:
        manifest(hashes, "hub load failed; see the gate exception above")
        raise
    hub = hub_resolution()
    try:
        dermatology(model, transform, device, rows, hashes)
        pathology(model, transform, device, rows, hashes)
    finally:
        manifest(hashes, hub)
    append_gate("part_c", 0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code:
            append_gate("part_c", code)
        raise
