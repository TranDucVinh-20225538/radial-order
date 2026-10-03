"""Pathology only. Does not re-embed dermatology."""

import csv
import sys

from embed.__main__ import (
    COLUMNS,
    ROOT,
    append_gate,
    hub_resolution,
    load_encoder,
    manifest,
    pathology,
)

NUMERIC = {
    "n_id",
    "n_new",
    "premap_mw",
    "auroc_mahalanobis",
    "auroc_gaussian",
    "gap",
    "auroc_euclidean",
}


def load_rows() -> list[dict]:
    path = ROOT / "results.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [row["modality"] for row in rows] != ["dermatology", "dermatology", "dermatology"]:
        raise SystemExit("results.csv is not the finished dermatology table")
    parsed = []
    for row in rows:
        item = {}
        for key in COLUMNS:
            value = row[key]
            if key in NUMERIC:
                item[key] = float(value)
            else:
                item[key] = value
        parsed.append(item)
    return parsed


def load_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    in_block = False
    for line in (ROOT / "MANIFEST.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("filename_list_sha256:"):
            in_block = True
            continue
        if not in_block:
            continue
        if not line.startswith("  "):
            break
        name, digest = line.strip().split(": ", 1)
        hashes[name] = digest
    expected = {
        "dermatology_smoke_id",
        "dermatology_smoke_new",
        "dermatology_id",
        "dermatology_new",
    }
    if set(hashes) != expected:
        raise SystemExit(f"MANIFEST filename hashes are {sorted(hashes)}")
    return hashes


def main() -> None:
    rows = load_rows()
    hashes = load_hashes()
    model, transform, device = load_encoder()
    hub = hub_resolution()
    try:
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
