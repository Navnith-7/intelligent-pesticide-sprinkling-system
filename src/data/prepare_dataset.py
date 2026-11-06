from pathlib import Path
import hashlib
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "PlantVillage-Dataset" / "raw" / "color"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path for path in DATA_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(f"No supported images found in {DATA_ROOT}")

    rows = []
    for path in tqdm(image_paths, desc="Hashing images"):
        rows.append({
            "image_path": path.relative_to(PROJECT_ROOT).as_posix(),
            "label": path.parent.name,
            "file_size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    manifest = pd.DataFrame(rows)
    manifest_path = OUTPUT_DIR / "plantvillage_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    duplicates = manifest[
        manifest.duplicated("sha256", keep=False)
    ].sort_values("sha256")

    duplicates_path = OUTPUT_DIR / "exact_duplicates.csv"
    duplicates.to_csv(duplicates_path, index=False)

    print("\n=== DATASET AUDIT ===")
    print(f"Images found: {len(manifest):,}")
    print(f"Classes found: {manifest['label'].nunique()}")
    print(f"Unique image hashes: {manifest['sha256'].nunique():,}")
    print(f"Images involved in exact duplicates: {len(duplicates):,}")
    print(f"Duplicate hash groups: {duplicates['sha256'].nunique():,}")
    print(f"\nManifest saved to: {manifest_path}")
    print(f"Duplicate report saved to: {duplicates_path}")

    print("\n=== CLASS COUNTS ===")
    print(manifest["label"].value_counts().sort_index().to_string())

if __name__ == "__main__":
    main()
