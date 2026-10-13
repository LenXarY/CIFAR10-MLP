"""Build the submission ZIP: train_models.ipynb + the three model files."""

import os
import sys
import zipfile


FILES = [
    "train_models.ipynb",
    "model_seed42.joblib",
    "model_seed2026.joblib",
    "model_seed3102.joblib",
]


def main():
    if len(sys.argv) < 2:
        print("usage: python work/package.py \"NAME, Givenname-STUDENTID\"")
        return 1
    stem = sys.argv[1].strip()
    zip_name = f"{stem}-AIAA3102-Assignment1.zip"

    missing = [f for f in FILES if not os.path.exists(f)]
    if missing:
        print("missing files:", missing)
        return 1

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in FILES:
            zf.write(name, os.path.basename(name))

    print(f"wrote {zip_name} ({os.path.getsize(zip_name) / 1e6:.1f} MB)")
    with zipfile.ZipFile(zip_name) as zf:
        for info in zf.infolist():
            print(f"  {info.filename:26s} {info.file_size / 1e6:8.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
