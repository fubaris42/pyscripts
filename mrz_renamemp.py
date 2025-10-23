#!/usr/bin/env python3

# TODO : SIMPLIFY & USE THREADING

import re
import warnings
from pathlib import Path
from PIL import Image, ImageOps
from fastmrz import FastMRZ
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

warnings.filterwarnings("ignore", category=UserWarning)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
fast_mrz = FastMRZ()  # Global instance


def preprocess_image(path: Path) -> Path:
    image = Image.open(path).convert("L")
    image = ImageOps.autocontrast(image)
    image = image.point(lambda x: 0 if x < 120 else 255).convert("RGB")

    tmp_path = path.with_name(f"tmp_{path.name}")
    image.save(tmp_path)
    return tmp_path


def rename_file_with_mrz(file_path: Path):
    try:
        tmp_img = preprocess_image(file_path)
        mrz_raw = fast_mrz.get_details(str(tmp_img), ignore_parse=True)
        tmp_img.unlink(missing_ok=True)

        if not isinstance(mrz_raw, str):
            return f"[!] MRZ not found: {file_path.name}"

        name = re.sub(
            r"[^A-Z0-9_]", "", mrz_raw.replace("\n", "_").replace("<", "_").upper()
        )
        name = re.sub(r"_+", "_", name).strip("_")
        new_name = f"{name}{file_path.suffix.lower()}"
        new_path = file_path.with_name(new_name)

        counter = 1
        while new_path.exists():
            new_name = f"{name}_{counter}{file_path.suffix.lower()}"
            new_path = file_path.with_name(new_name)
            counter += 1

        file_path.rename(new_path)
        return f"[✓] {file_path.name} → {new_name}"
    except Exception as e:
        return f"[!] Error: {file_path.name} → {e}"


def main():
    files = [
        f
        for f in Path.cwd().iterdir()
        if f.suffix.lower() in SUPPORTED_EXT and f.is_file()
    ]
    if not files:
        print("No supported image files found.")
        return

    with ProcessPoolExecutor(cpu_count()) as pool:
        for msg in pool.map(rename_file_with_mrz, files):
            print(msg)


main()
