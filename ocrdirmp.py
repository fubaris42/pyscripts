#!/usr/bin/env python3

import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from multiprocessing import Pool, cpu_count
from pathlib import Path

INPUT_DIR = "."
OUTPUT_DIR = "ocr_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_pdf_searchable(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            if page.get_text().strip():
                return True
        return False
    except Exception as e:
        print(f"[ERROR] Checking text in {pdf_path}: {e}")
        return True  # fail safe: assume searchable


def ocr_pdf(pdf_file):
    input_path = os.path.join(INPUT_DIR, pdf_file)
    output_path = os.path.join(OUTPUT_DIR, pdf_file)

    if is_pdf_searchable(input_path):
        return f"[SKIP] Already searchable: {pdf_file}"

    try:
        doc = fitz.open(input_path)
        new_pdf = fitz.open()

        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            pdf_bytes = pytesseract.image_to_pdf_or_hocr(image, extension="pdf")
            ocr_page = fitz.open("pdf", pdf_bytes)
            new_pdf.insert_pdf(ocr_page)

        new_pdf.save(output_path)
        return f"[OCR DONE] {pdf_file}"
    except Exception as e:
        return f"[ERROR] Failed to OCR {pdf_file}: {e}"


def main():
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(ocr_pdf, pdf_files)

    for res in results:
        print(res)


main()
