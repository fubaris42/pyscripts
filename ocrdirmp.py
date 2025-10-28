#!/usr/bin/env python3

# OPTIMIZED SCRIPT
# 1. Uses 80% of CPU count for multiprocessing Pool.
# 2. Reduces DPI from 300 to 200 for faster image rendering and OCR speed.
# 3. Includes Tesseract Page Segmentation Mode (PSM 3) for efficient page analysis.

import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from multiprocessing import Pool, cpu_count
import math

INPUT_DIR = "."
OUTPUT_DIR = "ocr_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_pdf_searchable(pdf_path):
    """Checks if the PDF contains extractable text."""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            # Check for non-empty text content
            if page.get_text().strip():
                doc.close()  # Close resource early
                return True
        doc.close()  # Close resource
        return False
    except Exception as e:
        print(f"[ERROR] Checking text in {pdf_path}: {e}")
        # Fail safe: assume searchable to avoid unnecessarily re-OCR'ing on error
        return True


def ocr_pdf(pdf_file):
    """Performs OCR on a non-searchable PDF and saves the result."""
    input_path = os.path.join(INPUT_DIR, pdf_file)
    output_path = os.path.join(OUTPUT_DIR, pdf_file)

    if is_pdf_searchable(input_path):
        return f"[SKIP] Already searchable: {pdf_file}"

    print(f"[START OCR] {pdf_file}")
    try:
        doc = fitz.open(input_path)
        new_pdf = fitz.open()

        # Tesseract configuration for efficiency
        # PSM 3: Fully automatic page segmentation, but no orientation and script detection
        custom_config = r"--psm 3"

        for page_num, page in enumerate(doc):
            # OPTIMIZATION: Reduce DPI from 300 to 200 for speed
            pix = page.get_pixmap(dpi=200)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))

            # Use Tesseract to get an OCR'ed PDF page
            pdf_bytes = pytesseract.image_to_pdf_or_hocr(
                image, extension="pdf", config=custom_config
            )

            ocr_page = fitz.open("pdf", pdf_bytes)
            new_pdf.insert_pdf(ocr_page)
            # print(f"  -> Page {page_num + 1}/{len(doc)} processed in {pdf_file}") # Remove page print for cleaner output

        new_pdf.save(output_path)
        new_pdf.close()
        doc.close()
        return f"[OCR DONE] {pdf_file}"
    except Exception as e:
        # Ensure files are closed even on error
        try:
            if "doc" in locals():
                doc.close()
            if "new_pdf" in locals():
                new_pdf.close()
        except:
            pass  # Ignore errors during cleanup
        return f"[ERROR] Failed to OCR {pdf_file}: {e}"


def main():
    """Main function to find PDFs and parallelize the OCR process."""
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]

    # Calculate 80% of available CPU cores
    total_cpus = cpu_count()
    # Use math.ceil to round up, max(1, ...) ensures at least one process
    process_count = max(1, math.ceil(total_cpus * 0.80))

    print(f"Total CPUs available: {total_cpus}")
    print(f"Using {process_count} processes for multiprocessing pool (80%).")

    if not pdf_files:
        print("No PDF files found in the current directory.")
        return

    # Initialize and run the multiprocessing pool
    with Pool(processes=process_count) as pool:
        # map applies ocr_pdf to every item in pdf_files list
        results = pool.map(ocr_pdf, pdf_files)

    # Print the final results
    print("\n--- Summary ---")
    for res in results:
        print(res)


if __name__ == "__main__":
    main()
