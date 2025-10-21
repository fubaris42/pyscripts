#!/usr/bin/env python3

import os
import fitz  # PyMuPDF
import concurrent.futures


def extract_first_page(pdf_path):
    """
    Opens a PDF, extracts its first page, and returns it as a new
    in-memory PDF document's byte representation. This function is designed
    to be run in a separate thread.

    Args:
        pdf_path (str): The file path to the PDF.

    Returns:
        bytes: The byte string of a new PDF containing only the first page,
               or None if the PDF is empty or an error occurs.
    """
    try:
        # Open the source PDF using a 'with' statement to ensure it's closed
        with fitz.open(pdf_path) as doc:
            if len(doc) > 0:
                # Create a new, empty PDF document in memory
                new_doc = fitz.open()
                # Insert the first page (page 0) of the source document
                new_doc.insert_pdf(doc, from_page=0, to_page=0)

                # Save the new single-page document to a byte buffer
                pdf_bytes = new_doc.tobytes()
                new_doc.close()

                # Use os.path.basename to get just the filename for cleaner output
                print(f"Processed first page of {os.path.basename(pdf_path)}")
                return pdf_bytes
            else:
                print(f"Skipping empty PDF: {os.path.basename(pdf_path)}")
                return None
    except Exception as e:
        # Catch potential errors during file processing
        print(f"Error processing {os.path.basename(pdf_path)}: {e}")
        return None


def main():
    """
    Finds all PDFs in the current directory, extracts the first page of each
    using multiple threads, and merges them into a single output file.
    """
    pdf_dir = "."
    output_file = "merged.pdf"

    # Find and sort all PDF files in the current directory
    try:
        # Create a list of full file paths for processing
        pdf_files = [
            os.path.join(pdf_dir, f)
            for f in os.listdir(pdf_dir)
            if f.lower().endswith(".pdf")
        ]
        pdf_files.sort()

        if not pdf_files:
            print("No PDF files found in the current directory.")
            return

    except OSError as e:
        print(f"Error accessing directory '{pdf_dir}': {e}")
        return

    print(f"Found {len(pdf_files)} PDF files to process.")

    # Use a ThreadPoolExecutor to process files in parallel. The 'with'
    # statement ensures threads are cleaned up properly.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # The map function applies 'extract_first_page' to each item in 'pdf_files'.
        # It processes them in parallel and returns the results in the same order.
        results = executor.map(extract_first_page, pdf_files)

        # Collect the results, filtering out any 'None' values from failed
        # operations or empty PDFs.
        first_pages_bytes = [result for result in results if result is not None]

    if not first_pages_bytes:
        print(
            "No pages were successfully extracted. The output file will not be created."
        )
        return

    print("\nAll files processed. Now merging...")

    # Create the final document for merging
    merged_doc = fitz.open()

    # Sequentially merge the single-page PDFs from their byte representations
    for pdf_bytes in first_pages_bytes:
        # Open the single-page PDF from the byte stream
        with fitz.open(stream=pdf_bytes, filetype="pdf") as single_page_doc:
            merged_doc.insert_pdf(single_page_doc)

    try:
        # Save the final merged document with optimizations
        merged_doc.save(output_file, garbage=4, deflate=True, clean=True)
        print(
            f"\nSuccessfully merged {len(first_pages_bytes)} pages into '{output_file}'"
        )
    except Exception as e:
        print(f"Error saving the final PDF: {e}")
    finally:
        # Ensure the final document object is closed
        merged_doc.close()


if __name__ == "__main__":
    main()
