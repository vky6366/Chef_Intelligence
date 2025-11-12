import os
import io
import csv
import tempfile
from pathlib import Path
import pdfplumber
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import pandas as pd

class PDFExtraction:
    def is_text_based(self, pdf_path, check_pages=3):
        try:
            reader = PdfReader(pdf_path)
            num_pages = min(check_pages, len(reader.pages))
            for i in range(num_pages):
                text = reader.pages[i].extract_text() or ""
                if text.strip():
                    return True
            return False
        except Exception:
            return False

    def extract_text_and_tables_textpdf(self, pdf_path, out_txt_path, tables_out_dir):
        os.makedirs(tables_out_dir, exist_ok=True)
        all_text = []

        with pdfplumber.open(pdf_path) as pdf:
            table_count = 0
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                all_text.append(f"=== PAGE {i} ===\n{text}\n\n")

                tables = page.extract_tables()
                for t in tables:
                    df = pd.DataFrame(t)
                    table_count += 1
                    csv_path = os.path.join(tables_out_dir, f"table_p{i}_{table_count}.csv")
                    df.to_csv(csv_path, index=False, header=False)

        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.writelines(all_text)

        return "".join(all_text) 

    def ocr_pdf_to_text_and_tables(self, pdf_path, out_txt_path, tables_out_dir, dpi=300):
        os.makedirs(tables_out_dir, exist_ok=True)
        pages = convert_from_path(pdf_path, dpi=dpi)
        all_text = []
        table_count = 0

        for i, img in enumerate(pages, start=1):
            text = pytesseract.image_to_string(img)
            all_text.append(f"=== PAGE {i} ===\n{text}\n\n")

            tsv = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
            if tsv is not None and not tsv.empty:
                tsv = tsv[tsv['text'].notna() & (tsv['text'].str.strip() != "")]
                if not tsv.empty:
                    grouped = tsv.groupby(['block_num', 'par_num', 'line_num'])
                    rows = [" ".join(g['text'].tolist()) for _, g in grouped]
                    if len(rows) > 1:
                        table_count += 1
                        csv_path = os.path.join(tables_out_dir, f"ocr_table_p{i}.csv")
                        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                            writer = csv.writer(csvfile)
                            for r in rows:
                                writer.writerow([col for col in r.split("  ") if col.strip()])

        with open(out_txt_path, "w", encoding="utf-8") as f:
            f.writelines(all_text)

        return "".join(all_text)

    def extract_pdf(self, pdf_path, output_dir="output_extraction"):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        txt_out = os.path.join(output_dir, "extracted_text.txt")
        tables_dir = os.path.join(output_dir, "tables")
        Path(tables_dir).mkdir(parents=True, exist_ok=True)

        if self.is_text_based(pdf_path):
            text = self.extract_text_and_tables_textpdf(pdf_path, txt_out, tables_dir)
        else:
            text = self.ocr_pdf_to_text_and_tables(pdf_path, txt_out, tables_dir)

        return text 

if __name__ == "__main__":
    pdf_file = r"C:\Users\vishw\Downloads\10Q-Q2-2025-as-filed.pdf"   # replace with user file
    outdir = "extracted_report"
    ext = PDFExtraction()
    res = ext.extract_pdf(pdf_file, outdir)
    print(res)
