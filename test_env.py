# test_env.py

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import fitz  # PyMuPDF

print("PyMuPDF version:", getattr(fitz, "__version__", "unknown"))

try:
    tv = pytesseract.get_tesseract_version()
    print("Tesseract binary found:", tv)
except Exception as e:
    print("Tesseract error:", e)
