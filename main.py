# main.py
# semantic search imports (add near top)
#from sentence_transformers import SentenceTransformer, util
import numpy as np
import time
import os
import re
import html
import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import pandas as pd
from rdflib import Graph, RDFS

# -------------------------
# Configuration / folders
# -------------------------
ROOT = os.path.abspath(os.path.dirname(__file__))
PDF_FOLDER = os.path.join(ROOT, "pdfs")
TEXT_FOLDER = os.path.join(ROOT, "texts")
CSV_FOLDER = os.path.join(ROOT, "csv")
OWL_PATH = os.path.join(ROOT, "combined_provision_classes.owl")

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(TEXT_FOLDER, exist_ok=True)
os.makedirs(CSV_FOLDER, exist_ok=True)

# -------------------------
# Helpers: extraction & cleaning
# -------------------------
def remove_page_numbers(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # skip isolated page numbers (1, 23) of up to 3 digits
        if not (stripped.isdigit() and len(stripped) <= 3):
            cleaned.append(line)
    return "\n".join(cleaned)

def ocr_page(page):
    pix = page.get_pixmap(dpi=300)
    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(img)

def extract_text_from_pdf(pdf_path: str, use_ocr_if_empty: bool = True) -> str:
    doc = fitz.open(pdf_path)
    all_text = ""
    for page in doc:
        try:
            text = page.get_text()
        except Exception:
            text = ""
        if not text.strip() and use_ocr_if_empty:
            try:
                text = ocr_page(page)
            except Exception:
                text = ""
        cleaned = remove_page_numbers(text)
        all_text += cleaned + "\n"
    doc.close()
    return all_text

# -------------------------
# Text -> CSV (section parsing)
# -------------------------
section_pattern = re.compile(r"^(\d+[A-Za-z]?)\.\s+(.*)")

def text_to_csv(txt_path: str, act_name: str):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f if ln.strip()]

    sections = []
    for i, line in enumerate(lines):
        m = section_pattern.match(line)
        if m:
            sections.append((i, m.group(1), m.group(2)))

    results = []
    for idx, (start_i, sec_num, sec_title) in enumerate(sections):
        end_i = sections[idx+1][0] if idx+1 < len(sections) else len(lines)
        desc_lines = lines[start_i+1:end_i]
        if desc_lines and section_pattern.match(desc_lines[0]):
            desc_lines = desc_lines[1:]
        description = " ".join(desc_lines).strip()
        results.append([act_name, f"Section {sec_num}", sec_title.strip(), description])

    df = pd.DataFrame(results, columns=["Act", "Section", "Title", "Description"])
    out_csv = os.path.join(CSV_FOLDER, f"{act_name}_Correct_Title_Description.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return out_csv, df

# -------------------------
# CSV -> OWL
# -------------------------
def csvs_to_owl(csv_paths, out_path=OWL_PATH):
    provision_classes = []
    for p in csv_paths:
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        act_name = str(df["Act"].iloc[0]) if not df.empty else os.path.splitext(os.path.basename(p))[0]
        for _, row in df.iterrows():
            desc = str(row.get("Description", "")).strip()
            if not desc or len(desc.split()) <= 10:
                # skip very short descriptions
                continue
            title = str(row.get("Title", "")).strip()
            full_desc = html.escape(f"{title}. {desc}")
            section_num = str(row.get("Section", "")).replace("Section", "").strip()
            class_id = f"Section_{section_num}_{act_name.replace(' ', '_')}"
            provision = f"""
    <owl:Class rdf:about="http://www.w3id.org/def/IndianLaw#{class_id}">
        <rdfs:subClassOf rdf:resource="http://www.w3id.org/def/nyon#Provision"/>
        <rdfs:comment xml:lang="en">{full_desc}</rdfs:comment>
        <rdfs:label xml:lang="en">{class_id}</rdfs:label>
    </owl:Class>
            """
            provision_classes.append(provision)

    owl_content = f"""<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
{''.join(provision_classes)}
</rdf:RDF>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(owl_content)
    return out_path

# -------------------------
# SPARQL search
# -------------------------
# -------------------------
# Semantic search helpers
# -------------------------
@st.cache_resource
def load_sentence_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load the sentence-transformers model once per session."""
    return SentenceTransformer(model_name)


def build_corpus_from_owl(owl_path):
    """
    Parse the ontology and return a list of entries:
    each entry is a dict: {'id': uri, 'label': label, 'comment': comment}
    """
    g = Graph()
    g.parse(owl_path)
    q = """
     SELECT ?s ?label ?comment WHERE {
       ?s rdfs:label ?label .
       OPTIONAL { ?s rdfs:comment ?comment . }
     }
     """
    entries = []
    for row in g.query(q, initNs={"rdfs": RDFS}):
        uri = str(row.s)
        label = str(row.label)
        comment = str(row.comment or "")
        # merge label into text so short comments can be matched
        text = f"{label}. {comment}".strip()
        entries.append({"id": uri, "label": label, "comment": comment, "text": text})
    return entries


@st.cache_data(show_spinner=False)
def build_embeddings(entries, model_name="all-MiniLM-L6-v2"):
    """
    Build embeddings for the entries list. Returns (texts, embeddings)
    caching avoids recomputing on reruns.
    """
    model = load_sentence_model(model_name)
    texts = [e["text"] for e in entries]
    if not texts:
        return [], np.array([])
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    return texts, embeddings


def semantic_search(entries, embeddings, query, top_k=5, model_name="all-MiniLM-L6-v2"):
    """
    Return top_k entries most similar to the query (by cosine similarity).
    embeddings: tensor
    """
    model = load_sentence_model(model_name)
    q_emb = model.encode(query, convert_to_tensor=True)
    # use sentence-transformers util to compute cos sim
    cos_scores = util.cos_sim(q_emb, embeddings)[0]  # shape: (N,)
    top_k = min(top_k, len(entries))
    results = util.semantic_search(q_emb, embeddings, top_k=top_k)[0]  # returns indices+scores
    # convert to (entry, score)
    out = []
    for r in results:
        idx = r["corpus_id"]
        score = float(r["score"])
        out.append((entries[idx], score))
    return out


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="LAKSHYA", layout="wide")
st.title("LAKSHYA - Legal Acts to Knowledge System for Hybrid Q&A")

# Top: Tesseract settings
with st.sidebar.expander("Tesseract / OCR settings (optional)"):
    st.write("If Tesseract is not on your system PATH, paste the path to the executable here.")
    tpath = st.text_input("Tesseract executable path (e.g. C:\\Program Files\\Tesseract-OCR\\tesseract.exe)")
    if tpath:
        pytesseract.pytesseract.tesseract_cmd = tpath
        st.success("Tesseract path set for this session.")
    else:
        st.write("Using system `tesseract` (if available).")

tabs = st.tabs(["PDF → Text", "Text → CSV", "CSV → Ontology", "Ontology QA", "Files & Downloads"])

# Tab 1: PDF -> Text
with tabs[0]:
    st.header("Extract text from PDF (PyMuPDF + OCR fallback)")
    st.markdown("Drop PDFs into the `pdfs/` folder or use the uploader below.")
    uploaded = st.file_uploader("Upload PDF (optional)", type=["pdf"])
    if uploaded:
        save_path = os.path.join(PDF_FOLDER, uploaded.name)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved to `{save_path}`")

    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
    chosen = st.selectbox("Choose PDF to extract", [""] + pdfs)
    use_ocr = st.checkbox("Use OCR when page has no extractable text (recommended)", value=True)
    if st.button("Extract Text") and chosen:
        pdf_path = os.path.join(PDF_FOLDER, chosen)
        try:
            with st.spinner("Extracting..."):
                text = extract_text_from_pdf(pdf_path, use_ocr_if_empty=use_ocr)
            out_txt = os.path.join(TEXT_FOLDER, chosen.replace(".pdf", ".txt"))
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(text)
            st.success(f"Saved text to `{out_txt}`")
            st.text_area("Extracted text (preview)", text[:20000], height=400)
        except Exception as e:
            st.error(f"Extraction failed: {e}")

# Tab 2: Text -> CSV
with tabs[1]:
    st.header("Convert extracted text into CSV (numbered sections)")
    txt_files = [f for f in os.listdir(TEXT_FOLDER) if f.lower().endswith(".txt")]
    chosen_txt = st.selectbox("Choose .txt file", [""] + txt_files)
    if st.button("Generate CSV") and chosen_txt:
        txt_path = os.path.join(TEXT_FOLDER, chosen_txt)
        act_name = os.path.splitext(chosen_txt)[0]
        try:
            csv_path, df = text_to_csv(txt_path, act_name)
            st.success(f"CSV created: `{csv_path}`")
            st.dataframe(df)
            with open(csv_path, "rb") as fh:
                st.download_button("Download CSV", fh, file_name=os.path.basename(csv_path))
        except Exception as e:
            st.error(f"CSV generation failed: {e}")

# Tab 3: CSV -> Ontology
with tabs[2]:
    st.header("Create OWL ontology from CSV files")
    csv_files = [f for f in os.listdir(CSV_FOLDER) if f.lower().endswith(".csv")]
    chosen_csvs = st.multiselect("Choose CSV files", csv_files)
    if st.button("Create Ontology") and chosen_csvs:
        csv_paths = [os.path.join(CSV_FOLDER, c) for c in chosen_csvs]
        try:
            with st.spinner("Generating ontology..."):
                out = csvs_to_owl(csv_paths)
            st.success(f"Ontology written to `{out}`")
            with open(out, "r", encoding="utf-8") as f:
                preview = f.read(2000)
            st.code(preview + ("\n\n... (truncated)" if os.path.getsize(out) > len(preview) else ""))
            with open(out, "rb") as fh:
                st.download_button("Download OWL", fh, file_name=os.path.basename(out))
        except Exception as e:
            st.error(f"Failed to create ontology: {e}")

# Tab 4: Ontology QA
# ----------------- Tab 4: Ontology QA (semantic + keyword fallback) -----------------
with tabs[3]:
    st.header("Ask natural-language questions (semantic search over the ontology)")

    # Load / prepare corpus & embeddings (cached)
    if not os.path.exists(OWL_PATH):
        st.warning("Ontology file not found. Create it first (CSV → Ontology).")
    else:
        # build entries and embeddings (cached)
        with st.spinner("Loading ontology and building embeddings..."):
            entries = build_corpus_from_owl(OWL_PATH)
            texts, embeddings = build_embeddings(entries)

        if not entries:
            st.info("Ontology is empty or has no label/comment data.")
        else:
            user_q = st.text_input("Ask a question in natural language (e.g. 'What happens if the due date is a holiday?')")

            col1, col2 = st.columns([1, 3])
            search_button = col1.button("Search")
            top_k = col2.slider("Results", min_value=1, max_value=10, value=3, help="How many top answers to show")

            if search_button and user_q:
                # 1) semantic search
                try:
                    with st.spinner("Running semantic search..."):
                        sem_results = semantic_search(entries, embeddings, user_q, top_k=top_k)

                    if sem_results:
                        st.success("Top semantic matches")
                        for entry, score in sem_results:
                            st.subheader(entry["label"])
                            st.write(entry["comment"])
                            st.caption(f"Similarity score: {score:.3f}")
                            st.markdown("---")
                    else:
                        st.info("No semantic matches found.")

                    # 2) also run a keyword SPARQL fallback (optional)
                    kw = " ".join(user_q.lower().split())  # collapse spaces
                    kw_short = kw.split()[:6]  # try first few words joined for fallback
                    fallback_kw = " ".join(kw_short)

                    # run your safe SPARQL filter search (as earlier)
                    sp_results = search_ontology(OWL_PATH, fallback_kw)
                    if sp_results:
                        st.success("Keyword-based SPARQL matches (fallback)")
                        for label, comment in sp_results:
                            st.write(f"**{label}**")
                            st.write(comment)
                            st.markdown("---")
                except Exception as e:
                    st.error(f"Search failed: {e}")

# Tab 5: Files & Downloads
with tabs[4]:
    st.header("Project files")
    st.write("PDFs:")
    for f in sorted(os.listdir(PDF_FOLDER)):
        st.write(f"- {f}")
    st.write("Texts:")
    for f in sorted(os.listdir(TEXT_FOLDER)):
        st.write(f"- {f}")
    st.write("CSVs:")
    for f in sorted(os.listdir(CSV_FOLDER)):
        st.write(f"- {f}")
    if os.path.exists(OWL_PATH):
        st.write("Ontology:", OWL_PATH)
        with open(OWL_PATH, "rb") as fh:
            st.download_button("Download OWL", fh, file_name=os.path.basename(OWL_PATH))
    st.write("Files are stored in the project's pdfs/, texts/, csv/ folders.")
