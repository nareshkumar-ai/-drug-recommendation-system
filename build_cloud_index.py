"""
Build Cloud FAISS Index for Gemini RAG (Offline Precomputation Script)
Strict pre-validation of CSV data before making any API requests:
  - Validates CSV existence, readable formatting, and required columns.
  - Confirms exactly 41 unique, non-null, non-empty disease profiles matching VALID_DISEASE_SET.
  - Rejects null, NaN, duplicate, or empty content rows.
  - Generates deterministic document IDs ("0" through "40") strictly aligned with vector index rows.
  - Writes:
      - vectorstore/faiss_gemini/index.faiss (C++ binary unit-normalized 768d IndexFlatIP)
      - vectorstore/faiss_gemini/documents.json (Pure UTF-8 JSON metadata, ZERO pickle)
      - vectorstore/faiss_gemini/checksums.json (SHA-256 verification hashes)

NOTE: This script is designed for offline local precomputation. DO NOT run without explicit approval.
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def normalize_disease_name(name):
    """
    Canonical disease normalization:
    - Strips leading and trailing whitespace.
    - Collapses internal whitespace runs to a single space.
    - Converts to lowercase for case-insensitive matching.
    - Resolves known label typo variants ('peptic ulcer diseae' -> 'peptic ulcer disease').
    """
    if not isinstance(name, str):
        return ""
    norm = " ".join(name.strip().split()).lower()
    if norm == "peptic ulcer diseae":
        norm = "peptic ulcer disease"
    return norm

# Whitelist of exactly 41 disease classes matching the ML training classes
RAW_ML_DISEASES = [
    'Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
    'Peptic ulcer diseae', 'AIDS', 'Diabetes ', 'Gastroenteritis', 'Bronchial Asthma',
    'Hypertension ', 'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)',
    'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
    'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis',
    'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
    'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
    'Osteoarthristis', 'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne',
    'Urinary tract infection', 'Psoriasis', 'Impetigo'
]

CANONICAL_DISEASE_SET = {normalize_disease_name(d) for d in RAW_ML_DISEASES}

REQUIRED_COLUMNS = ["Disease", "Description", "Diet", "Medication", "workout", "Precautions"]

def validate_preprocessed_csv(data_path):
    """Strict pre-validation of CSV data before any embedding or API calls."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Knowledge base CSV not found at: {data_path}")

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV at {data_path}: {type(e).__name__}")

    # 1. Column existence
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    # 2. Record count check
    if len(df) != 41:
        raise ValueError(f"CSV contains {len(df)} records, expected exactly 41")

    # 3. Check for null, NaN, or empty values in any required column
    for col in REQUIRED_COLUMNS:
        null_count = df[col].isna().sum()
        if null_count > 0:
            raise ValueError(f"Column '{col}' contains {null_count} null/NaN values")

        # Check for whitespace-only strings
        empty_mask = df[col].astype(str).str.strip().eq("")
        if empty_mask.any():
            bad_rows = df[empty_mask].index.tolist()
            raise ValueError(f"Column '{col}' contains empty or whitespace-only values at rows {bad_rows}")

    # 4. Disease whitelist & duplicate verification via canonical normalization
    seen_canonical = set()
    for idx, row in df.iterrows():
        raw_disease = str(row['Disease'])
        canonical_dis = normalize_disease_name(raw_disease)
        if not canonical_dis:
            raise ValueError(f"Row {idx} contains empty disease name")
        if canonical_dis not in CANONICAL_DISEASE_SET:
            raise ValueError(f"Row {idx} contains unwhitelisted disease: '{raw_disease}' (canonical: '{canonical_dis}')")
        if canonical_dis in seen_canonical:
            raise ValueError(f"Duplicate disease profile in CSV: '{raw_disease}' (canonical: '{canonical_dis}')")
        seen_canonical.add(canonical_dis)

    if len(seen_canonical) != 41:
        raise ValueError(f"CSV contains {len(seen_canonical)} unique canonical diseases, expected exactly 41")

    missing_expected = CANONICAL_DISEASE_SET - seen_canonical
    if missing_expected:
        raise ValueError(f"CSV missing expected canonical diseases: {missing_expected}")

    print(f"[PRE-CHECK PASSED] CSV validated: exactly 41 unique, complete, non-null disease profiles verified.")
    return df

def generate_cloud_index():
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, "LLM Preprocessed Data.csv")
    output_dir = os.path.join(base_path, "vectorstore", "faiss_gemini")

    # API Key check
    api_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] GEMINI_API_KEY environment variable is not set.")
        print("Please set GEMINI_API_KEY before running this script.")
        sys.exit(1)

    # 1. Run strict pre-validation before making any API requests
    print("[1/5] Running strict pre-validation on knowledge base CSV...")
    try:
        df = validate_preprocessed_csv(data_path)
    except Exception as e:
        print(f"[PRE-CHECK FAILED] Validation error: {type(e).__name__}")
        print("Aborting index generation without making any API calls.")
        sys.exit(1)

    # 2. Build deterministic documents and texts
    print("[2/5] Constructing deterministic document records...")
    documents = []
    texts_to_embed = []
    for idx, row in df.iterrows():
        content = (
            f"Disease: {str(row['Disease']).strip()}\n"
            f"Description: {str(row['Description']).strip()}\n"
            f"Diet: {str(row['Diet']).strip()}\n"
            f"Medication: {str(row['Medication']).strip()}\n"
            f"Workout: {str(row['workout']).strip()}\n"
            f"Precautions: {str(row['Precautions']).strip()}"
        )
        canonical_dis = normalize_disease_name(row["Disease"])
        texts_to_embed.append(content)
        documents.append({
            "id": str(idx),  # Deterministic string ID matching row index ("0" to "40")
            "page_content": content,
            "metadata": {
                "disease": str(row["Disease"]).strip(),
                "canonical_disease": canonical_dis
            }
        })

    # 3. Initialize Embeddings & Generate vectors
    print("[3/5] Initializing GoogleGenerativeAIEmbeddings (models/gemini-embedding-001, 768d)...")
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
            output_dimensionality=768,
            task_type="RETRIEVAL_DOCUMENT"
        )
    except ImportError:
        print("[ERROR] langchain-google-genai is not installed in the active environment.")
        sys.exit(1)

    print(f"Embedding {len(texts_to_embed)} disease documents in batch...")
    raw_embeddings = embeddings_model.embed_documents(texts_to_embed)
    vectors = np.array(raw_embeddings, dtype="float32")

    if vectors.shape != (41, 768):
        print(f"[ERROR] Unexpected vector shape: {vectors.shape}, expected (41, 768)")
        sys.exit(1)

    # Unit-normalize vectors for exact Cosine Similarity with Inner Product (IndexFlatIP)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_vectors = vectors / norms

    # 4. Build FAISS IndexFlatIP
    print("[4/5] Building FAISS IndexFlatIP (768 dimensions)...")
    import faiss
    index = faiss.IndexFlatIP(768)
    index.add(normalized_vectors)

    if index.ntotal != 41 or index.d != 768:
        print(f"[ERROR] FAISS index validation failed: ntotal={index.ntotal}, d={index.d}")
        sys.exit(1)

    # 5. Write artifacts and compute SHA-256 checksums
    print("[5/5] Writing artifacts and generating SHA-256 checksums...")
    os.makedirs(output_dir, exist_ok=True)
    index_file = os.path.join(output_dir, "index.faiss")
    docs_file = os.path.join(output_dir, "documents.json")
    checksums_file = os.path.join(output_dir, "checksums.json")

    faiss.write_index(index, index_file)
    with open(docs_file, "w", encoding="utf-8", newline="\n") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    def get_canonical_sha256(filepath, normalize_newlines=False):
        """
        Computes SHA-256 hash. For text files (normalize_newlines=True), normalizes
        CRLF (\r\n) to canonical LF (\n) to guarantee identical hashes across operating systems.
        """
        with open(filepath, "rb") as f:
            data = f.read()
        if normalize_newlines:
            data = data.replace(b"\r\n", b"\n")
        return hashlib.sha256(data).hexdigest()

    checksums = {
        "index.faiss": get_canonical_sha256(index_file, normalize_newlines=False),
        "documents.json": get_canonical_sha256(docs_file, normalize_newlines=True),
        "record_count": 41,
        "dimension": 768,
        "metric": "IndexFlatIP_Cosine"
    }

    with open(checksums_file, "w", encoding="utf-8", newline="\n") as f:
        json.dump(checksums, f, indent=2)

    print(f"[SUCCESS] Cloud vector store written to: {output_dir}")
    print(f"  - index.faiss: {os.path.getsize(index_file):,} bytes (SHA-256: {checksums['index.faiss']})")
    print(f"  - documents.json: {os.path.getsize(docs_file):,} bytes (SHA-256: {checksums['documents.json']})")
    print(f"  - checksums.json: {os.path.getsize(checksums_file):,} bytes")

if __name__ == "__main__":
    generate_cloud_index()
