import os
import ast
import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# Backward-compatibility shim for environments where Werkzeug >= 3.0 is installed alongside Flask 2.2.5
import werkzeug
if not hasattr(werkzeug, "__version__"):
    import importlib.metadata
    try:
        werkzeug.__version__ = importlib.metadata.version("werkzeug")
    except Exception:
        werkzeug.__version__ = "2.3.8"

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load optional .env file for local development
load_dotenv()

app = Flask(__name__)
# Note: Permissive CORS(app) was reviewed and intentionally omitted.
# The frontend (index.html) and backend are strictly same-origin.

# ==========================================
# PROJECT PATHS & ENVIRONMENT CONFIG
# ==========================================

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, "dataset")
MODEL_PATH = os.path.join(BASE_PATH, "models", "drug_gboot.pkl")
LLM_DATA_PATH = os.path.join(BASE_PATH, "LLM Preprocessed Data.csv")
CLOUD_INDEX_DIR = os.path.join(BASE_PATH, "vectorstore", "faiss_gemini")

# LLM Provider: 'ollama' (default for local offline dev) or 'gemini' (for cloud deployment)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
PORT = int(os.environ.get("PORT", 5000))

# ==========================================
# CONDITIONAL IMPORTS (CLOUD VS LOCAL)
# ==========================================

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

if LLM_PROVIDER == "gemini":
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    except ImportError as e:
        print(f"[WARN] Failed to import langchain-google-genai in cloud mode: {type(e).__name__}")
        ChatGoogleGenerativeAI = None
        GoogleGenerativeAIEmbeddings = None
elif LLM_PROVIDER == "ollama":
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_ollama import OllamaLLM
    except ImportError as e:
        print(f"[WARN] Failed to import local HuggingFace/Ollama packages: {type(e).__name__}")
        HuggingFaceEmbeddings = None
        OllamaLLM = None

# ==========================================
# LOAD MACHINE LEARNING MODEL
# ==========================================

try:
    import pickle
    with open(MODEL_PATH, "rb") as file:
        gboot = pickle.load(file)
    print("ML model loaded successfully!")
except Exception as e:
    print(f"Error loading ML model: {type(e).__name__}")
    gboot = None

# ==========================================
# LOAD ML DATASETS
# ==========================================

try:
    description_df = pd.read_csv(os.path.join(DATASET_PATH, "description.csv"))
    precautions_df = pd.read_csv(os.path.join(DATASET_PATH, "precautions_df.csv"))
    medications_df = pd.read_csv(os.path.join(DATASET_PATH, "medications.csv"))
    diets_df = pd.read_csv(os.path.join(DATASET_PATH, "diets.csv"))
    workout_df = pd.read_csv(os.path.join(DATASET_PATH, "workout_df.csv"))
    print("ML datasets loaded successfully!")
except Exception as e:
    print(f"Error loading ML datasets: {type(e).__name__}")
    description_df = None
    precautions_df = None
    medications_df = None
    diets_df = None
    workout_df = None

# ==========================================
# SYMPTOM LIST (132 FEATURES IN TRAINING ORDER)
# ==========================================

symptom_list = [
    'itching','skin_rash','nodal_skin_eruptions','continuous_sneezing',
    'shivering','chills','joint_pain','stomach_pain','acidity',
    'ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition',
    'spotting_ urination','fatigue','weight_gain','anxiety',
    'cold_hands_and_feets','mood_swings','weight_loss','restlessness',
    'lethargy','patches_in_throat','irregular_sugar_level','cough',
    'high_fever','sunken_eyes','breathlessness','sweating','dehydration',
    'indigestion','headache','yellowish_skin','dark_urine','nausea',
    'loss_of_appetite','pain_behind_the_eyes','back_pain','constipation',
    'abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload',
    'swelling_of_stomach','swelled_lymph_nodes','malaise',
    'blurred_and_distorted_vision','phlegm','throat_irritation',
    'redness_of_eyes','sinus_pressure','runny_nose','congestion',
    'chest_pain','weakness_in_limbs','fast_heart_rate',
    'pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
    'irritation_in_anus','neck_pain','dizziness','cramps','bruising',
    'obesity','swollen_legs','swollen_blood_vessels',
    'puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
    'swollen_extremeties','excessive_hunger','extra_marital_contacts',
    'drying_and_tingling_lips','slurred_speech','knee_pain',
    'hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
    'movement_stiffness','spinning_movements','loss_of_balance',
    'unsteadiness','weakness_of_one_body_side','loss_of_smell',
    'bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching',
    'toxic_look_(typhos)','depression','irritability','muscle_pain',
    'altered_sensorium','red_spots_over_body','belly_pain',
    'abnormal_menstruation','dischromic _patches',
    'watering_from_eyes','increased_appetite','polyuria',
    'family_history','mucoid_sputum','rusty_sputum',
    'lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections',
    'coma','stomach_bleeding','distention_of_abdomen',
    'history_of_alcohol_consumption','fluid_overload.1',
    'blood_in_sputum','prominent_veins_on_calf','palpitations',
    'painful_walking','pus_filled_pimples','blackheads','scurring',
    'skin_peeling','silver_like_dusting','small_dents_in_nails',
    'inflammatory_nails','blister','red_sore_around_nose',
    'yellow_crust_ooze'
]

symptoms_dict = {
    symptom: i
    for i, symptom in enumerate(symptom_list)
}

# ==========================================
# DISEASE MAPPING (41 CLASSES)
# ==========================================

diseases_list = {
    15: 'Fungal infection',
    4: 'Allergy',
    16: 'GERD',
    9: 'Chronic cholestasis',
    14: 'Drug Reaction',
    33: 'Peptic ulcer diseae',
    1: 'AIDS',
    12: 'Diabetes ',
    17: 'Gastroenteritis',
    6: 'Bronchial Asthma',
    23: 'Hypertension ',
    30: 'Migraine',
    7: 'Cervical spondylosis',
    32: 'Paralysis (brain hemorrhage)',
    28: 'Jaundice',
    29: 'Malaria',
    8: 'Chicken pox',
    11: 'Dengue',
    37: 'Typhoid',
    40: 'hepatitis A',
    19: 'Hepatitis B',
    20: 'Hepatitis C',
    21: 'Hepatitis D',
    22: 'Hepatitis E',
    3: 'Alcoholic hepatitis',
    36: 'Tuberculosis',
    10: 'Common Cold',
    34: 'Pneumonia',
    13: 'Dimorphic hemmorhoids(piles)',
    18: 'Heart attack',
    39: 'Varicose veins',
    26: 'Hypothyroidism',
    24: 'Hyperthyroidism',
    25: 'Hypoglycemia',
    31: 'Osteoarthristis',
    5: 'Arthritis',
    0: '(vertigo) Paroymsal  Positional Vertigo',
    2: 'Acne',
    38: 'Urinary tract infection',
    35: 'Psoriasis',
    27: 'Impetigo'
}

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

# Whitelist of all 41 canonical disease keys (lowercase, normalized)
CANONICAL_DISEASE_SET = {normalize_disease_name(d) for d in diseases_list.values()}

# Reverse mapping from canonical name back to original ML class label (exact preservation)
CANONICAL_TO_ML_LABEL = {normalize_disease_name(d): d for d in diseases_list.values()}

# Raw set for legacy or exact label validation
VALID_DISEASE_SET = set(diseases_list.values())

# ==========================================
# RAG TELEMETRY & VECTORSTORE
# ==========================================

vectorstore = None
llm = None
disease_doc_map = {}

# Distinct states: 'uninitialized', 'initialized_unverified', 'healthy', 'degraded'
rag_telemetry = {
    "state": "uninitialized",          # 'uninitialized' | 'initialized_unverified' | 'healthy' | 'degraded'
    "reachability": "untested",        # 'untested' | 'reachable' | 'unreachable'
    "last_error_category": None,       # 'auth_error' | 'rate_limited' | 'model_error' | 'timeout' | 'server_error' | None
    "last_request_time": None
}

def verify_file_sha256(filepath, expected_hash, normalize_newlines=False):
    """
    Verifies that a file matches its expected SHA-256 hash.
    If normalize_newlines=True, normalizes CRLF (\r\n) to canonical LF (\n)
    before hashing, ensuring consistent cross-platform verification for text files
    (e.g., JSON). For binary files (e.g., FAISS index), normalize_newlines must be False.
    """
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        data = f.read()
    if normalize_newlines:
        data = data.replace(b"\r\n", b"\n")
    h.update(data)
    return h.hexdigest().lower() == expected_hash.lower()

def search_similar_documents(query_text, k=1, score_threshold=0.65):
    """
    Guarantees strict Cosine Similarity lookup:
    1. Embeds query text using the configured embeddings model.
    2. Explicitly validates vector dimension against FAISS index dimension.
    3. Rejects invalid, NaN, infinite, or zero-norm vectors safely.
    4. Explicitly unit-normalizes query vector (L2 norm = 1.0).
    5. Searches FAISS IndexFlatIP (where doc vectors are also unit-normalized).
    6. Mathematical inner product is strictly Cosine Similarity in [-1.0, 1.0].
    7. Applies strict score_threshold (>= 0.65).
    """
    if vectorstore is None or not hasattr(vectorstore, "index") or vectorstore.index is None:
        return []

    # 1. Embed query text
    raw_vec = vectorstore.embedding_function.embed_query(query_text)
    if not raw_vec or not isinstance(raw_vec, (list, np.ndarray)):
        return []

    query_arr = np.array(raw_vec, dtype="float32")
    if query_arr.ndim == 1:
        query_arr = np.expand_dims(query_arr, axis=0)

    # 2. Validate dimensions
    expected_dim = vectorstore.index.d
    if query_arr.shape[1] != expected_dim:
        print(f"[SEARCH_ERROR] Dimension mismatch: query {query_arr.shape[1]} vs index {expected_dim}")
        return []

    # 3. Check for finite values and non-zero norm
    if not np.isfinite(query_arr).all():
        return []

    norm = np.linalg.norm(query_arr, axis=1, keepdims=True)
    if norm[0, 0] <= 1e-9:
        return []

    # 4. Explicit unit normalization: guaranteed L2 norm = 1.0
    normalized_query = query_arr / norm

    # 5. Search IndexFlatIP (Inner Product)
    distances, indices = vectorstore.index.search(normalized_query, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        # In IndexFlatIP with unit-normalized vectors, dist is strictly Cosine Similarity [-1.0, 1.0]
        cosine_sim = float(dist)
        if cosine_sim >= score_threshold:
            doc_id = vectorstore.index_to_docstore_id.get(idx)
            if doc_id is not None:
                doc = vectorstore.docstore.search(doc_id)
                if isinstance(doc, Document):
                    results.append((doc, cosine_sim))

    return results

def init_rag_system():
    global vectorstore, llm, disease_doc_map, rag_telemetry
    try:
        print(f"[RAG] Initializing in '{LLM_PROVIDER}' mode...")

        if LLM_PROVIDER == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
            if not api_key:
                rag_telemetry["state"] = "uninitialized"
                rag_telemetry["last_error_category"] = "auth_error"
                print("[WARN] GEMINI_API_KEY environment variable is not set. Chatbot disabled.")
                return

            if ChatGoogleGenerativeAI is None or GoogleGenerativeAIEmbeddings is None:
                rag_telemetry["state"] = "uninitialized"
                rag_telemetry["last_error_category"] = "server_error"
                print("[WARN] langchain-google-genai is not available in environment.")
                return

            # Required cloud FAISS files
            index_path = os.path.join(CLOUD_INDEX_DIR, "index.faiss")
            docs_path = os.path.join(CLOUD_INDEX_DIR, "documents.json")
            checksums_path = os.path.join(CLOUD_INDEX_DIR, "checksums.json")

            # Reject missing files immediately
            for req_file, fpath in [("index.faiss", index_path), ("documents.json", docs_path), ("checksums.json", checksums_path)]:
                if not os.path.exists(fpath):
                    rag_telemetry["state"] = "uninitialized"
                    rag_telemetry["last_error_category"] = "server_error"
                    print(f"[WARN] Required cloud FAISS file missing: {req_file} at {CLOUD_INDEX_DIR}")
                    return

            # Read and validate checksums.json metadata
            with open(checksums_path, "r", encoding="utf-8") as f:
                cdata = json.load(f)

            for req_key in ["index.faiss", "documents.json", "record_count", "dimension", "metric"]:
                if req_key not in cdata:
                    raise ValueError(f"checksums.json missing required metadata key: '{req_key}'")

            if cdata.get("record_count") != 41:
                raise ValueError(f"checksums.json record_count is {cdata.get('record_count')}, expected 41")
            if cdata.get("dimension") != 768:
                raise ValueError(f"checksums.json dimension is {cdata.get('dimension')}, expected 768")
            if cdata.get("metric") != "IndexFlatIP_Cosine":
                raise ValueError(f"checksums.json metric is '{cdata.get('metric')}', expected 'IndexFlatIP_Cosine'")

            # Verify SHA-256 hashes against checksums.json
            if not verify_file_sha256(index_path, cdata["index.faiss"], normalize_newlines=False):
                raise ValueError("index.faiss SHA-256 hash mismatch! File corrupted.")
            if not verify_file_sha256(docs_path, cdata["documents.json"], normalize_newlines=True):
                raise ValueError("documents.json SHA-256 hash mismatch! File corrupted.")

            # Load and validate documents.json schema
            with open(docs_path, "r", encoding="utf-8") as f:
                docs_data = json.load(f)

            if len(docs_data) != 41:
                raise ValueError(f"documents.json contains {len(docs_data)} records, expected exactly 41")

            # Load and validate FAISS binary index
            faiss_index = faiss.read_index(index_path)
            if faiss_index.d != 768:
                raise ValueError(f"FAISS index dimension is {faiss_index.d}, expected 768")
            if faiss_index.ntotal != 41:
                raise ValueError(f"FAISS index contains {faiss_index.ntotal} vectors, expected 41")

            doc_store_dict = {}
            index_to_id = {}
            disease_doc_map = {}
            seen_ids = set()
            seen_canonical_diseases = set()

            for i, item in enumerate(docs_data):
                # Validate deterministic ID uniqueness and ordering consistency
                expected_id = str(i)
                doc_id = str(item.get("id", ""))
                if doc_id != expected_id:
                    raise ValueError(f"Inconsistent document ID ordering: got '{doc_id}', expected '{expected_id}' at row {i}")
                if doc_id in seen_ids:
                    raise ValueError(f"Duplicate document ID: {doc_id}")
                seen_ids.add(doc_id)

                page_content = item.get("page_content")
                if not page_content or not isinstance(page_content, str) or len(page_content.strip()) < 20:
                    raise ValueError(f"Document {doc_id} has empty or insufficient page_content")

                metadata = item.get("metadata")
                if not isinstance(metadata, dict):
                    raise ValueError(f"Document {doc_id} metadata is not a dictionary")

                dis_name = metadata.get("disease")
                canonical_dis = normalize_disease_name(dis_name)
                if not canonical_dis or canonical_dis not in CANONICAL_DISEASE_SET:
                    raise ValueError(f"Document {doc_id} contains invalid or unwhitelisted disease: '{dis_name}'")

                if canonical_dis in seen_canonical_diseases:
                    raise ValueError(f"Duplicate disease record found in documents.json: '{dis_name}' (canonical: '{canonical_dis}')")
                seen_canonical_diseases.add(canonical_dis)

                doc = Document(page_content=page_content.strip(), metadata=metadata)
                doc_store_dict[doc_id] = doc
                index_to_id[i] = doc_id
                disease_doc_map[canonical_dis] = doc

            if len(seen_canonical_diseases) != 41:
                raise ValueError(f"Index contains {len(seen_canonical_diseases)} unique diseases, expected exactly 41")

            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key,
                output_dimensionality=768,
                task_type="RETRIEVAL_QUERY"
            )

            vectorstore = FAISS(
                embedding_function=embeddings,
                index=faiss_index,
                docstore=InMemoryDocstore(doc_store_dict),
                index_to_docstore_id=index_to_id
            )

            llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=api_key,
                temperature=0.2
            )
            # RAG initialized successfully locally; live external communication not yet verified
            rag_telemetry["state"] = "initialized_unverified"
            rag_telemetry["reachability"] = "untested"
            rag_telemetry["last_error_category"] = None
            print(f"[RAG] Cloud Gemini RAG initialized successfully with model '{GEMINI_MODEL}' and 41 verified records.")

        elif LLM_PROVIDER == "ollama":
            if OllamaLLM is None or HuggingFaceEmbeddings is None:
                rag_telemetry["state"] = "uninitialized"
                rag_telemetry["last_error_category"] = "server_error"
                print("[WARN] HuggingFace/Ollama packages not installed.")
                return

            if os.path.exists(LLM_DATA_PATH):
                df_llm = pd.read_csv(LLM_DATA_PATH)
                documents = []
                disease_doc_map = {}
                seen_canonical_diseases = set()

                for _, row in df_llm.iterrows():
                    content = (
                        f"Disease: {str(row['Disease']).strip()}\n"
                        f"Description: {str(row['Description']).strip()}\n"
                        f"Diet: {str(row['Diet']).strip()}\n"
                        f"Medication: {str(row['Medication']).strip()}\n"
                        f"Workout: {str(row['workout']).strip()}\n"
                        f"Precautions: {str(row['Precautions']).strip()}"
                    )
                    canonical_dis = normalize_disease_name(row['Disease'])
                    if canonical_dis in CANONICAL_DISEASE_SET and canonical_dis not in seen_canonical_diseases:
                        seen_canonical_diseases.add(canonical_dis)
                        doc = Document(page_content=content, metadata={"disease": str(row['Disease']).strip(), "canonical_disease": canonical_dis})
                        documents.append(doc)
                        disease_doc_map[canonical_dis] = doc

                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                raw_embeddings = embeddings.embed_documents([d.page_content for d in documents])
                vectors = np.array(raw_embeddings, dtype="float32")

                # Unit-normalize document vectors for exact Cosine Similarity with Inner Product (IndexFlatIP)
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                normalized_vectors = vectors / norms

                local_faiss_index = faiss.IndexFlatIP(vectors.shape[1])
                local_faiss_index.add(normalized_vectors)

                doc_store_dict = {str(i): doc for i, doc in enumerate(documents)}
                index_to_id = {i: str(i) for i in range(len(documents))}

                vectorstore = FAISS(
                    embedding_function=embeddings,
                    index=local_faiss_index,
                    docstore=InMemoryDocstore(doc_store_dict),
                    index_to_docstore_id=index_to_id
                )
                llm = OllamaLLM(model="gemma3:1b")
                rag_telemetry["state"] = "initialized_unverified"
                rag_telemetry["reachability"] = "untested"
                rag_telemetry["last_error_category"] = None
                print("[RAG] Local Ollama RAG system initialized successfully.")
            else:
                rag_telemetry["state"] = "uninitialized"
                rag_telemetry["last_error_category"] = "server_error"
                print("[WARN] LLM Preprocessed Data.csv not found.")

    except Exception as e:
        rag_telemetry["state"] = "uninitialized"
        rag_telemetry["last_error_category"] = "server_error"
        print(f"[ERROR] RAG initialization failed: {type(e).__name__}")
        vectorstore = None
        llm = None

# Initialize RAG on startup (non-blocking)
init_rag_system()

# ==========================================
# ML HELPER FUNCTIONS (PRESERVED)
# ==========================================

def get_predicted_value(patient_symptoms):
    if gboot is None:
        raise RuntimeError("ML model is not loaded.")

    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1

    prediction = gboot.predict([input_vector])[0]
    if prediction not in diseases_list:
        raise ValueError("Unknown disease prediction.")

    return diseases_list[prediction]

def helper(dis):
    canonical_dis = normalize_disease_name(dis)

    desc_mask = description_df['Disease'].apply(normalize_disease_name) == canonical_dis
    desc = description_df[desc_mask]['Description'].values
    desc = desc[0] if len(desc) > 0 else "No description available."

    prec_mask = precautions_df['Disease'].apply(normalize_disease_name) == canonical_dis
    pre = precautions_df[prec_mask][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values
    pre = [str(p).strip() for p in pre[0] if pd.notna(p) and str(p).strip()] if len(pre) > 0 else []

    med_mask = medications_df['Disease'].apply(normalize_disease_name) == canonical_dis
    med = medications_df[med_mask]['Medication'].values
    med = med[0] if len(med) > 0 else "[]"
    try:
        med = ast.literal_eval(med)
        med = [str(m).strip() for m in med if pd.notna(m) and str(m).strip()]
    except (ValueError, SyntaxError):
        med = []

    die_mask = diets_df['Disease'].apply(normalize_disease_name) == canonical_dis
    die = diets_df[die_mask]['Diet'].values
    die = die[0] if len(die) > 0 else "[]"
    try:
        die = ast.literal_eval(die)
        die = [str(d).strip() for d in die if pd.notna(d) and str(d).strip()]
    except (ValueError, SyntaxError):
        die = []

    wrkout_mask = workout_df['disease'].apply(normalize_disease_name) == canonical_dis
    wrkout = workout_df[wrkout_mask]['workout'].values
    wrkout = [str(w).strip() for w in wrkout if pd.notna(w) and str(w).strip()] if len(wrkout) > 0 else []

    return desc, pre, med, die, wrkout

# ==========================================
# ROUTES
# ==========================================

@app.route('/')
def index():
    return render_template(
        'index.html',
        symptom_list=sorted(symptom_list)
    )

@app.route('/health', methods=['GET'])
def health():
    ml_healthy = (
        gboot is not None
        and all(
            df is not None
            for df in [
                description_df,
                precautions_df,
                medications_df,
                diets_df,
                workout_df
            ]
        )
    )

    # State 1: Unhealthy (Essential ML model or dataset failed - container must restart)
    if not ml_healthy:
        return jsonify({
            "status": "unhealthy",
            "ml_status": "unhealthy",
            "ml_model_loaded": bool(gboot is not None),
            "datasets_loaded": bool(all(df is not None for df in [description_df, precautions_df, medications_df, diets_df, workout_df])),
            "rag_status": rag_telemetry["state"],
            "provider": LLM_PROVIDER
        }), 503

    # State 2: Degraded (Core ML operational; RAG is uninitialized or degraded - DO NOT restart container)
    if rag_telemetry["state"] in ["uninitialized", "degraded"]:
        return jsonify({
            "status": "degraded",
            "ml_status": "healthy",
            "ml_model_loaded": True,
            "datasets_loaded": True,
            "rag_status": rag_telemetry["state"],
            "rag_reachability": rag_telemetry["reachability"],
            "rag_last_error_category": rag_telemetry["last_error_category"],
            "rag_last_request_time": rag_telemetry["last_request_time"],
            "provider": LLM_PROVIDER,
            "note": "Disease prediction is fully functional. Chatbot is degraded."
        }), 200

    # State 3: Partially Healthy (ML operational, RAG initialized locally but live provider call unverified)
    # Note: Returns overall "partially_healthy" with HTTP 200 so container remains alive without false restart
    if rag_telemetry["state"] == "initialized_unverified":
        return jsonify({
            "status": "partially_healthy",
            "ml_status": "healthy",
            "ml_model_loaded": True,
            "datasets_loaded": True,
            "rag_status": "initialized_unverified",
            "rag_reachability": "untested",
            "provider": LLM_PROVIDER,
            "note": "Disease prediction ready. Chatbot initialized locally but awaiting first live query."
        }), 200

    # State 4: Fully Healthy (ML operational AND live RAG communication verified via successful request)
    if rag_telemetry["state"] == "healthy":
        return jsonify({
            "status": "healthy",
            "ml_status": "healthy",
            "ml_model_loaded": True,
            "datasets_loaded": True,
            "rag_status": "healthy",
            "rag_reachability": "reachable",
            "provider": LLM_PROVIDER
        }), 200

    # State 5: Unknown / Unexpected State (Safely handled; never default to healthy)
    return jsonify({
        "status": "degraded",
        "ml_status": "healthy",
        "ml_model_loaded": True,
        "datasets_loaded": True,
        "rag_status": str(rag_telemetry.get("state", "unknown")),
        "rag_reachability": str(rag_telemetry.get("reachability", "unknown")),
        "rag_last_error_category": rag_telemetry.get("last_error_category"),
        "rag_last_request_time": rag_telemetry.get("last_request_time"),
        "provider": LLM_PROVIDER,
        "note": "Disease prediction is functional. Chatbot state is unrecognized or unknown."
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.get('symptoms')
    if not symptoms or not isinstance(symptoms, str):
        return jsonify({'error': 'No symptoms provided. Please submit a comma-separated list of symptoms.'}), 400

    user_symptoms = [symptom.strip() for symptom in symptoms.split(',') if symptom.strip()]
    if not user_symptoms:
        return jsonify({'error': 'No valid symptoms found in request.'}), 400

    recognized_symptoms = [s for s in user_symptoms if s in symptoms_dict]
    if not recognized_symptoms:
        return jsonify({'error': 'None of the provided symptoms were recognized in our database. Please select from the recognized symptom list.'}), 400

    try:
        predicted_disease = get_predicted_value(user_symptoms)
        desc, pre, med, die, wrkout = helper(predicted_disease)

        return jsonify({
            'disease': predicted_disease,
            'description': desc,
            'precautions': pre,
            'medications': med,
            'diets': die,
            'workout': wrkout
        }), 200
    except Exception as e:
        print(f"[ML_ERROR] Prediction error: {type(e).__name__}")
        return jsonify({'error': 'An internal error occurred during disease prediction. Please verify your symptom inputs.'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({'error': 'Invalid request format. Content-Type must be application/json.'}), 400

    data = request.get_json(silent=True)
    if data is None or not isinstance(data, dict):
        return jsonify({'error': 'Invalid or malformed JSON payload.'}), 400

    raw_message = data.get('message')
    if raw_message is None or not isinstance(raw_message, str):
        return jsonify({'error': 'The "message" field is required and must be a string.'}), 400

    user_input = raw_message.strip()
    if not user_input:
        return jsonify({'error': 'The "message" field cannot be empty or whitespace-only.'}), 400

    raw_disease = data.get('disease') or data.get('active_disease')
    if raw_disease is not None and not isinstance(raw_disease, str):
        return jsonify({'error': 'The "disease" field, if supplied, must be a string.'}), 400

    active_disease = raw_disease.strip() if raw_disease else ""

    if vectorstore is None or llm is None:
        return jsonify({
            'error': 'The AI medical chatbot is temporarily unavailable. Please consult a qualified medical professional.'
        }), 503

    try:
        context = ""
        # 1. Disease-Aware Whitelist Retrieval
        if active_disease:
            canonical_active = normalize_disease_name(active_disease)
            if canonical_active not in CANONICAL_DISEASE_SET:
                return jsonify({
                    'error': f'Invalid active disease: "{active_disease}". Must be a recognized condition from the medical whitelist or omitted.'
                }), 400

            matched_doc = disease_doc_map.get(canonical_active)
            if matched_doc:
                # Direct exact metadata match: guarantees retrieval cannot return an unrelated disease
                context = matched_doc.page_content
            else:
                # Active disease is valid but its exact document is missing from knowledge base
                # DO NOT run generic similarity search to substitute an unrelated disease!
                return jsonify({
                    'response': f'Verified medical documentation for "{active_disease}" is currently unavailable in the database. Please consult a qualified healthcare professional.'
                }), 200
        else:
            # 2. Generic Query: Semantic search with mandatory cosine similarity threshold (>= 0.65)
            # search_similar_documents guarantees query vector normalization and returns true Cosine Similarity in [-1.0, 1.0]
            similar_docs = search_similar_documents(user_input, k=1, score_threshold=0.65)
            if similar_docs:
                top_doc, top_score = similar_docs[0]
                context = top_doc.page_content
            else:
                return jsonify({
                    'response': 'I could not find sufficiently reliable medical recommendations matching your query in the verified disease database. Please specify a condition or consult a qualified healthcare professional.'
                }), 200

        if not context:
            return jsonify({
                'response': 'No specific medical context was found for this query in the verified database. Please consult a healthcare professional.'
            }), 200

        prompt = f"""You are an educational medical information assistant.

The following information is retrieved from an academic educational knowledge base and must not be presented as clinically validated medical advice, prescriptions, or treatment plans.

Instructions:
1. Answer directly based only on the retrieved context below.
2. Clearly identify that listed medications, precautions, or lifestyle items are educational reference entries from the database, not prescribed treatments or guaranteed cures.
3. Do not present medication names as personal prescriptions; never suggest self-medication, administration methods, or dosage changes.
4. For serious conditions (such as Malaria, Heart attack, or Diabetes), emphasize that professional clinical evaluation, diagnostic testing, and licensed medical care are essential.
5. Directly answer only the specific category asked (e.g., precautions, diet, medication, workout, or description) without adding unrelated categories.
6. Do not invent diseases, medications, or dosage instructions.
7. Conclude with a clear reminder that this information is educational and that medical decisions must be discussed with a qualified healthcare professional.

Retrieved Context:
{context}

User Question:
{user_input}

Answer:"""

        raw_response = llm.invoke(prompt)
        if hasattr(raw_response, 'content'):
            if isinstance(raw_response.content, str):
                response_text = raw_response.content
            elif isinstance(raw_response.content, list):
                text_parts = []
                for part in raw_response.content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                    elif hasattr(part, "text"):
                        text_parts.append(getattr(part, "text"))
                response_text = "\n".join(text_parts).strip() if text_parts else str(raw_response.content)
            else:
                response_text = str(raw_response.content)
        else:
            response_text = str(raw_response)

        # Update telemetry on verified live success
        rag_telemetry["state"] = "healthy"
        rag_telemetry["reachability"] = "reachable"
        rag_telemetry["last_error_category"] = None
        rag_telemetry["last_request_time"] = datetime.now(timezone.utc).isoformat()

        return jsonify({'response': response_text}), 200

    except Exception as e:
        status_code = getattr(e, 'code', None) or getattr(getattr(e, '__cause__', None), 'code', None) or getattr(e, 'status_code', None)
        exc_class = type(e).__name__
        err_msg = str(e)
        err_lower = err_msg.lower()

        # Update telemetry on failure
        rag_telemetry["state"] = "degraded"
        rag_telemetry["reachability"] = "unreachable"
        rag_telemetry["last_request_time"] = datetime.now(timezone.utc).isoformat()

        # Determine sanitized error category
        category = "server_error"
        if status_code in [401, 403] or "api_key" in err_lower or "unauthenticated" in err_lower or "permissiondenied" in err_lower:
            category = "auth_error"
        elif status_code == 429 or "resourceexhausted" in err_lower or "quota" in err_lower:
            category = "rate_limited"
        elif status_code == 404 or "not found" in err_lower:
            category = "model_error"
        elif status_code in [408, 504] or "timeout" in err_lower or "timed out" in err_lower:
            category = "timeout"
        rag_telemetry["last_error_category"] = category

        # Safe server-side logging: ONLY safe diagnostic info (class name, numeric code, category)
        # NEVER log raw exception messages, request headers, API keys, tokens, or request payloads
        safe_code = status_code if isinstance(status_code, int) else "None"
        print(f"[CHAT_ERROR] Class: {exc_class}, Code: {safe_code}, Category: {category}")

        # Return sanitized, non-technical user-facing messages
        if category == "auth_error":
            return jsonify({'error': 'The medical assistant is temporarily unavailable due to a service configuration issue. Please contact the administrator.'}), 503
        elif category == "rate_limited":
            return jsonify({'error': 'The AI assistant is temporarily experiencing high demand (free tier rate limit reached). Please wait 30 seconds and try again.'}), 429
        elif category == "model_error":
            return jsonify({'error': 'The specified AI model is currently unavailable.'}), 503
        elif category == "timeout":
            return jsonify({'error': 'The AI medical chatbot request timed out. Please try again shortly.'}), 503
        else:
            return jsonify({'error': 'The AI medical chatbot encountered a temporary provider error. Please try again shortly or consult a healthcare professional.'}), 503

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=PORT,
        debug=False
    )
