
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import os
import ast

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

app = Flask(__name__)

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, "dataset")
MODEL_PATH = os.path.join(BASE_PATH, "models", "drug_gboot.pkl")
LLM_DATA_PATH = os.path.join(BASE_PATH, "LLM Preprocessed Data.csv")


# ==========================================
# LOAD MACHINE LEARNING MODEL
# ==========================================

try:
    with open(MODEL_PATH, "rb") as file:
        gboot = pickle.load(file)

    print("ML model loaded successfully!")

except Exception as e:
    print(f"Error loading ML model: {e}")
    gboot = None


# ==========================================
# LOAD ML DATASETS
# ==========================================

try:
    description_df = pd.read_csv(
        os.path.join(DATASET_PATH, "description.csv")
    )

    precautions_df = pd.read_csv(
        os.path.join(DATASET_PATH, "precautions_df.csv")
    )

    medications_df = pd.read_csv(
        os.path.join(DATASET_PATH, "medications.csv")
    )

    diets_df = pd.read_csv(
        os.path.join(DATASET_PATH, "diets.csv")
    )

    workout_df = pd.read_csv(
        os.path.join(DATASET_PATH, "workout_df.csv")
    )

    print("ML datasets loaded successfully!")

except Exception as e:
    print(f"Error loading ML datasets: {e}")
    raise


# ==========================================
# SYMPTOM LIST
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
# DISEASE MAPPING
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


# ==========================================
# RAG INITIALIZATION
# ==========================================

retriever = None
llm = None

try:
    print("Initializing RAG system...")

    if os.path.exists(LLM_DATA_PATH):

        df_llm = pd.read_csv(LLM_DATA_PATH)

        required_columns = [
            "Disease", "Description", "Diet",
            "Medication", "workout", "Precautions"
        ]

        missing_columns = [
            column for column in required_columns
            if column not in df_llm.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing RAG CSV columns: {missing_columns}"
            )

        documents = []

        for _, row in df_llm.iterrows():

            content = (
                f"Disease: {row['Disease']}\n"
                f"Description: {row['Description']}\n"
                f"Diet: {row['Diet']}\n"
                f"Medication: {row['Medication']}\n"
                f"Workout: {row['workout']}\n"
                f"Precautions: {row['Precautions']}"
            )

            documents.append(
                Document(
                    page_content=content,
                    metadata={"disease": row['Disease']}
                )
            )

        if not documents:
            raise ValueError("RAG CSV contains no documents.")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(
            documents,
            embeddings
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 1}
        )

        llm = OllamaLLM(model="gemma3:1b")

        print("RAG system initialized successfully!")

    else:
        print("RAG CSV file not found. Chatbot unavailable.")

except Exception as e:
    print(f"RAG initialization error: {e}")
    retriever = None
    llm = None


# ==========================================
# ML HELPER FUNCTIONS
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

    dis_norm = " ".join(dis.strip().split())
    if dis_norm.lower() == "peptic ulcer diseae":
        dis_norm = "Peptic ulcer disease"

    desc = description_df[
        description_df['Disease'].isin([dis, dis_norm])
    ]['Description'].values

    desc = desc[0] if len(desc) > 0 else "No description available."

    pre = precautions_df[
        precautions_df['Disease'].isin([dis, dis_norm])
    ][[
        'Precaution_1',
        'Precaution_2',
        'Precaution_3',
        'Precaution_4'
    ]].values

    pre = [str(p).strip() for p in pre[0] if pd.notna(p) and str(p).strip()] if len(pre) > 0 else []

    med = medications_df[
        medications_df['Disease'].isin([dis, dis_norm])
    ]['Medication'].values

    med = med[0] if len(med) > 0 else "[]"

    try:
        med = ast.literal_eval(med)
        med = [str(m).strip() for m in med if pd.notna(m) and str(m).strip()]
    except (ValueError, SyntaxError):
        med = []

    die = diets_df[
        diets_df['Disease'].isin([dis, dis_norm])
    ]['Diet'].values

    die = die[0] if len(die) > 0 else "[]"

    try:
        die = ast.literal_eval(die)
        die = [str(d).strip() for d in die if pd.notna(d) and str(d).strip()]
    except (ValueError, SyntaxError):
        die = []

    wrkout = workout_df[
        workout_df['disease'].str.strip().str.lower().isin([dis.strip().lower(), dis_norm.lower()])
    ]['workout'].values

    wrkout = [str(w).strip() for w in wrkout if pd.notna(w) and str(w).strip()] if len(wrkout) > 0 else []

    return desc, pre, med, die, wrkout


# ==========================================
# HOME ROUTE
# ==========================================

@app.route('/')
def index():

    return render_template(
        'index.html',
        symptom_list=sorted(symptom_list)
    )


# ==========================================
# ML PREDICTION ROUTE
# ==========================================

@app.route('/predict', methods=['POST'])
def predict():

    symptoms = request.form.get('symptoms')

    if not symptoms:
        return jsonify({
            'error': 'No symptoms provided'
        }), 400

    user_symptoms = [
        symptom.strip()
        for symptom in symptoms.split(',')
        if symptom.strip()
    ]

    try:

        predicted_disease = get_predicted_value(
            user_symptoms
        )

        desc, pre, med, die, wrkout = helper(
            predicted_disease
        )

        return jsonify({
            'disease': predicted_disease,
            'description': desc,
            'precautions': pre,
            'medications': med,
            'diets': die,
            'workout': wrkout
        })

    except Exception as e:

        print(f"Prediction error: {e}")

        return jsonify({
            'error': str(e)
        }), 500


# ==========================================
# RAG CHATBOT ROUTE
# ==========================================

@app.route('/chat', methods=['POST'])
def chat():

    if not request.is_json:
        return jsonify({
            'response': 'Invalid request format.'
        }), 400

    data = request.get_json(silent=True) or {}
    user_input = data.get('message', '').strip()
    active_disease = data.get('disease', '').strip()

    if not user_input:
        return jsonify({
            'response': 'Please enter a question.'
        }), 400

    if retriever is None or llm is None:
        return jsonify({
            'response': 'The RAG chatbot is unavailable. Check the RAG CSV, embedding model and Ollama setup.'
        }), 503

    try:

        query = user_input
        if active_disease and active_disease.lower() not in user_input.lower():
            query = f"{active_disease} - {user_input}"

        docs = retriever.invoke(query)

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        if not context:
            context = "No relevant information was retrieved."

        prompt = f"""
You are a medical information assistant.

Use only the information in the retrieved context.
Directly answer only the specific question asked. If the user asks about a specific
aspect (such as precautions, diet, medication, workout, or description), provide
information strictly for that category and do not include unrelated categories.
Do not invent diseases, medications, or dosage instructions.
If the answer is not available in the context, clearly say so.
Always remind users that this information is educational
and that medical decisions should be discussed with a
qualified healthcare professional.

Retrieved Context:
{context}

User Question:
{user_input}

Answer:
"""

        response = llm.invoke(prompt)

        return jsonify({
            'response': response
        })

    except Exception as e:

        print(f"Chatbot error: {e}")

        return jsonify({
            'response': 'The chatbot encountered an error. Please check that Ollama is running and gemma3:1b is available.'
        }), 500


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == '__main__':

    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True
    )
