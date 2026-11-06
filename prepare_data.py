import os
import pandas as pd

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, 'dataset')
OUTPUT_PATH = os.path.join(BASE_PATH, 'LLM Preprocessed Data.csv')

def normalize_disease_name(name):
    if not isinstance(name, str):
        return ""
    norm = " ".join(name.strip().split()).lower()
    if norm == "peptic ulcer diseae":
        norm = "peptic ulcer disease"
    return norm

desc_df = pd.read_csv(os.path.join(DATASET_PATH, 'description.csv'))
meds_df = pd.read_csv(os.path.join(DATASET_PATH, 'medications.csv'))
diets_df = pd.read_csv(os.path.join(DATASET_PATH, 'diets.csv'))
prec_df = pd.read_csv(os.path.join(DATASET_PATH, 'precautions_df.csv'))
work_df = pd.read_csv(os.path.join(DATASET_PATH, 'workout_df.csv'))

# Precaution consolidation
prec_df['Precautions'] = prec_df[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].fillna('').apply(
    lambda x: ', '.join([s.strip() for s in x if str(s).strip()]).strip(', '), axis=1
)

# Workout grouping
work_df_grouped = work_df.groupby('disease')['workout'].apply(
    lambda x: ', '.join([s.strip() for s in x if str(s).strip()])
).reset_index()

# Add canonical disease column for 100% reliable 41-class joining
desc_df['canonical_disease'] = desc_df['Disease'].apply(normalize_disease_name)
meds_df['canonical_disease'] = meds_df['Disease'].apply(normalize_disease_name)
diets_df['canonical_disease'] = diets_df['Disease'].apply(normalize_disease_name)
prec_df['canonical_disease'] = prec_df['Disease'].apply(normalize_disease_name)
work_df_grouped['canonical_disease'] = work_df_grouped['disease'].apply(normalize_disease_name)

# Merge datasets using canonical disease keys
final_df = desc_df.merge(meds_df[['canonical_disease', 'Medication']], on='canonical_disease', how='left')
final_df = final_df.merge(diets_df[['canonical_disease', 'Diet']], on='canonical_disease', how='left')
final_df = final_df.merge(prec_df[['canonical_disease', 'Precautions']], on='canonical_disease', how='left')
final_df = final_df.merge(work_df_grouped[['canonical_disease', 'workout']], on='canonical_disease', how='left')

# Drop helper join column
final_df.drop(columns=['canonical_disease'], inplace=True)

# Validate zero nulls and exactly 41 rows
assert len(final_df) == 41, f"Expected 41 rows, got {len(final_df)}"
for col in ["Disease", "Description", "Diet", "Medication", "workout", "Precautions"]:
    assert final_df[col].isna().sum() == 0, f"Column '{col}' contains nulls"

final_df.to_csv(OUTPUT_PATH, index=False)
print(f"Consolidated data successfully saved to {OUTPUT_PATH} with 41 complete records.")
