import pandas as pd
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, 'dataset')
OUTPUT_PATH = os.path.join(BASE_PATH, 'LLM Preprocessed Data.csv')

desc_df = pd.read_csv(os.path.join(DATASET_PATH, 'description.csv'))
meds_df = pd.read_csv(os.path.join(DATASET_PATH, 'medications.csv'))
diets_df = pd.read_csv(os.path.join(DATASET_PATH, 'diets.csv'))
prec_df = pd.read_csv(os.path.join(DATASET_PATH, 'precautions_df.csv'))
work_df = pd.read_csv(os.path.join(DATASET_PATH, 'workout_df.csv'))

# Merge dataframes
# Note: prec_df has Precaution_1, 2, 3, 4. Let's merge them into a single string.
prec_df['Precautions'] = prec_df[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].fillna('').apply(lambda x: ', '.join(x).strip(', '), axis=1)

# work_df has multiple entries per disease sometimes? Let's check or consolidate.
# In the original code it was using one row.
# Let's group workout by disease.
work_df_grouped = work_df.groupby('disease')['workout'].apply(lambda x: ', '.join(x)).reset_index()

# Merge all
final_df = desc_df.merge(meds_df, on='Disease', how='left')
final_df = final_df.merge(diets_df, on='Disease', how='left')
final_df = final_df.merge(prec_df[['Disease', 'Precautions']], on='Disease', how='left')
final_df = final_df.merge(work_df_grouped, left_on='Disease', right_on='disease', how='left')

# Drop the extra disease column from work_df_grouped
final_df.drop(columns=['disease'], inplace=True)

final_df.to_csv(OUTPUT_PATH, index=False)
print(f"Consolidated data saved to {OUTPUT_PATH}")
