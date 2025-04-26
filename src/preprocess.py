import pandas as pd
import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm
import os
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory containing MIMIC-III csv.gz files
MIMIC_DIR = '~/UTAustin/W25/ai_in_healthcare/physionet.org/files/mimiciii/1.4/'

# Discretization thresholds based on clinical ranges
THRESHOLDS = {
	'heart_rate': {'low': 60, 'high': 100},  		# <60: low, 60-100: normal, >100: high
	'blood_pressure': {'low': 90, 'high': 120},  	# <90: low, 90-120: normal, >120: high
	'lactate': {'low': 1.0, 'high': 2.0}  			# <1.0: low, 1.0-2.0: normal, >2.0: high
}

def calc_age(row):
	intime = row["ADMITTIME"]
	dob = row["DOB"]

	if pd.isnull(intime) or pd.isnull(dob):
		return np.nan

	try:
		dt_intime = intime.to_pydatetime()
		dt_dob = dob.to_pydatetime()
	except Exception:
		return np.nan

	if dt_dob.year > dt_intime.year:
		dt_dob = dt_dob.replace(year=dt_dob.year - 100)

	try:
		age = (dt_intime - dt_dob).days / 365.25
	except OverflowError:
		return 90.0

	if age >= 300:
		return 90.0

	return age

def discretize_value(value, feature):
	if pd.isna(value):
		return 1
	thresholds = THRESHOLDS[feature]
	if value < thresholds['low']:
		return 0
	elif value <= thresholds['high']:
		return 1
	else:
		return 2

def get_sepsis_patients():
	logging.info("Loading patient data...")
	patients = pd.read_csv(os.path.join(MIMIC_DIR, 'PATIENTS.csv.gz'), 
						  usecols=['SUBJECT_ID', 'DOB'], parse_dates=['DOB'])
	admissions = pd.read_csv(os.path.join(MIMIC_DIR, 'ADMISSIONS.csv.gz'), 
							usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME'], parse_dates=['ADMITTIME'])
	icustays = pd.read_csv(os.path.join(MIMIC_DIR, 'ICUSTAYS.csv.gz'), 
						  usecols=['SUBJECT_ID', 'HADM_ID', 'LOS'])
	diagnoses = pd.read_csv(os.path.join(MIMIC_DIR, 'DIAGNOSES_ICD.csv.gz'), 
						   usecols=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'])

	merged = pd.merge(admissions, patients, on='SUBJECT_ID')
	merged['AGE'] = merged.apply(calc_age, axis=1).astype("float32")
	merged = merged[merged['AGE'] >= 18]

	sepsis_diags = diagnoses[diagnoses['ICD9_CODE'].str.startswith('9959', na=False)]
	sepsis_hadm = sepsis_diags['HADM_ID'].unique()
	sepsis_adm = merged[merged['HADM_ID'].isin(sepsis_hadm)]

	icustays = icustays[icustays['LOS'] > 1]
	sepsis_icu = pd.merge(sepsis_adm, icustays, on=['SUBJECT_ID', 'HADM_ID'])

	result = sepsis_icu[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']].drop_duplicates()
	logging.info(f"Found {len(result)} sepsis patients.")
	return result

def preprocess_vitals_and_actions(sepsis_hadm, chartevents_file, inputevents_file, temp_dir='pickled_data'):
	os.makedirs(temp_dir, exist_ok=True)
	vitals_file = os.path.join(temp_dir, 'filtered_vitals.pkl')
	actions_file = os.path.join(temp_dir, 'filtered_actions.pkl')

	if not os.path.exists(vitals_file):
		vital_chunks = []
		for chunk in tqdm(pd.read_csv(chartevents_file, 
									 usecols=['HADM_ID', 'CHARTTIME', 'ITEMID', 'VALUENUM'],
									 chunksize=1000000),
						 desc="Processing CHARTEVENTS"):
			chunk = chunk[
				(chunk['HADM_ID'].isin(sepsis_hadm)) & 
				(chunk['ITEMID'].isin([220045, 220181, 223762])) & 
				(chunk['VALUENUM'].notnull())
			]
			if not chunk.empty:
				vital_chunks.append(chunk)
		vitals = pd.concat(vital_chunks) if vital_chunks else pd.DataFrame()
		vitals.to_pickle(vitals_file)
	else:
		vitals = pd.read_pickle(vitals_file)

	if not os.path.exists(actions_file):
		action_chunks = []
		for chunk in tqdm(pd.read_csv(inputevents_file, 
									 usecols=['HADM_ID', 'STARTTIME', 'ITEMID'],
									 chunksize=100000),
						 desc="Processing INPUTEVENTS_MV"):
			chunk = chunk[
				(chunk['HADM_ID'].isin(sepsis_hadm)) & 
				(chunk['ITEMID'].isin([225168, 225158, 225152]))
			]
			if not chunk.empty:
				action_chunks.append(chunk)
		actions = pd.concat(action_chunks) if action_chunks else pd.DataFrame()
		actions.to_pickle(actions_file)
	else:
		actions = pd.read_pickle(actions_file)

	logging.info(f"Filtered vitals: {len(vitals)} rows, {vitals['HADM_ID'].nunique()} unique HADM_IDs")
	logging.info(f"Filtered actions: {len(actions)} rows, {actions['HADM_ID'].nunique()} unique HADM_IDs")
	return vitals, actions

def process_trajectories():
	patients = get_sepsis_patients()
	sepsis_hadm = patients['HADM_ID'].unique()
	transitions = defaultdict(lambda: defaultdict(int))

	vitals, actions = preprocess_vitals_and_actions(
		sepsis_hadm,
		os.path.join(MIMIC_DIR, 'CHARTEVENTS.csv.gz'),
		os.path.join(MIMIC_DIR, 'INPUTEVENTS_MV.csv.gz')
	)

	valid_patients = 0
	for hadm_id in tqdm(patients['HADM_ID'].unique(), desc="Processing patients"):
		patient_vitals = vitals[vitals['HADM_ID'] == hadm_id].copy()
		patient_actions = actions[actions['HADM_ID'] == hadm_id].copy()

		if patient_vitals.empty:
			continue

		patient_vitals['CHARTTIME'] = pd.to_datetime(patient_vitals['CHARTTIME'], errors='coerce')
		patient_vitals = patient_vitals[patient_vitals['CHARTTIME'].notna()]
		
		if patient_vitals.empty:
			continue

		item_counts = patient_vitals['ITEMID'].value_counts()
		if not (item_counts.get(220045, 0) > 0 or item_counts.get(220181, 0) > 0):
			continue

		patient_actions['STARTTIME'] = pd.to_datetime(patient_actions['STARTTIME'], errors='coerce')
		patient_actions = patient_actions[patient_actions['STARTTIME'].notna()]

		# Discretize vitals and create separate columns
		patient_vitals['heart_rate'] = patient_vitals.apply(
			lambda x: discretize_value(x['VALUENUM'] if x['ITEMID'] == 220045 else np.nan, 'heart_rate'), axis=1
		)
		patient_vitals['blood_pressure'] = patient_vitals.apply(
			lambda x: discretize_value(x['VALUENUM'] if x['ITEMID'] == 220181 else np.nan, 'blood_pressure'), axis=1
		)
		patient_vitals['lactate'] = patient_vitals.apply(
			lambda x: discretize_value(x['VALUENUM'] if x['ITEMID'] == 223762 else np.nan, 'lactate'), axis=1
		)

		# Aggregate vitals to hourly, imputing missing lactate as normal (1)
		patient_vitals['hour'] = patient_vitals['CHARTTIME'].dt.floor('h')
		vitals_agg = patient_vitals.groupby('hour').agg({
			'heart_rate': 'first',
			'blood_pressure': 'first',
			'lactate': 'first'
		}).reset_index()
		vitals_agg['lactate'] = vitals_agg['lactate'].fillna(1)
		vitals_agg = vitals_agg.ffill().bfill()

		# Map actions to hourly buckets
		if not patient_actions.empty:
			patient_actions['hour'] = patient_actions['STARTTIME'].dt.floor('h')
			patient_actions['action'] = patient_actions['ITEMID'].map({
				225168: 1,  # Antibiotics
				225158: 2,  # Fluids
				225152: 3   # Vasopressors
			})
			patient_actions = patient_actions.groupby('hour')['action'].first().reset_index()
			patient_actions['hour'] = pd.to_datetime(patient_actions['hour'])  # Ensure datetime
		else:
			# Create dummy DataFrame with datetime 'hour' column
			patient_actions = pd.DataFrame({'hour': pd.Series(dtype='datetime64[ns]'), 'action': pd.Series(dtype='int64')})

		# Merge vitals and actions
		df = pd.merge_asof(
			vitals_agg.sort_values('hour'),
			patient_actions.sort_values('hour'),
			on='hour',
			direction='nearest',
			tolerance=pd.Timedelta('1h')
		)
		df['action'] = df['action'].fillna(0)

		# Compute transitions
		for i in range(len(df) - 1):
			current_state = (
				int(df.iloc[i]['heart_rate']) if pd.notna(df.iloc[i]['heart_rate']) else 1,
				int(df.iloc[i]['blood_pressure']) if pd.notna(df.iloc[i]['blood_pressure']) else 1,
				int(df.iloc[i]['lactate']) if pd.notna(df.iloc[i]['lactate']) else 1
			)
			action = int(df.iloc[i]['action'])
			next_state = (
				int(df.iloc[i + 1]['heart_rate']) if pd.notna(df.iloc[i + 1]['heart_rate']) else 1,
				int(df.iloc[i + 1]['blood_pressure']) if pd.notna(df.iloc[i + 1]['blood_pressure']) else 1,
				int(df.iloc[i + 1]['lactate']) if pd.notna(df.iloc[i + 1]['lactate']) else 1
			)
			transitions[(current_state, action)][next_state] += 1

		valid_patients += 1
		del patient_vitals, patient_actions, vitals_agg, df

	logging.info(f"Processed {valid_patients} patients with valid data")
	probs = {}
	for key, counts in transitions.items():
		total = sum(counts.values())
		if total > 0:
			probs[str(key)] = {str(k): v / total for k, v in counts.items()}

	os.makedirs('processed_data', exist_ok=True)
	with open('processed_data/transition_probs.json', 'w') as f:
		json.dump(probs, f, indent=2)

if __name__ == "__main__":
	process_trajectories()
