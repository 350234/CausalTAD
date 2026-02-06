import os
import re
import numpy as np
import pandas as pd
import copy
import scipy.io
import inflect
import ucimlrepo 
import pickle
import string
import hashlib
from typing import List
from pandas import DataFrame
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_20newsgroups


MIXED = [ 'vifd', 'fraudecom',  'fakejob',  'seismic', 'lymphography',  '20news-0', '20news-1','20news-2','20news-3','20news-4','20news-5']
ODDS = ['breastw', 'cardio', 'ecoli', 'lymphography', 'vertebral', 'wbc', 'wine', 'yeast', 'heart', 'arrhythmia', 
		'mulcross', 'annthyroid', 'covertype', 'glass', 'http', 'ionosphere', 'letter_recognition', 'mammography',  'musk', 
		'optdigits', 'pendigits', 'pima', 'satellite', 'satimage-2', 'seismic', 'shuttle', 'smtp', 'speech', 'thyroid', 'vowels']

# Map of dataset names to their corresponding dataset IDs in the UCI ML repository
DATA_MAP ={
	# ucimlrepo
	'breastw':15,
	'cardio':193,
	'ecoli': 39,
	'lymphography': 63,
	'vertebral': 212,
	'wbc':17,
	'wine': 109,
	'yeast':110,
	# fraud detection
	'vifd': None,
	'fraudecom': None,
	'fakejob': None,
	'fakenews': None,
	# without feature names 
	'heart': 96,
	'arrhythmia': None, # download from https://odds.cs.stonybrook.edu/arrhythmia-dataset/
	'mulcross': None, # download from  https://www.openml.org/search?type=data&sort=runs&id=40897&status=active
	# adbench datasets:
	'annthyroid': 2,
	'covertype':31,
	'glass': 14,
	'http': 16,
	'ionosphere': 18,
	'letter_recognition':20,
	'mammography': 23,
	'mulcross': None,
	'musk': 25,
	'optdigits':26,
	'pendigits':28,
	'pima':29,
	'satellite':30,
	'satimage-2':31,
	'seismic': None,
	'shuttle':32,
	'smtp':34,
	'speech':36,
	'thyroid':38,
	'vowels':40,
	#20news:
	'20news-0': None,
	'20news-1': None,
	'20news-2': None,
	'20news-3': None,
	'20news-4': None,
	'20news-5': None,
}

def _hash_row(row: pd.Series) -> str:
	"""Convert a row to a unique hash string for duplicate checks."""
	# Concatenate columns as strings, then compute md5
    content = '|'.join(map(str, row.values))
    return hashlib.md5(content.encode()).hexdigest()

def split_and_save(df: pd.DataFrame,
                   train_path: str = 'train.csv',
                   test_path: str = 'test.csv',
                   label_col: str = None) -> None:
    """
		Split train/test by the label column (default: last column):
			- 'Yes' -> 1,  'No' -> 0
			- Split positives/negatives in half: first half for train, second half for test
		Save two CSVs without the index.
    """
    if df.empty:
				raise ValueError('Input DataFrame is empty.')

	# If label column is not specified, use the last column
    if label_col is None:
        label_col = df.columns[-1]

	# 2. Group by label
    pos = df[df[label_col] == "Yes"]
    neg = df[df[label_col] == "No"]

	# 3. Split each group in half
    def split_half(sub_df):
        mid = len(sub_df) // 2
        return sub_df.iloc[:mid], sub_df.iloc[mid:]

    pos_train, pos_test = split_half(pos)
    neg_train, neg_test = split_half(neg)

	# 4. Merge
    train_df = pd.concat([pos_train, neg_train], ignore_index=True)
    test_df  = pd.concat([pos_test, neg_test], ignore_index=True)

	# 5. Save
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
	print(f'Training set saved to {Path(train_path).resolve()}, {len(train_df)} rows.')
	print(f'Test set saved to {Path(test_path).resolve()}, {len(test_df)} rows.')

def check_no_overlap(csv1: str, csv2: str) -> None:
    """
	Check if two CSV files share identical rows (by content hash).
	Raise AssertionError on duplicates; otherwise print a pass message.
    """
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    set1 = {_hash_row(row) for _, row in df1.iterrows()}
    set2 = {_hash_row(row) for _, row in df2.iterrows()}

    overlap = set1 & set2
    if overlap:
		raise AssertionError(f'Found {len(overlap)} duplicate rows. Check the split logic.')
	print('✅ Check passed: no duplicate rows between train and test.')

def test_split_data(tridx,evidx):
	overlap = np.intersect1d(tridx, evidx)
	if len(overlap) > 0:
		print("Train and test overlap indices:", overlap, "Removing them from test...\n")
		evidx = evidx[~np.isin(evidx, overlap)]
		myoverlap = np.intersect1d(tridx, evidx)
		if len(myoverlap) == 0:
			print("Removed indices", overlap, "from test set\n")
		return tridx, evidx
	else:
		print("No overlapping indices between train and test\n")
		return tridx, evidx
def combine_time_columns(df):
    """
    Combine time-related columns into new date columns.
    Args:
        df: DataFrame containing time columns
    Returns:
        DataFrame with new date columns added
    """
	# Create a function to process each row and combine date info
    def create_accident_date(row):
        month_str2num = {
			'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
			}
        month = month_str2num[row['the month when the accident happened'].lower()]
        week_num = row['the week number within the month when the accident happened']
        week_str2num = {
			'monday': 1,
			'tuesday': 2,
			'wednesday': 3,
			'thursday': 4,
			'friday': 5,
			'saturday': 6,
			'sunday': 7,
			'0': 8,
			}
        day_num = week_str2num[row['the day of the week when the accident happened'].lower()]
        day = 7*(int(week_num)-1) + day_num

		# Return the combined date string
        return f"{month}-{day}"

    def create_claim_date(row):
        month_str2num = {
			'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,'0':13
			}
        week_str2num = {
			'monday': 1,
			'tuesday': 2,
			'wednesday': 3,
			'thursday': 4,
			'friday': 5,
			'saturday': 6,
			'sunday': 7,
			'0': 8,
			}
        day_num = week_str2num[row['the day of the week when the insurance claim was filed'].lower()]
        month = month_str2num[row['the month when the insurance claim was filed'].lower()]
        week_num = row['the week of the month when the insurance claim was filed']
        if day_num == 8:
            return "unknown"
        day = 7*(int(week_num)-1) + day_num
        
		# Return the combined date string
        return f"{month}-{day}"
    
	# Apply functions to create new columns
    df['the date when the accident happened'] = df.apply(create_accident_date, axis=1)
    df['the date when the insurance claim was filed'] = df.apply(create_claim_date, axis=1)
    new_cols = ['the date when the accident happened',
                'the date when the insurance claim was filed']
    for i, col in enumerate(new_cols):
        df.insert(i, col, df.pop(col))
    df.drop(['the month when the accident happened', 'the week number within the month when the accident happened',
			'the day of the week when the accident happened'], axis=1, inplace=True)
    df.drop(['the day of the week when the insurance claim was filed', 'the month when the insurance claim was filed',
			'the week of the month when the insurance claim was filed'], axis=1, inplace=True)

    return df

def verbalize_dataset(dataset_name, data_dir):
	dataset_dir = Path(data_dir) / dataset_name
	os.makedirs(dataset_dir, exist_ok = True)
	pkl_file = dataset_dir / 'data.pkl'
	if dataset_name == 'seismic':
		save_dir = Path(data_dir) / 'seismic'
		csv_path = save_dir / f'{dataset_name}.csv'
	else:
		save_dir = Path(data_dir) / 'processed'
		save_dir.mkdir(parents=True, exist_ok=True)
		csv_path = save_dir / f'{dataset_name}_processed.csv'
	
	if dataset_name == 'wine':
		dataset_id = DATA_MAP[dataset_name]
		df = ucimlrepo.fetch_ucirepo(id=dataset_id).data['original']
		# np_data = load_adbench_data(dataset_name)
		columns = [name.replace('_', ' ') for name in df.columns[:-1] ]

		X = pd.DataFrame(data = np_data['X'], columns = columns)
		y = np_data['y']
	elif dataset_name == 'breastw':
		dataset_id = DATA_MAP[dataset_name]
		df = ucimlrepo.fetch_ucirepo(id=dataset_id).data['original']
		columns = [name.replace('_', ' ') for name in df.columns[1:-1] ]
		# np_data = load_adbench_data(dataset_name)

		X = pd.DataFrame(data = np_data['X'], columns = columns)
		y = np_data['y']

	elif dataset_name == 'cardio':
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		# get columns descriptions
		var_info = uci_dataset['metadata']['additional_info']['variable_info']
		L = [ k.split(' - ') for k in var_info.split('\n') ]
		column_dict = {}
		for k, v in L:
			column_dict[k] = v.strip('\r')

		df = uci_dataset.data['original']
		df = df[df['NSP'] != 2].reset_index(drop=True)
		y = df['NSP'].map({3:1, 1:0}) # map pathologic to 1, normal to 0
		y = y.to_numpy()

		df.drop(['CLASS','NSP'], inplace = True, axis = 1)
		new_columns = [ column_dict[c] for c in df.columns]
		df.columns = new_columns
		X = df 
	elif dataset_name == 'ecoli':
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		columns = uci_dataset['variables']['description'][:8]
		X = uci_dataset.data['original'].drop(['class'], axis = 1)
		X.columns = columns
		X = X.drop(X.columns[0], axis=1)# drop id column
		y = uci_dataset.data['original']['class'].map({'omL':1,'imL':1,'imS':1, 'cp':0, 'im':0, 'pp':0, 'imU':0, 'om':0})
		y = y.to_numpy()
	elif dataset_name == 'lymphography':
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		df = uci_dataset.data['original']
		y = df['class'].map({1:1,2:0,3:0,4:1}) # 142 normal, 6 anomalies
		y = y.to_numpy()

		df.drop('class', inplace = True, axis = 1)
		df.drop('no. of nodes in', inplace = True, axis = 1)

		var_info = uci_dataset['metadata']['additional_info']['variable_info']
		df['lymphatics'] = df['lymphatics'].map({1:'normal', 2:'arched', 3:'deformed', 4:'displaced'}).astype('object')
		df['defect in node'] = df['defect in node'].map({1:'no',2:'lacunar', 3:'lac. marginal', 4:'lac. central'}).astype('object')
		df['changes in lym'] = df['changes in lym'].map({1:'bean',2:'oval', 3:'round'}).astype('object')
		df['changes in node'] = df['changes in node'].map({1:'no',2:'lacunar', 3:'lac. marginal', 4:'lac. central'}).astype('object')
		df['changes in stru'] = df['changes in stru'].map({1:'no',2:'grainy', 3:'drop-like', 4:'coarse', 5:'diluted', 6: 'reticular', 7:'stripped', 8:'faint'}).astype('object')
		df['special forms'] = df['special forms'].map({1:'no',2:'chalices', 3:'vesicles'}).astype('object')
		
		for k in ['block of affere', 'bl. of lymph. c', 'bl. of lymph. s', 'by pass', 'extravasates', 'regeneration of', 'early uptake in', 'dislocation of', 'exclusion of no']:
			df[k] = df[k].map({1:'no',2:'yes'}).astype('object')
		
		X = df
	
	elif dataset_name == 'vertebral':
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		df = uci_dataset.data['original']
		
		df_anomaly = df[df['class'] == 'Normal'] # 100 normal data is treated as abnormal
		df_normal = df[df['class'] != 'Normal'] # 210
		df_anomaly = df_anomaly.sample(n=30, random_state = 42)
		df = pd.concat([df_anomaly, df_normal], axis = 0, ignore_index=True)
	
		y = df['class'].map({'Spondylolisthesis':0, 'Normal':1, 'Hernia': 0}) # 210 normal, 30 anomalies
		y = y.to_numpy()
		df.drop('class', inplace = True, axis = 1)
		df.columns = [name.replace('_', ' ') for name in df.columns ]
		X = df
	elif dataset_name == 'covertype':
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		df = uci_dataset.data['original']
		
		for column in df.columns:
			if 'Soil' in column or 'Wilderness' in column:
				df.drop(column, axis =1 , inplace = True)
		df_normal = df[df['Cover_Type'] == 2]
		df_anomaly = df[df['Cover_Type'] == 4]
		df = pd.concat([df_anomaly, df_normal], axis = 0, ignore_index=True)
		
		y = df['Cover_Type'].map({2:0, 4:1})
		y = y.to_numpy()
		df.drop('Cover_Type', inplace = True, axis = 1)
		
		df.columns = [name.replace('_', ' ') for name in df.columns ]
		X = df
	elif dataset_name == 'heart':
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		df = uci_dataset.data['original']
		
		y = df['diagnosis'] 
		y = y.to_numpy()
		
		X = uci_dataset.data['original'].drop(['diagnosis'], axis = 1)

	elif dataset_name == 'wbc':
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		df = uci_dataset.data['original']
		# downsample anomaly to 21 samples
		df_anomaly = df[df['Diagnosis'] == 'M']
		df_normal = df[df['Diagnosis'] == 'B']
		df_anomaly = df_anomaly.sample(n=21, random_state = 42)
		df = pd.concat([df_anomaly, df_normal], axis = 0, ignore_index=True)
	
		y = df['Diagnosis'].map({'M':1, 'B':0}) # 142 normal, 6 anomalies
		y = y.to_numpy()
		df.drop('Diagnosis', inplace = True, axis = 1)
		df.drop('ID', inplace = True, axis = 1)

		X = df
	elif dataset_name == 'yeast':
		# the split is different than the one in the ADbench
		dataset_id = DATA_MAP[dataset_name]
		uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
		df = uci_dataset.data['original']
		columns = [ s.rstrip('.') for s in uci_dataset['variables']['description'][1:9] ]
	
		y = df['localization_site'].map({'CYT':0, 'NUC':0, 'MIT':0,'ME3':0, 'ME2':1, 'ME1':1, 'EXC':0, 'VAC':0, 'POX':0, 'ERL':0}) 
		y = y.to_numpy()
		df.drop('localization_site', inplace = True, axis = 1)
		df.drop('Sequence_Name', inplace = True, axis = 1)
		df.columns = columns

		X = df

	elif dataset_name == 'mental':
		idf = pd.read_csv(Path(data_dir) / 'mental'/ 'Mental-Health-Twitter.csv').astype(str)

		df = downsample_95_5(idf, col='label', seed=42)
		df = df.iloc[:, 1:]  

		# Columns to drop
		deleted_cols = [
			"post_id",
			"user_id",
			"followers",
			"favourites"
			]
		
		# Drop columns
		for col in deleted_cols:
			df.drop([col], inplace=True, axis=1)

		column_replacement = {
			'post_created': 'the time when the post was created',
			'post_text': 'the content of the post',
			'friends': 'the number of friends',
			'statuses': 'the number of statuses',
			'retweets': 'the number of retweets',
			'label': 'class'
			}

		# Rename the columns
		df.rename(columns=column_replacement, inplace=True)

		columns = [ split_on_uppercase(c) for c in df.columns]
		df.columns = columns
		df['class'] = df['class'].replace({
			'1': 'No', 
			'0': 'Yes'
			})
		        # In-place slice, drop column 0


		df.to_csv(Path(data_dir) / 'mental'/ 'mental_health.csv', index=False)
		print(f"verbalized data saved to: {Path(data_dir) / 'mental'/ 'mental_health.csv'}")


	elif dataset_name == 'vifd':
		df = pd.read_csv(Path(data_dir) / 'vifd'/ 'carclaims_raw.csv').astype(str)
		
		replace_map = {
			"0": "Unknown"
		}
		df["Age"] = df["Age"].replace(replace_map)

		replaced_cols = [
			"NumberOfCars",
			"AgeOfVehicle"
			]
		# Columns to drop
		deleted_cols = [
			"PolicyNumber",
			'RepNumber',
			'Year'
			]
		
		# Apply replacements
		for col in replaced_cols:	
			df[col] = df[col].apply(replace_col)
		
		# Drop columns
		for col in deleted_cols:
			df.drop([col], inplace=True, axis=1)

		column_replacement = {
			'Month': 'the month when the accident happened',
			'WeekOfMonth': 'the week number within the month when the accident happened',
			'DayOfWeek': 'the day of the week when the accident happened',
			'Make': 'the brand of the vehicle',
			'AccidentArea': 'the type of area where the accident happened',
			'DayOfWeekClaimed': 'the day of the week when the insurance claim was filed',
			'MonthClaimed': 'the month when the insurance claim was filed',
			'WeekOfMonthClaimed': 'the week of the month when the insurance claim was filed',
			'Sex': 'the sex of the driver',
			'MaritalStatus': 'the marital status of the driver',
			'Age': 'the age of the driver',
			'Fault': 'the party at fault for the accident',
			'PolicyType': 'the type of policy',
			'VehicleCategory': 'the category of the vehicle',
			'VehiclePrice': 'the price of the vehicle',
			'Deductible': 'the deductible amount',
			'DriverRating': 'the rating of the driver',
			'Days:Policy-Accident': 'the days between the policy start and the accident',
			'Days:Policy-Claim': 'the days between the policy start and the claim',
			'PastNumberOfClaims': 'the number of previous claims made by the policy holder',
			'AgeOfVehicle': 'the age of the vehicle',
			'AgeOfPolicyHolder': 'the age of the policy holder',
			'PoliceReportFiled': 'whether the police report was filed',
			'WitnessPresent': 'whether a witness was present',
			'AgentType': 'the type of agent',
			'NumberOfSuppliments': 'the number of suppliments',
			'AddressChange-Claim': 'whether the address has changed since the claim or the time of such change',
			'NumberOfCars': 'the number of vehicles under the policy',
			'BasePolicy': 'the base type of the policy',
			'FraudFound': 'fraud found',
			}

		# Rename the columns
		df.rename(columns=column_replacement, inplace=True)

		df = combine_time_columns(df)

		split_and_save(df, str(Path(data_dir) / 'vifd'/ 'carclaims_train.csv'), str(Path(data_dir) / 'vifd'/ 'carclaims_test.csv'),"fraud found")
		check_no_overlap(str(Path(data_dir) / 'vifd'/ 'carclaims_train.csv'), str(Path(data_dir) / 'vifd'/ 'carclaims_test.csv'))

		# df.to_csv(Path(data_dir) / 'vifd'/ 'carclaims.csv', index=False)
		# print(f"verbalized data saved to: {Path(data_dir) / 'vifd'/ 'carclaims.csv'}")
	
	elif dataset_name == 'arrhythmia':
		data_path = Path(data_dir) / 'arrhythmia' / 'arrhythmia.mat'
		if not os.path.exists(data_path):
			print("Please download the dataset from https://odds.cs.stonybrook.edu/arrhythmia-dataset/ and put it to data/arrhythmia")
			raise ValueError('arrhythmia.mat is not found in {}'.format(data_path))
		data = scipy.io.loadmat(data_path)
		X_np, y = data['X'], data['y']
		X = convert_np_to_df(X_np)

	elif dataset_name == 'mulcross':
		data_path = Path(data_dir) / 'mulcross' / 'mulcross.arff'
		if not os.path.exists(data_path):
			print("Please download the dataset from <MULCROSS_DATASET_URL> and put it to data/mulcross")
			raise ValueError('mulcross.arff is not found in {}'.format(data_path))	
		data, meta = scipy.io.arff.loadarff(data_path)
		X = [ [x[i] for i in range(4)] for x in data]
		X_np = np.array(X)
		y = [ x[4] for x in data]
		y = [ 0 if y == b'Normal' else 1 for y in y]
		y = np.array(y)
		X = convert_np_to_df(X_np)
	elif dataset_name == 'seismic':
		data_path = Path(data_dir) / 'seismic' / 'seismic-bumps.arff'
		if not os.path.exists(data_path):
			print("Please download the dataset from <SEISMIC_DATASET_URL> and put it to data/seismic")
			raise ValueError('mulcross.arff is not found in {}'.format(data_path))	
		data, meta = scipy.io.arff.loadarff(data_path)
		df = pd.DataFrame(data)

		column_replacement = {
			'seismic': 'result of shift seismic hazard assessment in the mine working obtained by the seismic method',
			'seismoacoustic': 'result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method',
   			'shift': 'information about type of a shift',
			'genergy': 'seismic energy recorded within previous shift by the most active geophone (GMax) out of geophones monitoring the longwall',
			'gpuls': 'a number of pulses recorded within previous shift by GMax',
			'gdenergy': 'a deviation of energy recorded within previous shift by GMax from average energy recorded during eight previous shifts',
			'gdpuls': 'a deviation of a number of pulses recorded within previous shift by GMax from average number of pulses recorded during eight previous shifts',
			'ghazard': 'result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method based on registration coming from GMax only',
			'nbumps': 'the number of seismic bumps recorded within previous shift',
			'nbumps2': 'the number of seismic bumps (in energy range [10^2,10^3)) registered within previous shift',
			'nbumps3': 'the number of seismic bumps (in energy range [10^3,10^4)) registered within previous shift',
			'nbumps4': 'the number of seismic bumps (in energy range [10^4,10^5)) registered within previous shift',
			'nbumps5': 'the number of seismic bumps (in energy range [10^5,10^6)) registered within the last shift',
			'nbumps6': 'the number of seismic bumps (in energy range [10^6,10^7)) registered within previous shift',
			'nbumps7': 'the number of seismic bumps (in energy range [10^7,10^8)) registered within previous shift',
			'nbumps89': 'the number of seismic bumps (in energy range [10^8,10^10)) registered within previous shift',
			'energy': 'total energy of seismic bumps registered within previous shift',
			'maxenergy': 'the maximum energy of the seismic bumps registered within previous shift',
		}
		# Rename the columns
		df.rename(columns=column_replacement, inplace=True)

		# Replace categorical values in the columns
		df['result of shift seismic hazard assessment in the mine working obtained by the seismic method'] = df['result of shift seismic hazard assessment in the mine working obtained by the seismic method'].replace({b'a': 'lack of hazard', b'b': 'low hazard', b'c': 'high hazard', b'd': 'danger state'})
		df['result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method'] = df['result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method'].replace({b'a': 'lack of hazard', b'b': 'low hazard', b'c': 'high hazard', b'd': 'danger state'})
		df['result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method based on registration coming from GMax only'] = df['result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method based on registration coming from GMax only'].str.decode('utf-8').replace({
			'a': 'lack of hazard',
			'b': 'low hazard',
			'c': 'high hazard',
			'd': 'danger state'
			})
		df['information about type of a shift'] = df['information about type of a shift'].str.decode('utf-8').replace({
			'W': 'coal-getting', 
			'N': 'preparation shift'
			})
		df['class'] = df['class'].str.decode('utf-8').replace({
			'0': 'No', 
			'1': 'Yes'
			})
		df.to_csv(csv_path, index=False, encoding='utf-8')
		print(f"verbalized data saved to: {csv_path}")

	elif dataset_name == 'fraudecom':
		# data downloaded from <FRAUD_ECOM_DATASET_URL>
		# add one index for device id that only appears once
		# preprocessing code adapted from <FRAUD_ECOM_PREPROCESS_REF>
		# remove device id
		import calendar

		data_path = Path(data_dir) / 'fraudecom'
		dataset = pd.read_csv(data_path / "Fraud_Data.csv")              # Users information
		IP_table = pd.read_csv(data_path / "IpAddress_to_Country.csv")   # Country from IP in 

		IP_table.upper_bound_ip_address.astype("float")
		IP_table.lower_bound_ip_address.astype("float")
		dataset.ip_address.astype("float")

		# function that takes an IP address as argument and returns country associated based on IP_table
		def IP_to_country(ip) :
			try :
				return IP_table.country[(IP_table.lower_bound_ip_address < ip)                            
										& 
										(IP_table.upper_bound_ip_address > ip)].iloc[0]
			except IndexError :
				return "Unknown"     
			
		# To affect a country to each IP :
		dataset["IP_country"] = dataset.ip_address.apply(IP_to_country)
		# We convert signup_time and purchase_time en datetime
		#dataset = pd.read_csv(data_path / "Fraud_data_with_country.csv")
		dataset.signup_time = pd.to_datetime(dataset.signup_time, format = '%Y-%m-%d %H:%M:%S')
		dataset.purchase_time = pd.to_datetime(dataset.purchase_time, format = '%Y-%m-%d %H:%M:%S')

		# --- 2 ---
		# Column month
		dataset["month_purchase"] = dataset.purchase_time.apply(lambda x: calendar.month_name[x.month])

		# --- 3 ---
		# Column week
		dataset["weekday_purchase"] = dataset.purchase_time.apply(lambda x: calendar.day_name[x.weekday()])
		# --- 4 ---
		# map the device id that appears only once to 0
		device_duplicates = pd.DataFrame(dataset.groupby(by = "device_id").device_id.count())  # at this moment, index column name and first column name both are equal to "device_id"
		device_duplicates.rename(columns={"device_id": "freq_device"}, inplace=True)           # hence we need to replace the "device_id" column name
		device_duplicates.reset_index(level=0, inplace= True)                                  # and then we turn device_id from index to column

		dataset = dataset.merge(device_duplicates, on= "device_id")
		indices = dataset[dataset.freq_device == 1].index
		dataset.loc[indices, "device_id"]= "0"

		le = LabelEncoder()
		dataset['device_id'] = le.fit_transform(dataset['device_id']).astype('object')
		for column in ['user_id', 'signup_time', 'purchase_time', 'ip_address', 'freq_device']:
			dataset.drop(column, axis=1, inplace = True)

		dataset.columns = [name.replace('_', ' ') for name in dataset.columns ]
		y = dataset['class'].to_numpy()
		X = dataset.drop("class", axis = 1)
		X = dataset.drop("device id", axis = 1)
		csv_path = Path(data_dir) / 'fraudecom' / 'fraudecom.csv'
		X.to_csv(csv_path, index=False, encoding='utf-8')
		print(f"verbalized data saved to: {csv_path}")

	elif dataset_name == 'fakejob':
		# data download link: <FAKEJOB_DATASET_URL>
		df = pd.read_csv( Path(data_dir) / 'fakejob'/ 'fake_job_postings.csv')

		# deal with Nan values
		df['location'].fillna('Unknown', inplace=True)
		df['department'].fillna('Unknown', inplace=True)
		df['salary_range'].fillna('Not Specified', inplace=True)
		df['employment_type'].fillna('Not Specified', inplace=True)
		df['required_experience'].fillna('Not Specified', inplace=True)
		df['required_education'].fillna('Not Specified', inplace=True)
		df['industry'].fillna('Not Specified', inplace=True)
		df['function'].fillna('Not Specified', inplace=True)
		df.drop('job_id', inplace=True, axis=1)

		text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
		df[text_columns] = df[text_columns].fillna('NaN')
		
		y = df['fraudulent'].to_numpy()
		X = df.drop('fraudulent', axis=1)
		X.columns = [name.replace('_', ' ') for name in X.columns ]
	
	
	elif dataset_name.startswith('20news-'):
		def data_generator(subsample=None, target_label=None):
			dataset = fetch_20newsgroups(subset='train')
			groups = [['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'],
				['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'],
				['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
				['misc.forsale'],
				['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'],
				['talk.religion.misc', 'alt.atheism', 'soc.religion.christian']]

			def flatten(l):
				return [item for sublist in l for item in sublist]
			label_list = dataset['target_names']
			label = []
			for _ in dataset['target']:
				_ = label_list[_]
				if _ not in flatten(groups):
					raise NotImplementedError
				
				for i, g in enumerate(groups):
					if _ in g:
						label.append(i)
						break
			label = np.array(label)
			print("Number of labels", len(label))
			idx_n = np.where(label==target_label)[0]
			idx_a = np.where(label!=target_label)[0]
			label[idx_n] = 0
			label[idx_a] = 1
			# subsample
			if int(subsample * 0.95) > sum(label == 0):
				pts_n = sum(label == 0)
				pts_a = int(0.05 * pts_n / 0.95)
			else:
				pts_n = int(subsample * 0.95)
				pts_a = int(subsample * 0.05)

			idx_n = np.random.choice(idx_n, pts_n, replace=False)
			idx_a = np.random.choice(idx_a, pts_a, replace=False)
			idx = np.append(idx_n, idx_a)
			np.random.shuffle(idx)

			text = [dataset['data'][i] for i in idx]
			label = label[idx]
			del dataset
	
			text = [_.strip().replace('<br />', '') for _ in text]

			print(f'number of normal samples: {sum(label==0)}, number of anomalies: {sum(label==1)}')

			return text, label
		target_label = int(dataset_name.split('-')[1])
		text, label = data_generator(subsample=10000, target_label=target_label)
		label_mapped = np.where(label == 0, 'No', 'Yes')
		X = pd.DataFrame({'text': text, 'class': label})
		X['text'] = X['text'].astype('string')
		X['class'] = X['class'].astype('string')
		y = label
		csv_path = Path(data_dir) / f'20news-{target_label}'
		if not csv_path.exists():
			csv_path.mkdir(parents=True, exist_ok=True)
			print(f"Created directory: {csv_path}")
		csv_path = csv_path / f'20news-{target_label}.xlsx'
		X.to_excel(csv_path, index=False)
		print(f"verbalized data saved to: {csv_path}")

		
	elif dataset_name in DATA_MAP.keys():
		# datasets from ADBench
		# dataset_root = Path(adbench.__file__).parent.absolute() / "datasets/Classical"
		n = DATA_MAP[dataset_name]
		for npz_file in os.listdir(dataset_root):
			if npz_file.startswith(str(n) + '_'):
				print(dataset_name, npz_file)
				data = np.load(dataset_root / npz_file, allow_pickle=False)
				break
		else: 
			ValueError('{} is not found.'.format(dataset_name))
		X_np, y = data['X'], data['y']
		X = convert_np_to_df(X_np)
	else:
		raise ValueError('Invalid dataset name {}'.format(dataset_name))
	
	return


def number_to_words(match):
	p = inflect.engine()
	num_str = match.group()
	try:
		# Support non-negative integers
		return p.number_to_words(int(num_str))
	except ValueError:
		return num_str  # Keep as-is if conversion fails

def replace_numbers_with_words(text: str) -> str:
	p = inflect.engine()
	# Regex match non-negative integers
	return re.sub(r'\d+', number_to_words, str(text))

def replace_col(x:str):
	if x == '1 vehicle':
		return '1'
	elif x == '2 vehicles':
		return '2'
	elif x == 'more than 7':
		return 'more than 7 years'
	else:
		return x

def downsample_95_5(df: DataFrame, col: str = 'class', seed: int = 42) -> DataFrame:
    """
	Keep all rows where col==1;
	downsample col==0 to make the ratio 1:0 ≈ 95:5;
	return with original columns and reset index from 0.
    """
    if col not in df.columns:
		raise KeyError(f'Column "{col}" does not exist')

    ones = df[df[col] == '1']
    zeros = df[df[col] == '0']

    if ones.empty:
		raise ValueError('No samples with col==1; cannot construct a 95:5 split')

	# Compute required number of 0s
    n_zeros = int(len(ones) * 5 / 95)
	n_zeros = max(0, min(n_zeros, len(zeros)))          # Boundary guard
    zeros_sample = zeros.sample(n=n_zeros, random_state=seed) if n_zeros else zeros.iloc[:0]

	# Concatenate + shuffle + reset index
    out = pd.concat([ones, zeros_sample], ignore_index=True) \
            .sample(frac=1, random_state=seed) \
            .reset_index(drop=True)
    return out

# def load_adbench_data(dataset):
# 	dataset_root = Path(adbench.__file__).parent.absolute() / "datasets/Classical"
# 	if not os.path.exists(dataset_root):
# 		from adbench.myutils import Utils
# 		Utils().download_datasets(repo='jihulab')
	
# 	if dataset == 'cardio':
# 		return np.load(dataset_root / '6_cardio.npz', allow_pickle=False)

# 	for npz_file in os.listdir(dataset_root):
# 		if dataset in npz_file.lower():
# 			return np.load(dataset_root / npz_file, allow_pickle=False)
# 	else: 
# 		ValueError('{} is not found.'.format(dataset))


def split_on_uppercase(s):
		return ''.join(' ' + i if i.isupper() else i for i in s).lower().strip()
def convert_np_to_df(X_np):
	n_train, n_cols = X_np.shape
	# Add missing column names
	L = list(string.ascii_uppercase) + [letter1+letter2 for letter1 in string.ascii_uppercase for letter2 in string.ascii_uppercase]
	columns = [ L[i] for i in range(n_cols) ]
	df = pd.DataFrame(data = X_np, columns = columns)
	return df


def normalize(X, method, n_buckets):
	# method: ['quantile', 'equal_width', 'language', 'none', 'standard'] 
	# n_buckets: 0-100
	X = copy.deepcopy(X)
	def ordinal(n):
		if np.isnan(n):
			return 'NaN'
		n = int(n)
		if 10 <= n % 100 <= 20:
			suffix = 'th'
		else:
			suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
		return 'the ' + str(n) + suffix + ' percentile'
	
	word_list = ['Minimal', 'Slight', 'Moderate', 'Noticeable', 'Considerable', 'Significant', 'Substantial', 'Major', 'Extensive', 'Maximum']
	def get_word(n):
		n = int(n)
		if n == 10:
			return word_list[-1]
		return word_list[n]
	
	if method == 'quantile':
		for column in X.columns:
			if X[column].dtype in ['float64', 'int64', 'uint8', 'int16'] and  X[column].nunique() > 1:
				ranks = X[column].rank(method='min')
				X[column] = ranks / len(X[column]) * 100
				X[column] = X[column].apply(ordinal)
					
	elif method == 'equal_width':
		for column in X.columns:
			if X[column].dtype in ['float64', 'int64', 'uint8', 'int16']:
				if X[column].nunique() > 1:
					X[column] = X[column].astype('float64')
					X[column] = (X[column] - X[column].min()) / (X[column].max() - X[column].min()) * n_buckets 
				
				if 10 % n_buckets == 0:
					X[column] = X[column].round(0) / 10
					X[column] = X[column].round(1) 
				else: 
					X[column] = X[column].round(0) / 100
					X[column] = X[column].round(2)
	elif method == 'standard':
		for column in X.columns:
			if X[column].dtype in ['float64', 'int64', 'uint8', 'int16']:
				scaler = StandardScaler()
				scaler.fit(X[column].values.reshape(-1,1))
				X[column] = scaler.transform(X[column].values.reshape(-1,1))
				X[column] = X[column].round(1) 

	elif method == 'language':
		for column in X.columns:
			if X[column].dtype in ['float64', 'int64', 'uint8', 'int16'] and X[column].nunique() > 1:
				X[column] = X[column].astype('float64')
				X[column] = (X[column] - X[column].min()) / (X[column].max() - X[column].min()) * 10
				X[column] = X[column].apply(get_word)
	else:
		raise ValueError('Invalid method. Choose either percentile, language or decimal')
	return X

def main():
	# Example usage
	data_dir = 'data'
	for i in range(1):
		dataset_name = 'vifd'
		verbalize_dataset(dataset_name, data_dir)
	# dataset_name = 'vifd'  # Change to the dataset you want to verbalize
	# verbalize_dataset(dataset_name, data_dir)

if __name__ == '__main__':
	main()



