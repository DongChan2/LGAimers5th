import os
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import shap
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def print_stats(df: pd.DataFrame):
    num_normal = len(df[df["target"] == "Normal"])
    num_abnormal = len(df[df["target"] == "AbNormal"])

    print(f"  Total: Normal: {num_normal}, AbNormal: {num_abnormal}" + f" ratio: {num_abnormal/num_normal}")


def split_preprocess(df,test_df):
    scaler = StandardScaler()
    train_df,val_df = train_df, val_df = train_test_split(df,test_size=0.3,stratify=df["target"],random_state=RANDOM_STATE)
    num_cols = [col for col in  train_df.select_dtypes(exclude=['object']).columns if train_df[col].nunique() > 5]

    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    val_df[num_cols] = scaler.transform(val_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])
    return train_df,val_df,test_df    


#데이터 읽어오기
ROOT_DIR = "data"
RANDOM_STATE = 110
# Load data
train_data = pd.read_csv(os.path.join(ROOT_DIR, "train.csv"))
test_data = pd.read_csv(os.path.join(ROOT_DIR, "test.csv"))
# training데이터기준 missing value가 50%이상인 컬럼 제거
all_data = len(train_data) 
missing = train_data.isnull().sum()
missing_cols = missing[missing/all_data>0.5].sort_values(ascending=True).index
train_data = train_data.drop(columns = missing_cols)
test_data = test_data.drop(columns = missing_cols)
# 정보가 없는 데이터 제거
useless = [col for col in  train_data.drop(columns=['target']).select_dtypes(exclude=['object']).columns if (train_data[col].nunique() <= 1)]
train_data = train_data.drop(columns = useless)
test_data = test_data.drop(columns = useless)
#training, test 데이터 프레임으로 분리 => 'target'을 제외한 object객체 제거
train_df = train_data.select_dtypes(exclude=['object'])
test_df = test_data.select_dtypes(exclude=['object'])
train_df['target'] = train_data['target']
test_df['target'] =test_data['target']
dam_columns = [column for column in train_df.columns if 'Dam' in column] + ['target']
fill1_columns = [column for column in train_df.columns if 'Fill1' in column] + ['target']
fill2_columns = [column for column in train_df.columns if 'Fill2' in column] + ['target']
train_dam_df = train_df[dam_columns]
train_fill1_df = train_df[fill1_columns]
train_fill2_df = train_df[fill2_columns]
train_ac_df = train_df[ac_columns]
for i in ['AutoClav','Fill1','Fill2','Dam']:

    columns = [column for column in train_df.columns if i in column] + ['target']
    



train_df,val_df,test_df = split_preprocess(train_df,test_df)

