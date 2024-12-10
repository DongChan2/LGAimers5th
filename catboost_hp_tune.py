import os
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
ROOT_DIR = "data"
RANDOM_STATE = 110

def undersampling(df,normal_ratio):

    df_normal = df[df["target"] == "Normal"]
    df_abnormal = df[df["target"] == "AbNormal"]

    num_normal = len(df_normal)
    num_abnormal = len(df_abnormal)
    print(f"  Total: Normal: {num_normal}, AbNormal: {num_abnormal}")

    df_normal = df_normal.sample(n=int(num_abnormal * normal_ratio), replace=False, random_state=RANDOM_STATE)
    df_concat = pd.concat([df_normal, df_abnormal], axis=0).reset_index(drop=True)
    return df_concat
def SMOTE_OverSampling(df):
    from sklearn.preprocessing import LabelEncoder
    from imblearn.over_sampling import SMOTE

    label_encoders = {}
    X = df.drop(columns='target')
    y = df['target']
    cat_cols = [col for col in  X.select_dtypes(exclude=['object']).columns if X[col].nunique() <= 5]
    
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    smote = SMOTE(sampling_strategy='auto', random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    for col in cat_cols:
        X_resampled[col] = X_resampled[col].round().astype(int)
        X_resampled[col] = label_encoders[col].inverse_transform(X_resampled[col])        
    resampled_df['target'] = y_resampled
    return resampled_df

def AdaSyn_OverSampling(df):
    from sklearn.preprocessing import LabelEncoder
    from imblearn.over_sampling import ADASYN

    label_encoders = {}
    X = df.drop(columns='target')
    y = df['target']
    cat_cols = [col for col in  X.select_dtypes(exclude=['object']).columns if X[col].nunique() <= 5]
    
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    adasyn = ADASYN(sampling_strategy='auto', random_state=RANDOM_STATE)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
    for col in cat_cols:
        X_resampled[col] = X_resampled[col].round().astype(int)
        X_resampled[col] = label_encoders[col].inverse_transform(X_resampled[col])        
    resampled_df['target'] = y_resampled
    return resampled_df

# Load data
train_data = pd.read_csv(os.path.join(ROOT_DIR, "train.csv"))
test_data = pd.read_csv(os.path.join(ROOT_DIR, "test.csv"))
all_data = len(train_data)
missing = train_data.isnull().sum()
missing_cols = missing[missing/all_data>0.5].sort_values(ascending=True).index
train_data = train_data.drop(columns = missing_cols)
test_data = test_data.drop(columns = missing_cols)
useless = [col for col in  train_data.drop(columns=['target']).select_dtypes(exclude=['object']).columns if (train_data[col].nunique() <= 1)]
train_data = train_data.drop(columns = useless)
test_data = test_data.drop(columns = useless)
train_df = train_data.select_dtypes(exclude=['object'])
test_df = test_data.select_dtypes(exclude=['object'])
train_df['target'] = train_data['target']
test_df['target'] =test_data['target']
train_df, val_df = train_test_split(
    train_df,
    test_size=0.3,
    stratify=train_df["target"],
    random_state=RANDOM_STATE,
)


def print_stats(df: pd.DataFrame):
    num_normal = len(df[df["target"] == "Normal"])
    num_abnormal = len(df[df["target"] == "AbNormal"])

    print(f"  Total: Normal: {num_normal}, AbNormal: {num_abnormal}" + f" ratio: {num_abnormal/num_normal}")
    
# Print statistics
print(f"  \tAbnormal\tNormal")
print_stats(train_df)
print_stats(val_df)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

num_cols = [col for col in  train_data.select_dtypes(exclude=['object']).columns if train_data[col].nunique() > 5]

train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
val_df[num_cols] = scaler.transform(val_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

X_train = train_df.drop(columns =['target'])
y_train = train_df['target']
X_val = val_df.drop(columns =['target'])
y_val = val_df['target']
X_test = test_df.drop(columns =['target'])

from sklearn.metrics import classification_report, make_scorer, f1_score
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight
class_name =['AbNormal', 'Normal']
class_weights = compute_class_weight(class_weight='balanced', classes=class_name, y=y_val)
class_weights_dict = {class_name[i]: class_weights[i] for i in range(len(class_weights))}


scorer = make_scorer(f1_score,pos_label='AbNormal')
model = CatBoostClassifier(loss_function='Logloss',cat_features=X_train.select_dtypes(include=['object']).columns.to_list(), 
                           verbose=50,
                           eval_metric='F1',
                           class_weights=class_weights_dict,
                           task_type='GPU',  # GPU 사용 설정
                            devices='0')
# 하이퍼파라미터 공간 정의
param_dist = {
    'iterations': [1500],
    'learning_rate': [0.05,0.1],  # 로그 스케일 샘플링
    'depth': [6],
    'l2_leaf_reg': [1, 3, 5,],
    'bagging_temperature': [1],
    'random_strength': [1, 5, 10, 20],
    'border_count': [32, 64, 128, 254]
}

model.randomized_search(
    param_dist,
    X_train,y_train,cv=4,n_iter=5,verbose=True,refit=True,stratified=True
            
)

# 모델 학습 및 최적의 하이퍼파라미터 찾기
model.fit(X_train, y_train,eval_set=(X_val,y_val))



pred_train = model.predict(X_train)
pred_val = model.predict(X_val)
train_f1score = f1_score(y_train,pred_train,pos_label='AbNormal')
val_f1score = f1_score(y_val,pred_val,pos_label ='AbNormal')
print(train_f1score)
print(val_f1score)
print(classification_report(y_train, pred_train))
print(classification_report(y_val, pred_val))

test_pred = best_model.predict(X_test)
# 제출 데이터 읽어오기 (df_test는 전처리된 데이터가 저장됨)
df_sub = pd.read_csv("submission.csv")
df_sub["target"] = test_pred

# 제출 파일 저장
df_sub.to_csv("submission.csv", index=False) 