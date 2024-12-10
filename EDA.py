#%%
import os
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
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



def feature_engineering1(df,test_df):
    from sklearn.impute import KNNImputer
    df = df.copy()
    test_df = test_df.copy()

    # 결측 컬럼 제거
    nan = df.isnull().sum()/len(df)<0.5
    df = df[nan[nan].index]
    test_df = test_df[nan[nan].index]
    norm_cols = [col for col in df.columns if 'NORMAL' in col]
    for i in norm_cols:
        df[i] = df[i].apply(lambda x:np.nan if x=='OK' else x).astype(float)
        test_df[i] = test_df[i].apply(lambda x:np.nan if x=='OK' else x).astype(float)
    imputer = KNNImputer()
    x_norm_cols = [col for col in norm_cols if 'X ' in col]
    df[x_norm_cols] = imputer.fit_transform(df[x_norm_cols])
    test_df[x_norm_cols]= imputer.transform(test_df[x_norm_cols])
    return df,test_df
def feature_engineering2(df):
    # 종속 변수 저장
    target = df['target']
    # 독립변수만 필터링 (문자열  컬럼 제거)
    df = df.select_dtypes(exclude='object')
    # Insp. 가 포함된 컬럼 모두 제거 
    insp_cols =[col for col in df.columns if 'Insp.' in col]
    df = df.drop(columns=insp_cols)
    #CURE 컬럼
    cure_cols = [col for col in df.columns if 'CURE' in col]
    cure_start_cols = [col for col in cure_cols if 'START POSITION' in col]
    cure_standby_cols = [col for col in cure_cols if 'STANDBY POSITION' in col]
    cure_end_cols = [col for col in cure_cols if 'END POSITION' in col]

    for i in range(0,len(cure_start_cols),3):
        suffix = " ".join(cure_start_cols[i].split(" ")[-2:])
        start_x,start_theta,start_z = cure_start_cols[i:i+3]
        standby_x,standby_theta,standby_z = cure_standby_cols[i:i+3]
        end_x,end_theta,end_z = cure_end_cols[i:i+3]
        df[start_x] = df[start_x] - df[standby_x]
        df[start_z] = df[start_z] - df[standby_z]
        df[end_x] = df[end_x] - df[standby_x]
        df[end_z] = df[end_z] - df[standby_z]
        df[start_theta] = df[start_theta] - df[standby_theta]
        df[end_theta] = df[end_theta] - df[standby_theta]
        df[f'CURE STANDBY - START POSITION {suffix}'] = np.sqrt((df[start_x])**2 + (df[start_z])**2 )
        df[f'CURE STANDBY - END POSITION {suffix}'] = np.sqrt((df[end_x])**2 + (df[end_z])**2 )
        df=df.drop(columns=cure_standby_cols[i:i+3])
    # NORMAL 좌표 컬럼 
    norm_cols = [col for col in df.columns if 'NORMAL' in col]
    st1_norm = [col for col in norm_cols if '(Stage1)' in col]
    st2_norm = [col for col in norm_cols if '(Stage2)' in col]
    st3_norm = [col for col in norm_cols if '(Stage3)' in col]
    standby = [col for col in df.columns if 'Standby' in col]
    clean = [col for col in df.columns if 'Clean' in col]
    purge = [col for col in df.columns if 'Purge' in col]
    zero = [col for col in df.columns if 'Zero' in col]
    for i in range(0,len(st1_norm),3):
        suffix = " ".join(st1_norm[i].split(" ")[-2:])
        
        
        st1_x,st1_y,st1_z = st1_norm[i:i+3]
        st2_x,st2_y,st2_z = st2_norm[i:i+3]
        st3_x,st3_y,st3_z = st3_norm[i:i+3]
        standby_x,standby_y,standby_z = standby[i:i+3]
        clean_x,clean_y,clean_z = clean[i:i+3]
        purge_x,purge_y,purge_z = purge[i:i+3]
        
        zero_x,zero_y,zero_z = zero[:3]
        
        df[st1_x] = df[st1_x] - df[standby_x]
        df[st1_y] = df[st1_y] - df[standby_y]
        df[st1_z] = df[st1_z] - df[standby_z]
        df[f'HEAD NORMAL(Stage1) {suffix}'] = np.sqrt(df[st1_x]**2 + df[st1_y]**2 + df[st1_z]**2)
        df[st2_x] = df[st2_x] - df[standby_x]
        df[st2_y] = df[st2_y] - df[standby_y]
        df[st2_z] = df[st2_z] - df[standby_z]
        df[f'HEAD NORMAL(Stage2) {suffix}'] = np.sqrt(df[st2_x]**2 + df[st2_y]**2 + df[st2_z]**2)
        df[st3_x] = df[st3_x] - df[standby_x]
        df[st3_y] = df[st3_y] - df[standby_y]
        df[st3_z] = df[st3_z] - df[standby_z]
        df[f'HEAD NORMAL(Stage3) {suffix}'] = np.sqrt(df[st3_x]**2 + df[st3_y]**2 + df[st3_z]**2)
        
        df[clean_x] = df[clean_x] - df[standby_x]
        df[clean_y] = df[clean_y] - df[standby_y]
        df[clean_z] = df[clean_z] - df[standby_z]
        df[f'Head Clean {suffix}'] = np.sqrt(df[clean_x]**2 + df[clean_y]**2 + df[clean_z]**2)
        df[purge_x] = df[purge_x] - df[standby_x]
        df[purge_y] = df[purge_y] - df[standby_y]
        df[purge_z] = df[purge_z] - df[standby_z]
        df[f'Head Purge {suffix}'] = np.sqrt(df[purge_x]**2 + df[purge_y]**2 + df[purge_z]**2)
        if i == 0:
            df[zero_x] = df[zero_x] - df[standby_x]
            df[zero_y] = df[zero_y] - df[standby_y]
            df[zero_z] = df[zero_z] - df[standby_z]
            df[f'Head Zero {suffix}'] = np.sqrt(df[zero_x]**2 + df[zero_y]**2 + df[zero_z]**2)
        df = df.drop(columns = standby[i:i+3])
        
    cats = [col for col in df.columns if ('Receip' in col) or ('WorkMode' in col)]
    useless =[col for col in df.columns if df[col].nunique()==1]
    df['WorkMode Collect Result_Dam']*=1000
    df[cats] = df[cats].astype(int).astype(str)
    df = df.drop(columns=useless)
    df['target']=target
    return df 


#데이터 읽어오기
ROOT_DIR = "data"
RANDOM_STATE = 110
# Load data
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
df,test_df = feature_engineering1(train_data,test_data)
df = feature_engineering2(df)
test_df = feature_engineering2(test_df)