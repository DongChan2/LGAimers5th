#%%
import pandas as pd 

def feature_engineering(df):
    # 종속 변수 저장
    target = df['target']
    # 결측 컬럼 제거
    df = df.dropna(axis=1)
    # 독립변수만 필터링 (문자열  열 제거)
    df = df.select_dtypes(exclude='object')
    # Insp. 가 포함된 열 모두 제거 
    insp_cols =[col for col in df.columns if 'Insp.' in col]
    df = df.drop(columns=insp_cols)
    # 범주형 변수 처리
    cure_cols = [col for col in df.columns if 'CURE' in col]
    cure_start_cols = [col for col in cure_cols if 'START POSITION' in col]
    cure_standby_cols = [col for col in cure_cols if 'STANDBY POSITION' in col]
    cure_end_cols = [col for col in cure_cols if 'END POSITION' in col]
    for standby,start,end in zip(cure_standby_cols,cure_start_cols,cure_end_cols):
        suffix = " ".join(standby.split(' ')[-3:])
        df[f'CURE POSITION START TO END {suffix}'] = ((df[standby] - df[start]) - (df[standby]-df[end])).astype(object)
        df = df.drop(columns=[standby,start,end])

    head_cols =[col for col in df.columns if ('HEAD' in col) or ('Head' in col)]
    head_norm = [col for col in head_cols if 'NORMAL' in col]
    head_standby_cols =  [col for col in head_cols if 'Standby' in col]
    head_clean_cols =  [col for col in head_cols if 'Clean' in col]
    head_purge_cols =  [col for col in head_cols if 'Purge' in col]
    for standby,clean,purge in zip(head_standby_cols,head_clean_cols,head_purge_cols):
        suffix = " ".join(standby.split(' ')[-3:])
        df[f'HEAD POSITION CLEAN TO PURGE {suffix}'] = ((df[standby] - df[clean]) - (df[standby]-df[purge])).astype(object)
        df = df.drop(columns=[standby,clean,purge])
    df[head_norm] = df[head_norm].astype(object)
    ds_cols =[col for col in df.columns if 'Distance Speed' in col]
    for i in range(0,len(ds_cols),4):
        df[ds_cols[i]] = df[ds_cols[i:i+4]].mean(axis=1)
        df = df.drop(columns = ds_cols[i+1:i+4])
    palletid_cols =[col for col in df.columns if ('PalletID' in col) or ('Receip' in col) or ('WorkMode' in col)]
    df[palletid_cols] = df[palletid_cols].astype(object)
    useless =[col for col in df.columns if df[col].nunique()==1]
    df = df.drop(columns=useless)
    df['target']=target
    return df 
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)
# %%
import pandas as pd 
df = pd.read_csv('data/train.csv')
def feature_engineering(df):
    # 종속 변수 저장
    target = df['target']
    # 결측 컬럼 제거
    df = df.dropna(axis=1)
    # 독립변수만 필터링 (문자열  열 제거)
    df = df.select_dtypes(exclude='object')
    # Insp. 가 포함된 열 모두 제거 
    insp_cols =[col for col in df.columns if 'Insp.' in col]
    df = df.drop(columns=insp_cols)
    # 범주형 변수 처리
    df['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Dam'] = np.zeros(len(df))
    df['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill1'] = np.zeros(len(df))
    df['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill2'] = np.zeros(len(df))
    x_cols = sorted([col for col in df.columns if ('X AXIS' in col) or ('Position X' in col)])
    y_cols = sorted([col for col in df.columns if  ('Y AXIS' in col) or ('Position Y' in col)])
    z_cols = sorted([col for col in df.columns if  ('Z AXIS' in col) or ('Position Z' in col)])

    for x,y,z in zip(x_cols,y_cols,z_cols):
        name = x.replace("X","")
        df[name] = "("+df[x].astype(str)+"," + df[y].astype(str)+"," + df[z].astype(str)+")"
        df=df.drop(columns=[x,y,z])
    # CURE 좌표 
    theta_cols = [col for col in df.columns if 'Θ' in col]
    cure_x_cols = [col for col in df.columns if 'POSITION X' in col]
    cure_z_cols = [col for col in df.columns if 'POSITION Z' in col]

    for x,z,theta in zip(cure_x_cols,cure_z_cols,theta_cols):
        name = x.replace("X","")
        df[name] =  "("+df[x].astype(str)+"," + df[z].astype(str)+"," + df[theta].astype(str)+")"
        df=df.drop(columns=[x,theta,z])



    ds_cols =[col for col in df.columns if 'Distance Speed' in col]
    for i in range(0,len(ds_cols),4):
        df[ds_cols[i]] = df[ds_cols[i:i+4]].astype(str).apply(lambda x: '_'.join(x), axis=1)
        df = df.drop(columns = ds_cols[i+1:i+4])
    palletid_cols =[col for col in df.columns if ('PalletID' in col) or ('Receip' in col) or ('WorkMode' in col)]
    df[palletid_cols] = df[palletid_cols].astype(object)
    useless =[col for col in df.columns if df[col].nunique()==1]
    pressure_temp_cols = [col for col in df.columns if ('Pressure' in col)or('Temp.' in col)]
    for i in range(0,8,2):
        name = " ".join(pressure_temp_cols[i].split(" ")[:2]) + "x Time_AutoClave" 
        df[name] =  df[pressure_temp_cols[i]]*df[pressure_temp_cols[i+1]]
    productions_cols =  [col for col in df.columns if ('Machine Tact' in col) or ('Qty' in col)]
    for i in range(0,len(productions_cols),2):
        name = "Production Qty per Tact time" + " ".join(productions_cols[i].split(" ")[-2:])
        df[name] = df[productions_cols[i+1]]/df[productions_cols[i]]
    df = df.drop(columns=useless)
    df['target']=target
    return df 