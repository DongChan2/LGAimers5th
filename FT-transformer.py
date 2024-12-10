#%%
import torch 
import torch.nn as nn 
import torch.optim as optim 
import random
import warnings
warnings.filterwarnings('ignore')
import os 
import data 
from datetime import datetime
import random 
import os
from pprint import pprint
import delu 
from loss import CenterLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import classification_report,f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from rtdl_revisiting_models import FTTransformer
import torch.nn.functional as F
def feature_engineering1(df,test_df):
    from sklearn.impute import KNNImputer,SimpleImputer
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
    imputer = SimpleImputer(strategy='mean')
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
        
    cats = [col for col in df.columns if ('WorkMode' in col)]
    useless =[col for col in df.columns if df[col].nunique()==1]
    # df['WorkMode Collect Result_Dam']*=1000
    df[cats] = df[cats].astype(str)
    df = df.drop(columns=useless)
    df['target']=target
    return df 
#데이터 읽어오기
ROOT_DIR = "data"
RANDOM_STATE = 110
# Load data
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
train_val_df,test_df = feature_engineering1(train_data,test_data)
train_val_df = feature_engineering2(train_val_df)
test_df = feature_engineering2(test_df)
#%%

device = 'cuda' if torch.cuda.is_available else 'cpu'
train_df,val_df = train_test_split(train_val_df,test_size=0.3,random_state=RANDOM_STATE,stratify=train_val_df['target'])
scaler = StandardScaler()
encoder = LabelEncoder()
num_cols = train_df.drop(columns='target').select_dtypes(exclude='object').columns.to_list()
cat_cols = train_df.drop(columns='target').select_dtypes(include='object').columns.to_list()

train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
val_df[num_cols] = scaler.transform(val_df[num_cols])
test_df[num_cols] = scaler.fit_transform(test_df[num_cols])

for col in cat_cols:
    train_df[col] = encoder.fit_transform(train_df[col])
    val_df[col] = encoder.transform(val_df[col])
    test_df[col] = encoder.transform(test_df[col])

X_train = torch.Tensor(train_df.drop(columns= 'target')[num_cols + cat_cols].values)
y_train = torch.Tensor(train_df['target'].map({'AbNormal':1,'Normal':0}).values)
X_val = torch.Tensor(val_df.drop(columns='target')[num_cols + cat_cols].values)
y_val = torch.Tensor(val_df['target'].map({'AbNormal':1,'Normal':0}).values)

default_kwargs = FTTransformer.get_default_kwargs()
model = FTTransformer(
    n_cont_features=X_train.shape[1]-3
    ,
    cat_cardinalities=[8, 7, 6],
    d_out=2,
    **default_kwargs,
)
optimizer = torch.optim.AdamW(
    # Instead of model.parameters(),
    model.make_parameter_groups(),
    lr=1e-3,
    weight_decay=1e-5,
)

#Train


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        # 각 클래스에 대한 중심 벡터를 초기화 (각 중심은 feature dimension을 가짐)
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim)).to(device)

    def forward(self, features, labels):
        # 각 샘플의 라벨에 해당하는 중심을 선택하여 거리 계산
        batch_size = features.size(0)
        centers_batch = self.centers.index_select(0, labels.long())

        # L2 거리 계산
        loss = F.mse_loss(features, centers_batch)
        return loss
EPOCHS=20
features={}
def extract_fv(module,input):
    features['fc'] = input
hook = model.backbone.output.register_forward_pre_hook(extract_fv)

class_sample_count = torch.tensor([(y_train == 0).sum(), (y_train == 1).sum()])
class_weights = len(y_train) / class_sample_count.float()
print(class_weights)
criterion = nn.CrossEntropyLoss(weight=class_weights) # 라벨별로 가중치 설정
# criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1/0.06])) # 라벨별로 가중치 설정
center = CenterLoss(num_classes=2,feat_dim=192,device=device)

model.to(device)
criterion.to(device)


sample_weights = class_weights[y_train.long()]
sampler = WeightedRandomSampler(weights=sample_weights, 
                                num_samples=len(sample_weights), replacement=True)
train_dataset = TensorDataset(X_train,y_train)
train_loader = DataLoader(train_dataset, batch_size=256)
val_dataset = (X_val,y_val)
max_val_score=0
best_model=None

for epoch in range(EPOCHS):
    total_t_pred =[]
    total_t_y =[]
    total_v_pred =[]
    total_v_y =[]
    train_loss = 0
    val_loss = 0
    model.train()
    #train
    print("Training Start")
    for x,y in tqdm(train_loader):
        x=x.to(device)
        y=y.to(device).long()
        x_num = x[:,:-3]
        x_cat = x[:,-3:].long()

        pred = model(x_num,x_cat)
        cls_loss=criterion(pred,y)
        center_loss = center(features['fc'][0],y)
        # loss = cls_loss + center_loss
        loss = cls_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        total_t_pred.append(pred)
        total_t_y.append(y)
    #Validation
    with torch.no_grad():
        model.eval()
        for x,y in tqdm(delu.iter_batches(val_dataset,batch_size=512,shuffle=True)):
            x=x.to(device)
            y=y.to(device).long()
            x_num = x[:,:-3]
            x_cat = x[:,-3:].long()

            pred = model(x_num,x_cat)
            loss=criterion(pred,y)
            total_v_pred.append(pred)
            total_v_y.append(y)


    total_t_pred = torch.cat(total_t_pred,dim=0)
    total_t_y = torch.cat(total_t_y,dim=0)
    total_v_pred = torch.cat(total_v_pred,dim=0)
    total_v_y = torch.cat(total_v_y,dim=0)
    
    train_pred =(total_t_pred.argmax(dim=-1)).detach().cpu()
    val_pred = total_v_pred.argmax(dim=-1).detach().cpu()
    
    
    train_score=f1_score(train_pred,total_t_y.detach().cpu())
    valid_score=f1_score(val_pred,total_v_y.detach().cpu())
    if valid_score>max_val_score:
        max_val_score=valid_score
        best_model=model.state_dict()
    print(f"Epoch:{epoch}/{EPOCHS}\tTrain loss : {criterion(total_t_pred.squeeze(),total_t_y).item()}\tValidation loss:{criterion(total_v_pred.squeeze(),total_v_y).item()}")
    print(f"Epoch:{epoch}/{EPOCHS}\tTrain Score : {train_score}\tValidation Score:{valid_score}")
    
torch.save(best_model,f"weights/best_{max_val_score:.4f}.ckpt")
    
        

