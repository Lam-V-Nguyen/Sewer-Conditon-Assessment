# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 10:07:53 2022

@author: vanln
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score




folder = r'C:\Users\vanln\OneDrive\Documents\Lam_Data\Tutorial\QGIS_Condition_Assessment_Tutorial\Data'
data_path = os.path.join(folder, 'Data.csv')
data = pd.read_csv(data_path)
column_drop = ['PipeID']
data = data.drop(columns=column_drop)


lb_column = ['PipeType','NetworkTyp','Connection','PipeForm','Material',
             'Geology','SoilType','Building','LandCover','RoadClass']


X = data.iloc[:,:-2]
y_reg = data.iloc[:,-2:-1]
y_clas = data.iloc[:,-1:]



# Classification
x_train_clas, x_test_clas, y_train_clas, y_test_clas = train_test_split(X, y_clas, test_size=0.2, random_state=42, stratify=y_clas)

ordinalEncoder_clas = OrdinalEncoder()
scaler_clas = MinMaxScaler(feature_range=(0, 1))
x_train_clas[lb_column] = ordinalEncoder_clas.fit_transform(x_train_clas[lb_column])
x_test_clas[lb_column] = ordinalEncoder_clas.fit_transform(x_test_clas[lb_column])
scaler_clas.fit(x_train_clas)
X_train_clas = scaler_clas.transform(x_train_clas)
X_test_clas = scaler_clas.transform(x_test_clas)


index_clas = ['GM','ACC','F-Score','MCC']
df_clas_train = pd.DataFrame(index=index_clas)
df_clas_test = pd.DataFrame(index=index_clas)


## 1. Multi-layer Perceptron
# =============================================================================
# MLP_model = MLPClassifier(early_stopping=True, random_state=42)
# parameters = {'activation':['logistic', 'tanh', 'relu'],
#               'solver':['lbfgs', 'sgd', 'adam'],
#               'hidden_layer_sizes': tuple(np.arange(1,201,1))}
# MLP_search = GridSearchCV(MLP_model, parameters, scoring='accuracy', n_jobs=-1)
# MLP_result = MLP_search.fit(X_train_clas, y_train_clas)
# =============================================================================

MLP_model = MLPClassifier(early_stopping=True, random_state=42,
                          activation='logistic', solver='adam', hidden_layer_sizes=51)
MLP_result = MLP_model.fit(X_train_clas, y_train_clas)

# On the training dataset
rs_MLP = []
Y_pred = MLP_result.predict(X_train_clas)
rs_MLP.append(geometric_mean_score(y_true=y_train_clas, y_pred=Y_pred, average='weighted'))
rs_MLP.append(accuracy_score(y_true=y_train_clas, y_pred=Y_pred))
rs_MLP.append(f1_score(y_true=y_train_clas, y_pred=Y_pred, average='weighted'))
rs_MLP.append(matthews_corrcoef(y_true=y_train_clas, y_pred=Y_pred))
df_clas_train['MLP'] = rs_MLP

# On the validation dataset
rs_MLP = []
Y_pred = MLP_result.predict(X_test_clas)
rs_MLP.append(geometric_mean_score(y_true=y_test_clas, y_pred=Y_pred, average='weighted'))
rs_MLP.append(accuracy_score(y_true=y_test_clas, y_pred=Y_pred))
rs_MLP.append(f1_score(y_true=y_test_clas, y_pred=Y_pred, average='weighted'))
rs_MLP.append(matthews_corrcoef(y_true=y_test_clas, y_pred=Y_pred))
df_clas_test['MLP'] = rs_MLP




## 2. Support Vector Machine (SVM)
# =============================================================================
# SVM_model = SVC(random_state=42)
# parameters = {'kernel':['linear','rbf' , 'poly','sigmoid'], # 
#               'degree': np.arange(1,11,1), 'gamma': 2.0**np.arange(-10, 3, 7),
#               'C': 2.0**np.arange(-15,4,1)}
# SVM_search = GridSearchCV(SVM_model, parameters, scoring='accuracy', n_jobs=-1)
# SVM_result = SVM_search.fit(X_train_clas, y_train_clas)
# =============================================================================
SVM_model = SVC(random_state=42, kernel='poly', C=pow(2,2), degree=3, gamma=pow(2,-3), probability=True)
SVM_result = SVM_model.fit(X_train_clas, y_train_clas)

# On the training dataset
rs_SVM = []
Y_pred = SVM_result.predict(X_train_clas)
rs_SVM.append(geometric_mean_score(y_true=y_train_clas, y_pred=Y_pred, average='weighted'))
rs_SVM.append(accuracy_score(y_true=y_train_clas, y_pred=Y_pred))
rs_SVM.append(f1_score(y_true=y_train_clas, y_pred=Y_pred, average='weighted'))
rs_SVM.append(matthews_corrcoef(y_true=y_train_clas, y_pred=Y_pred))
df_clas_train['SVM'] = rs_SVM

# On the validation dataset
rs_SVM = []
Y_pred = SVM_result.predict(X_test_clas)
rs_SVM.append(geometric_mean_score(y_true=y_test_clas, y_pred=Y_pred, average='weighted'))
rs_SVM.append(accuracy_score(y_true=y_test_clas, y_pred=Y_pred))
rs_SVM.append(f1_score(y_true=y_test_clas, y_pred=Y_pred, average='weighted'))
rs_SVM.append(matthews_corrcoef(y_true=y_test_clas, y_pred=Y_pred))
df_clas_test['SVM'] = rs_SVM



## 3. Random Forest
# =============================================================================
# RF_model = RandomForestClassifier(random_state=42)
# parameters = {'max_features':np.arange(1,X_train_clas.shape[1]+1,1),
#               'criterion':['gini','entropy'],
#               'n_estimators':np.arange(10,1010,10)}
# RF_search = GridSearchCV(RF_model, parameters, scoring='accuracy', n_jobs=-1)
# RF_result = RF_search.fit(X_train_clas, y_train_clas)
# =============================================================================
RF_model = RandomForestClassifier(random_state=42,
                                  criterion="gini", n_estimators=240, max_features=1)
RF_result = RF_model.fit(X_train_clas, y_train_clas)

# On the training dataset
rs_RF = []
Y_pred = RF_result.predict(X_train_clas)
rs_RF.append(geometric_mean_score(y_true=y_train_clas, y_pred=Y_pred, average='weighted'))
rs_RF.append(accuracy_score(y_true=y_train_clas, y_pred=Y_pred))
rs_RF.append(f1_score(y_true=y_train_clas, y_pred=Y_pred, average='weighted'))
rs_RF.append(matthews_corrcoef(y_true=y_train_clas, y_pred=Y_pred))
df_clas_train['RF'] = rs_RF

# On the validation dataset
rs_RF = []
Y_pred = RF_result.predict(X_test_clas)
rs_RF.append(geometric_mean_score(y_true=y_test_clas, y_pred=Y_pred, average='weighted'))
rs_RF.append(accuracy_score(y_true=y_test_clas, y_pred=Y_pred))
rs_RF.append(f1_score(y_true=y_test_clas, y_pred=Y_pred, average='weighted'))
rs_RF.append(matthews_corrcoef(y_true=y_test_clas, y_pred=Y_pred))
df_clas_test['RF'] = rs_RF


# =============================================================================
# fpr, tpr, thresholds = roc_curve(y_test_clas, Y_pred)
# rs_RF.append(auc(fpr, tpr))
# precision, recall, _ = precision_recall_curve(y_test_clas, Y_pred)
# rs_RF.append(auc(recall, precision))
# rs_RF.append(accuracy_score(y_true=y_test_clas, y_pred=Y_pred))
# =============================================================================






# Regression
x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

ordinalEncoder_reg = OrdinalEncoder()
scaler_reg = MinMaxScaler(feature_range=(0, 1))
x_train_reg[lb_column] = ordinalEncoder_reg.fit_transform(x_train_reg[lb_column])
x_test_reg[lb_column] = ordinalEncoder_reg.fit_transform(x_test_reg[lb_column])
scaler_reg.fit(x_train_reg)
X_train_reg = scaler_reg.transform(x_train_reg)
X_test_reg = scaler_reg.transform(x_test_reg)


index_reg = ['R2','MAE','RMSE']
df_reg_train = pd.DataFrame(index=index_reg)
df_reg_test = pd.DataFrame(index=index_reg)



# 1. Multi-layer Perceptron
# MLP_model = MLPRegressor(early_stopping=True, random_state=42, max_iter=500)
# parameters = {'activation':['logistic', 'tanh', 'relu'],'solver':['lbfgs', 'sgd', 'adam']
#               ,'hidden_layer_sizes': tuple(np.arange(1,201,1))}
# MLP_search = GridSearchCV(MLP_model, parameters, scoring='r2', n_jobs=-1)
# MLP_result = MLP_search.fit(X_train_reg, y_train_reg)
# MLP_result.best_params_ = {'activation': 'relu', 'hidden_layer_sizes': 170, 'solver': 'lbfgs'}
MLP_model = MLPRegressor(early_stopping=True, random_state=42, max_iter=500,
                          activation='tanh', hidden_layer_sizes=71, solver='adam')
MLP_result = MLP_model.fit(X_train_reg, y_train_reg)

# =============================================================================
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaler.fit(y_train_reg)
# y_train = scaler.transform(y_train_reg)
# y_test = scaler.transform(y_test_reg)
# MLP_result = MLP_model.fit(X_train_reg, y_train)
# Y_pred = MLP_result.predict(X_train_reg)
# y_train_pred = scaler.inverse_transform(Y_pred.reshape(-1, 1))
# y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
# r2_score(y_true=y_train_actual, y_pred=y_train_pred)
# =============================================================================

# On the training dataset
rs_MLP = []
Y_pred = MLP_result.predict(X_train_reg)
rs_MLP.append(r2_score(y_true=y_train_reg, y_pred=Y_pred))
rs_MLP.append(mean_absolute_error(y_true=y_train_reg, y_pred=Y_pred))
rs_MLP.append(np.sqrt(mean_squared_error(y_true=y_train_reg, y_pred=Y_pred)))
df_reg_train['MLP'] = rs_MLP

# On the validation dataset
rs_MLP = []
Y_pred = MLP_result.predict(X_test_reg)
rs_MLP.append(r2_score(y_true=y_test_reg, y_pred=Y_pred))
rs_MLP.append(mean_absolute_error(y_true=y_test_reg, y_pred=Y_pred))
rs_MLP.append(np.sqrt(mean_squared_error(y_true=y_test_reg, y_pred=Y_pred)))
df_reg_test['MLP'] = rs_MLP



# 2. Support Vector Regression
# =============================================================================
# SVR_model = SVR()
# parameters = {'kernel':['rbf'], # ,'sigmoid'
#               'gamma': 2.0**np.arange(-15, 5, 1), # 'degree': np.arange(1,5,1), 
#               'C': 2.0**np.arange(-5,15,1)}
# SVR_search = GridSearchCV(SVR_model, parameters, scoring='r2', n_jobs=-1, return_train_score=True)
# SVR_result = SVR_search.fit(X_train_reg, y_train_reg)
# # 1. {'C': 128.0, 'gamma': 0.0009765625, 'kernel': 'rbf'}: -0.05767020734137627
# # 2. {'C': 4.0, 'gamma': 3.0517578125e-05, 'kernel': 'sigmoid'}: -0.11038383998705528
# # SVR_result.best_params_ =
# # SVR_result.best_score_ =
# =============================================================================
SVR_model = SVR(C=128, gamma=4, kernel='rbf')
SVR_result = SVR_model.fit(X_train_reg, y_train_reg)

# On the training dataset
rs_SVR = []
Y_pred = SVR_result.predict(X_train_reg)
rs_SVR.append(r2_score(y_true=y_train_reg, y_pred=Y_pred))
rs_SVR.append(mean_absolute_error(y_true=y_train_reg, y_pred=Y_pred))
rs_SVR.append(np.sqrt(mean_squared_error(y_true=y_train_reg, y_pred=Y_pred)))
df_reg_train['SVR'] = rs_SVR

# On the validation dataset
rs_SVR = []
Y_pred = SVR_result.predict(X_test_reg)
rs_SVR.append(r2_score(y_true=y_test_reg, y_pred=Y_pred))
rs_SVR.append(mean_absolute_error(y_true=y_test_reg, y_pred=Y_pred))
rs_SVR.append(np.sqrt(mean_squared_error(y_true=y_test_reg, y_pred=Y_pred)))
df_reg_test['SVR'] = rs_SVR



# 3. RandomForestRegressor (RFR)
# =============================================================================
# RFR_model = RandomForestRegressor(random_state=42)
# parameters = {'max_features': np.arange(1,X_train_reg.shape[1]+1,1), #[round(X_train.shape[1]/3)]
#               'n_estimators':np.arange(1,101,1)}
# RFR_search = GridSearchCV(RFR_model, parameters, scoring='r2', n_jobs=-1)
# RFR_result = RFR_search.fit(X_train_reg, y_train_reg)
# # RFR_result.best_params_ = {'max_features': 2, 'n_estimators': 96}
# # RFR_result.best_score_ = 
# =============================================================================
RFR_model = RandomForestRegressor(random_state=42, n_estimators=30,max_features=3)
RFR_result = RFR_model.fit(X_train_reg, y_train_reg)

# On the training dataset
rs_RFR = []
Y_pred = RFR_result.predict(X_train_reg)
rs_RFR.append(r2_score(y_true=y_train_reg, y_pred=Y_pred))
rs_RFR.append(mean_absolute_error(y_true=y_train_reg, y_pred=Y_pred))
rs_RFR.append(np.sqrt(mean_squared_error(y_true=y_train_reg, y_pred=Y_pred)))
df_reg_train['RFR'] = rs_RFR

# On the validation dataset
rs_RFR = []
Y_pred = RFR_result.predict(X_test_reg)
rs_RFR.append(r2_score(y_true=y_test_reg, y_pred=Y_pred))
rs_RFR.append(mean_absolute_error(y_true=y_test_reg, y_pred=Y_pred))
rs_RFR.append(np.sqrt(mean_squared_error(y_true=y_test_reg, y_pred=Y_pred)))
df_reg_test['RFR'] = rs_RFR

