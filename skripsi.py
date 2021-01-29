#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Aktifkan Package
import pandas as pd
import numpy as np


# In[2]:


#Input Data
dataset = pd.read_excel('data nov-jan - coba benerin pekerjaan.xlsx')
dataset.head()


# In[3]:


dataset.shape


# In[4]:


#drop duplicate data
dataset.drop_duplicates(subset=None, keep='first', inplace=True)


# In[5]:


dataset.shape


# #2. Data Cleaning

# In[6]:


data_cl = dataset.copy()


# In[7]:


#Merubah kolom gaji ke mean dari range

#1. split data gaji
data_cl[['MinGaji', 'MaxGaji']] = data_cl['Gaji'].str.split(" -",expand=True)
data_cl.tail()


# In[8]:


#mencari mean gaji (MinGaji - MaxGaji)
data_cl = data_cl.dropna()
data_cl['MinGaji'] = data_cl['MinGaji'].astype(int)
data_cl['MaxGaji'] = data_cl['MaxGaji'].astype(int)
data_cl['MeanGaji'] = (data_cl['MaxGaji'] + data_cl['MinGaji'])/2
data_cl.head()


# #3. Data Preparation

# In[9]:


data_cl.columns


# In[10]:


#ambil data yg diperlukan
data_prep = data_cl[['Nama Pekerjaan', 'Perusahaan', 'Fungsi Pekerjaan',
       'Tipe Industri', 'Pengalaman Kerja', 'MeanGaji']]
data_prep.head()


# In[11]:


dataset = data_prep.copy()


# In[12]:


q_low = dataset["MeanGaji"].quantile(0.01)
q_hi  = dataset["MeanGaji"].quantile(0.95)

dataset = dataset[(dataset["MeanGaji"] < q_hi) & (dataset["MeanGaji"] > q_low)]


# In[13]:


import seaborn as sns

ax = sns.boxplot(x="MeanGaji", data=dataset)


# In[14]:


#Label Encoder
from sklearn import preprocessing
category_col =['Nama Pekerjaan', 'Perusahaan', 'Fungsi Pekerjaan',
       'Tipe Industri', 'Pengalaman Kerja']
labelEncoder = preprocessing.LabelEncoder()


# In[15]:


#Daftar Kategorik
mapping_dict={}
for col in category_col:
 dataset[col] = labelEncoder.fit_transform(dataset[col])
 le_name_mapping = dict(zip(labelEncoder.classes_,
labelEncoder.transform(labelEncoder.classes_)))
 mapping_dict[col]=le_name_mapping


# In[16]:


mapping_dict


# In[17]:


#Variabel Independen
X = dataset.drop(["MeanGaji"], axis=1)
X.head()


# In[18]:


#Variabel Dependen
y=dataset["MeanGaji"]
y.head()


# In[19]:


#DataTesting dan Training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=1234)


# In[20]:


#Proporsi Data Testing dan Training
print(X_train.shape)
print(X_test.shape)


# In[21]:


#Analisis Random Forest
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=100,
random_state=1234)
forest_reg.fit(X_train, y_train)


# In[22]:


from sklearn.metrics import mean_squared_error

salary_predictions = forest_reg.predict(X_train)
forest_mse = mean_squared_error(y_train, salary_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[23]:


from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, X_train, y_train,
                                scoring="neg_mean_squared_error", cv=5)
forest_rmse_scores = np.sqrt(-forest_scores)

#display score
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(forest_rmse_scores)


# In[24]:


#menggunakan Grid Search untuk mencari parameter terbaik

from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30],
     'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10, 30],
     'max_features': [2, 4, 6, 8]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True, error_score= np.nan)
grid_search.fit(X_train, y_train)


# In[25]:


grid_search.best_params_


# In[26]:


grid_search.best_estimator_


# In[27]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[28]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[29]:


# Feature importances into a dataframe
features = list(X_train.columns)
feature_importances = pd.DataFrame({'feature': features,
'importance': grid_search.best_estimator_.feature_importances_})
feature_importances .plot(x ='feature', y='importance', kind =
'barh', color="lime")


# In[30]:


#menggunakan model final untuk uij coba test set
final_model = grid_search.best_estimator_

final_predictions = final_model.predict(X_test)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[31]:


final_rmse


# In[32]:


#compute 95% confidence interval
from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))


# In[33]:


#Save Model
import pickle
pickle.dump(final_model, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1,0,165,0,0]]))

