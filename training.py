import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('properties_for_ml.csv')
df = dataset.copy()
df.head()

df=df.iloc[:,2:len(df)]
df["has_lift"]=df["has_lift"].astype(int)
df["has_parking_space"]=df["has_parking_space"].astype(int)
df["has_swimming_pool"]=df["has_swimming_pool"].astype(int)
df["price"]=df["price"].astype(int)
df["floor"].fillna(df.floor.mean(),inplace=True)
df.floor=df.floor.astype(int)
df.head()

df[['published_at','unpublished_at']] = df[['published_at','unpublished_at']].apply(pd.to_datetime) 
df['total_published_time'] = (df['unpublished_at'] - df['published_at']).dt.days
df['total_published_time']

df['building_age'] = (2022 - df['construction_year'])

del df['published_at']
del df['unpublished_at']
del df['construction_year']
df.head()

df.describe().T
df.corr()

# aykırı yöntem ile silme işlemi
df=df.dropna() #veri setinde boş değerleri siler.
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
np.sort(df_scores)[0:20]

esik_deger = np.sort(df_scores)[13]
aykiri_tf = df_scores > esik_deger
aykiri_tf

aykiri_df = df[df_scores<esik_deger] #aykırı olan değerlere eriştim.
#print(aykiri_df)

new_df = df[df_scores>esik_deger] #aykırı olmayan değerlere eriştim.
#print(new_df)

del new_df['days_on_market'] #dummy değişken olan days_on_market kolonunu sildim.
new_df

correlations = new_df.corr()
#print(correlations["construction_year"]) #tek tek korelasyon incelemesi


from sklearn.model_selection import train_test_split
total_published_time = new_df.iloc[:,7:8]
sol=new_df.iloc[:,0:7]
sag=new_df.iloc[:,8:]
all_data = pd.concat([sol,sag],axis=1)
x_train,x_test,y_train,y_test=train_test_split(all_data,total_published_time, test_size=0.25,random_state=0)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train,y_train) #y_train verinin %75'lik kısmını oluşturur.

#backward elimination

import statsmodels.api as sm
X= np.append(arr=np.ones((9423,1)).astype(int), values=all_data,axis=1)

X_all = all_data.iloc[:,[0,1,2,3,4,5,6,7]].values
X_all=np.array(X_all,dtype=float)
model=sm.OLS(total_published_time,X_all).fit()
print(model.summary())

X_all1 = all_data.iloc[:,[0,1,3,4,5,6,7]].values
X_all1=np.array(X_all1,dtype=float)
model2=sm.OLS(total_published_time,X_all1).fit()
print(model2.summary())

X_all2 = all_data.iloc[:,[1,3,4,5,6,7]].values
X_all2=np.array(X_all2,dtype=float)
model3=sm.OLS(total_published_time,X_all2).fit()
print(model3.summary())

X_all3 = all_data.iloc[:,[1,3,4,6,7]].values
X_all3=np.array(X_all3,dtype=float)
model4=sm.OLS(total_published_time,X_all3).fit()
print(model4.summary())

x_train2,x_test2,y_train2,y_test2=train_test_split(X_all3,total_published_time, test_size=0.25,random_state=0)
lm2 = LinearRegression()
lm2.fit(x_train2,y_train2)

#SVR

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

svr_rbf = SVR("rbf").fit(x_train,y_train)

svr_params={"C": [0.1,0.4,5,10,20,30,40,50]}
svr_cv_model=GridSearchCV(svr_rbf,svr_params,cv=10)
svr_cv_model.fit(x_train,y_train)

print(pd.Series(svr_cv_model.best_params_)[0])
svr_tuned=SVR("rbf", C=pd.Series(svr_cv_model.best_params_)[0]).fit(x_train,y_train)

#print(np.sqrt(mean_squared_error(y_test,y_pred3)))

import joblib
import pickle 
saved_model = pickle.dumps(lm)
lm_from_pickle = pickle.loads(saved_model)

joblib.dump(lm, 'model1.pkl')
  
saved_model2 = pickle.dumps(lm2)
lm2_from_pickle = pickle.loads(saved_model2)

joblib.dump(lm2, 'model2.pkl')

saved_model3 = pickle.dumps(svr_cv_model)
svr_cv_model_from_pickle = pickle.loads(saved_model3)

joblib.dump(svr_cv_model, 'svr_cv_model.pkl')
  
saved_model4 = pickle.dumps(svr_tuned)
svr_tuned_from_pickle = pickle.loads(saved_model4)

joblib.dump(svr_tuned, 'svr_tuned_model.pkl')

"""
# korelasyon ısı haritası 
correlations = new_df.corr()
import seaborn as sns #tüm veri kümesi için ısı haritası çıkarma.
import matplotlib.pyplot as plt
sns.heatmap(correlations)
plt.show()
print("building_age",correlations["building_age"]["total_published_time"])
"""
"""
# pivot table 
area = pd.cut(new_df["area"],[50,100,200])
room_count = pd.cut(new_df["room_count"],[1,3,5])
floor = pd.cut(new_df["floor"],[0,10,20])
price = pd.cut(new_df["price"],[4000,2000000,20000000])
building_age = pd.cut(new_df["building_age"],[0,15,40])
area.head(10)
new_df.pivot_table("total_published_time", ["has_lift","area","room_count","floor","price","building_age","has_parking_space"])

"""


