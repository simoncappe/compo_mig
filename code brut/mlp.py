import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, train_test_split

from evaluator_08 import mlp_regressor,perf,average_perf,eval
ox=['SiO2','Al2O3','MgO','CaO','Na2O','K2O','SO3']
oxides=['SiO2','MgO','Na2O','Al2O3','CaO']
df0=pd.read_csv('ymo.csv')
df0=df0[ox+['Young']]


df1=pd.read_csv('ym.csv')
grandeur_mesuree=df0.columns[-1]
data=df0[df0.columns[0:-1]]
y=df0[[grandeur_mesuree]]
x_train,x_test,y_train,y_test=train_test_split(data,y,test_size=0.2)
mlp_regressor=MLPRegressor(hidden_layer_sizes=(300,),max_iter=10000,)
mlp_regressor.fit(x_train,y_train)
predict=mlp_regressor.predict(x_test)
predi=predict/predict.max(axis=None)
y_=y_test/np.max(y_test)
nb_tests=5
avg_rmse,avg_r2=0,0
kf=KFold(n_splits=nb_tests,shuffle=True)
for train_index, test_index in kf.split(data):
    x_train, x_test=data.iloc[train_index], data.iloc[test_index]
    y_train, y_test=y.iloc[train_index], y.iloc[test_index]
    f=mlp_regressor.fit(x_train,y_train)
    p=perf(f,x_test,y_test)
    avg_rmse+=p[0]
    avg_r2+=p[1]
avg_rmse,avg_r2=avg_rmse/nb_tests, avg_r2/nb_tests
avg_rmse,avg_r2
fig,ax=plt.subplots(1,2)
ax[1].scatter(predi,y_,marker='x',c='black')
ax[1].plot(y_,y_,c='red')
ax[1].set_xlabel('valeurs prédites')
ax[1].set_ylabel('valeurs mesurées')
ax[1].set_title('données traitées ; r2 = '+str(round(avg_r2,2))+' et rmse = '+str(round(avg_rmse,0)))

df0=df1
grandeur_mesuree=df0.columns[-1]
data=df0[df0.columns[0:-1]]
y=df0[[grandeur_mesuree]]
x_train,x_test,y_train,y_test=train_test_split(data,y,test_size=0.2)
mlp_regressor=MLPRegressor(hidden_layer_sizes=(300,),max_iter=10000,)
mlp_regressor.fit(x_train,y_train)
predict=mlp_regressor.predict(x_test)
predi=predict/predict.max(axis=None)
y_=y_test/np.max(y_test)
nb_tests=2
avg_rmse,avg_r2=0,0
kf=KFold(n_splits=nb_tests,shuffle=True)
for train_index, test_index in kf.split(data):
    x_train, x_test=data.iloc[train_index], data.iloc[test_index]
    y_train, y_test=y.iloc[train_index], y.iloc[test_index]
    f=mlp_regressor.fit(x_train,y_train)
    p=perf(f,x_test,y_test)
    avg_rmse+=p[0]
    avg_r2+=p[1]
avg_rmse,avg_r2=avg_rmse/nb_tests, avg_r2/nb_tests
avg_rmse,avg_r2
ax[0].scatter(predi,y_,marker='x',c='black')
ax[0].plot(y_,y_,c='red')
ax[0].set_xlabel('valeurs prédites')
ax[0].set_ylabel('valeurs mesurées')
ax[0].set_title('données brutes ; r2 = '+str(round(avg_r2,2))+' et rmse = '+str(round(avg_rmse,0)))

plt.show()

