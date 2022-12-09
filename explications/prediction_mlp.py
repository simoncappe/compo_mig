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
from eval import eval_mlp


ox=['SiO2','Al2O3','MgO',  #les oxides intervenant dans 
'CaO','Na2O','K2O','SO3']  #mon modèle


hidden_layer=100  #le nombres de couches de mon réseau
iter=1000         #le maximum d'itération

df=pd.read_csv('ym_prepared.csv',usecols=ox+['Young'])
grandeur_mesuree=df.columns[-1]
data=df[df.columns[0:-1]]
y=df[[grandeur_mesuree]]
x_train,x_test,y_train,y_test=train_test_split(data,y,test_size=0.2)
mlp_regressor=MLPRegressor(hidden_layer_sizes=(hidden_layer,),max_iter=iter,)
mlp_regressor.fit(x_train,y_train)


def f(x):
    '''fonction qui prend un array 
    ou une liste en argument
    correspondant aux compositions
    et renvoie la grandeur choisie
    (en l'occurence le module de Young)'''
    x=np.array([x])
    p=pd.DataFrame(x,columns=ox)
    return mlp_regressor.predict(p)[0]

rmse,r=eval_mlp(df0=df,hidden_layer=hidden_layer,iter=iter)
#rmse=incertitude;r=coefficient de corrélation