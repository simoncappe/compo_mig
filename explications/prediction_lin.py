import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from preparator  import preparator,compare_slicings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from eval import eval_lin


ox=['SiO2','Al2O3','MgO','CaO',     #Les oxides qui
'Na2O','K2O','SO3']                 #interviennes dans ma prédiction

dg=1      #le degrès de ma régression

df=pd.read_csv('ym_prepared.csv',usecols=ox+['Young']) #le fichier a partir duquel je vais prédire

grandeur_mesuree=df.columns[-1]
data=df[df.columns[0:-1]]
y=df[[grandeur_mesuree]]

x_train,x_test,y_train,y_test=train_test_split(data,y,test_size=0.1)
polynomial_features=PolynomialFeatures(degree=dg)
poly_regression_alg=LinearRegression()
model=Pipeline([
    ("polynomial_features", polynomial_features),
    ("linear_regression", poly_regression_alg)
])
model.fit(x_train,y_train)#entraînerment de mon modèle

def f(x):
    '''fonction qui prend un array 
    ou une liste en argument
    correspondant aux compositions
    et renvoie la grandeur choisie
    (en l'occurence le module de Young)'''
    x=np.array([x])
    p=pd.DataFrame(x,columns=ox)
    return model.predict(p)[0][0]

rmse,r=eval_lin(df0=df,deg=dg)#rmse= incertitude du modèle;
                              #r=coefficient de corrélation

