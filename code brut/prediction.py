import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from preparator_33 import preparator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from opt import f_min,trouve_compos
from findor_02 import domains
oxides=['SiO2','MgO','Na2O','Al2O3','CaO']
#preparator('density','Density',oxides,nmin=80,alpha=0.633,gap=0.184,add_to_name='o',domains=True,nb_slices_init=10)

def ideal(file_name,oxides,deg,precision,v='min'):
    
    
    f=open(file_name+'domains.txt','r')#on trouve les domaines de définitions de la régression à venir
    id_list=domains(f.readline())
    doms=domains(id_list)
    
    
    
    df0=pd.read_csv(file_name+'.csv')#j'ouvre la dataframe pour la régresser
    grandeur_mesuree=df0.columns[-1]
    data=df0[df0.columns[0:-1]]#mes antécédents
    y=df0[[grandeur_mesuree]]#mes images
    x_train,x_test,y_train,y_test=train_test_split(data,y,test_size=0.2)
    polynomial_features=PolynomialFeatures(degree=deg)
    poly_regression_alg=LinearRegression()
    model=Pipeline([
    ("polynomial_features", polynomial_features),
    ("linear_regression", poly_regression_alg)
    ])
    model.fit(x_train,y_train)#c'est bon, mon modèle est régressé

    def f(x):
        x=100*np.array([x])
        p=pd.DataFrame(x,columns=oxides)#ma fonction ne mange que des dataframes
        return model.predict(p)[0][0]#j'ai entraîné mon modèle sur des pourcents, mon x est en fraction
    #la fonction f rend la valeur prédite par la régression

    #je  veux maintenant optimiser (maximiser ou minimiser)
    compos=[]
    vals=[]
    for dom in doms:
        compos_possibles=trouve_compos(dom,precision)#trouve les compos possibles dans le domaine avec une précision donnée
        val,compo=f_min(f,compos_possibles)#donne le minimum de densité parmis les compos possibles
        vals.append(val)
        compos.append(compo)
    return compos,vals
    
    
#preparator('density','Density',oxides,nmin=80,alpha=0.633,gap=0.184,add_to_name='o',domains=True,nb_slices_init=10)
#vals,comps=ideal('densityo',2,0.05)
#print(vals,comps)