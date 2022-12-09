import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from preparator_33  import preparator,compare_slicings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from evaluator_08 import mlp_regressor,perf,average_perf,eval
oxides=['SiO2','MgO','Na2O','Al2O3','CaO']
ox=['SiO2','Al2O3','MgO','CaO','Na2O','K2O','SO3']
df1=pd.read_csv('density.csv')
#compare_slicings(df1,oxides,nmin=50,n1=1,n2=30)
#preparator('tenace','Fracture Toughness',ox,nmin=10,alpha=0.1,add_to_name='o',nb_slices=7)

#preparator('density','Density',nmin=50,alpha=0.1,
#add_to_name='o',oxides=oxides,nb_slices=7)
df0=pd.read_csv('ymo.csv')
df1=pd.read_csv('ym.csv')
df0=df0[ox+['Young']]
dg=2



grandeur_mesuree=df0.columns[-1]
data=df0[df0.columns[0:-1]]
y=df0[[grandeur_mesuree]]



x_train,x_test,y_train,y_test=train_test_split(data,y,test_size=0.1)
polynomial_features=PolynomialFeatures(degree=dg)
poly_regression_alg=LinearRegression()
model=Pipeline([
    ("polynomial_features", polynomial_features),
    ("linear_regression", poly_regression_alg)
])
model.fit(x_train,y_train)

rmse0,r20=eval(df0,dg)
rmse0,r20=round(rmse0,2),round(r20,2)
predict=model.predict(x_test)
#print(predict.max(axis=0))
predict=predict/predict.max(axis=None)
vals=y_test/np.max(y_test)
fig,ax=plt.subplots(1,2)
ax[1].scatter(vals,predict,marker='x',c='black')
ax[1].set_xlabel('valeurs mesurées')
ax[1].set_ylabel('valeurs prédites')
ax[1].plot(vals,vals,c='red')
ax[1].set_title('données préparées ; r2 = '+str(round(r20,2)) +' et rmse = '+str(round(rmse0,0)))

df0=df1
randeur_mesuree=df0.columns[-1]
data=df0[df0.columns[0:-1]]
y=df0[[grandeur_mesuree]]
x_train,x_test,y_train,y_test=train_test_split(data,y,test_size=0.1)
polynomial_features=PolynomialFeatures(degree=dg)
poly_regression_alg=LinearRegression()
model=Pipeline([
    ("polynomial_features", polynomial_features),
    ("linear_regression", poly_regression_alg)
])
model.fit(x_train,y_train)


rmse0,r20=eval(df0,dg)
rmse0,r20=round(rmse0,2),round(r20,2)

predict=model.predict(x_test)
predict=predict/predict.max(axis=None)
vals=y_test/np.max(y_test)

ax[0].scatter(vals,predict,marker='x',c='black')
ax[0].plot(vals,vals,c='red')
ax[0].set_xlabel('valeurs mesurées')
ax[0].set_ylabel('valeurs prédites')
ax[0].set_title('données brutes ; r2 = '+str(round(r20,2)) +' et rmse = '+str(round(rmse0,0)))
plt.show()