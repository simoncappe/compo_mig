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


df0=pd.read_csv('densityo.csv')
grandeur_mesuree=df0.columns[-1]
data=df0[df0.columns[0:-1]]
y=df0[[grandeur_mesuree]]

x_train,x_test,y_train,y_test=train_test_split(data,y,test_size=0.2)
polynomial_features=PolynomialFeatures(degree=5)
poly_regression_alg=LinearRegression()
model=Pipeline([
    ("polynomial_features", polynomial_features),
    ("linear_regression", poly_regression_alg)
])
for k in range(3):
    model.fit(x_train,y_train)

ptest=model.predict(x_test)
ptrain=model.predict(x_train)
ptrain=ptrain/ptrain.max()
ptest=ptest/ptest.max()
y_train=y_train/np.max(y_train)
y_test=y_test/np.max(y_test)

rmsetest=round(np.sqrt(mean_squared_error(y_test,ptest)),2)
rtest=round(r2_score(y_test,ptest),2)
rmsetrain=round(np.sqrt(mean_squared_error(y_train,ptrain)),2)
rtrain=round(r2_score(y_train,ptrain),2)

fig,ax=plt.subplots(1,2)
ax[0].scatter(ptrain,y_train,marker='x',c='black')
ax[0].plot(ptrain,ptrain,c='red')
ax[0].set_xlabel('valeurs prédites')
ax[0].set_ylabel('valeurs mesurées')
ax[0].set_title('valeurs d entraînement ; r2 = '+ str(rtrain)+' et rmse = ' + str(rmsetrain))



ax[1].scatter(ptest,y_test,marker='x',c='black')
ax[1].plot(ptest,ptest,c='red')
ax[1].set_xlabel('valeurs prédites')
ax[1].set_ylabel('valeurs mesurées')
ax[1].set_title('valeurs de test ; r2 = '+ str(rtest)+' et rmse = ' + str(rmsetest))


plt.show()





#print(f"RMSE= {round(np.sqrt(mean_squared_error(y_test,train_predictions)),2)}")
#print(f"R2_score={round(r2_score(y_test,train_predictions),2)}")