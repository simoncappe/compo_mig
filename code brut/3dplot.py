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
from preparator_33 import preparator
from findor_03 import domains
ox=['SiO2','Al2O3','MgO','CaO',
 'Na2O','K2O','SO3']

df0=pd.read_csv('ymo.csv')
oxides=['SiO2', 'MgO', 'Na2O', 'Al2O3', 'CaO']
df0=df0[ox+['Young']]
grandeur_mesuree=df0.columns[-1]
data=df0[df0.columns[0:-1]]
y=df0[[grandeur_mesuree]]
x_train,x_test,y_train,y_test=train_test_split(data,y,test_size=0.2)
polynomial_features=PolynomialFeatures(degree=1)
poly_regression_alg=LinearRegression()
model=Pipeline([
    ("polynomial_features", polynomial_features),
    ("linear_regression", poly_regression_alg)
])
model.fit(x_train,y_train)
test_predictions=model.predict(x_test)

f=open('densityodomains_nbslices=10.npy', 'rb')
id_list=np.load(f,allow_pickle=True)
doms=domains(id_list)
dom=doms[3]
oxides=['SiO2', 'MgO',
 'Na2O', 'Al2O3', 'CaO']
ox=['SiO2','Al2O3','MgO','CaO',
 'Na2O','K2O','SO3']
r=round(r2_score(y_test,test_predictions),2)
rmse=round(np.sqrt(mean_squared_error(y_test,test_predictions)),2)
xsi=72.50
xmg=3.70
xca=7.
xso=0.1
xko=0.1

def fc(xal,xna):
    z=np.array([xmg,xna,xal,xca,xso,xko])
    x=np.array([[100.-np.sum(z),xal,xmg,xca,xna,xso,xko]])
    p=pd.DataFrame(x,columns=ox)
    return model.predict(p)[0][0]
x=[70.61,3.7,14.99,3.3,7.4]
"""def density(x):
    x=np.array([x])
    p=pd.DataFrame(x,columns=oxides)
    return model.predict(p)[0][0]"""
def young(x):
    x=np.array([x])
    p=pd.DataFrame(x,columns=ox)
    return model.predict(p)[0][0]
print(rmse,r)
x1=[70.41,3.3,3.70,7.40,14.99,0.10,0.10]
e1=young(x1)
x2=[69.11,3.30,3.70,8.70,14.99,0.1,0.1]
e2=young(x2)
e_predit_2=74.18
x3=[69.11,3.30,5.0,7.40,14.99,0.1,0.1]
e_predit_3=74.45
e3=young(x3)
print('e1 = ',e1,'; e2 = ',e2,'; e3 = ',e3)

#den=round(density(x),2)
#print(den,'+/-',rmse)
#print('coefficient de corrélation = ',r)
min_al,max_al=0.,10.
min_na,max_na=10.,20.
N=100
ax_x=np.linspace(min_al,max_al,N)
ax_y=np.linspace(min_na,max_na,N)
fcvect=np.vectorize(fc)

x,y=np.meshgrid(ax_x,ax_y)
z=fcvect(x,y)
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.plot_wireframe(x,y,z)
ax.set_xlabel('Al2O3')
ax.set_ylabel('Na2O')
ax.set_zlabel('Module de Young (GPa)')
ax.set_title('Evolution du module de Young, incertitude = ' + str(rmse) + ' GPa')
plt.show()


        
