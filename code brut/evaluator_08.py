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


oxides=['SiO2','MgO','Na2O','Al2O3','CaO']

#size = 496
#nmin=80
#alpha=0.633
#gap=0.184

def polynomial_regressor(x,y,deg=1):
    polynomial_features=PolynomialFeatures(degree=deg)
    poly_regression_alg=LinearRegression()
    model=Pipeline(
        [("polynomial_features", polynomial_features),
        ("linear_regression", poly_regression_alg)])
    model.fit(x,y)
    return model

def mlp_regressor(x,y,deg):
    mlp_regressor=MLPRegressor(hidden_layer_sizes=(300,),max_iter=10000,)
    mlp_regressor.fit(x,y)
    return mlp_regressor

def perf(f,x,y):
    predictions=f.predict(x)
    rmse=(mean_squared_error(y,predictions))
    r2=r2_score(y,predictions)
    return rmse,r2

def average_perf(regressor,df,x_cols,y_col,nb_tests=5):
    data=df[x_cols] ; ytot=df[y_col]
    avg_rmse,avg_r2=0,0
    kf=KFold(n_splits=nb_tests,shuffle=True)
    for train_index, test_index in kf.split(data):
        x_train, x_test=data.iloc[train_index], data.iloc[test_index]
        y_train, y_test=ytot.iloc[train_index], ytot.iloc[test_index]
        f=regressor(x_train,y_train)
        p=perf(f,x_test,y_test)
        avg_rmse+=p[0]
        avg_r2+=p[1]
    return avg_rmse/nb_tests, avg_r2/nb_tests
    

def evaluator(regressor,file_name,property,oxides,nmin_min, nmin_max,nb_nmin,alpha_min,alpha_max,nb_alpha,gap_min,gap_max,nb_gap,nb_si):
    Nmin=np.linspace(nmin_min,nmin_max,nb_nmin)
    Alpha=np.linspace(alpha_min,alpha_max,nb_alpha)
    Gap=np.linspace(gap_min,gap_max,nb_gap)
    NMIN,GAP,ALPHA,RMSE,R2,SIZE=[],[],[],[],[],[]
    for g,gap in enumerate(Gap):
        for nmin in Nmin:
            for a,alpha in enumerate(Alpha):         
                b,prep=preparator(file_name,property,oxides,nmin,alpha,nb_slices=nb_si)
                if b:              
                    NMIN.append(nmin)
                    GAP.append(gap)
                    ALPHA.append(alpha)
                    rmse,r2=average_perf(regressor,prep,oxides,property)
                    RMSE.append(rmse)
                    if r2<0: R2.append(0)
                    else: R2.append(r2)
                    SIZE.append(len(prep))
    return NMIN,GAP,ALPHA,RMSE,R2,SIZE

def graphe(X,Y,Z,T,combined=True):  
    for i in range(len(T)):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        img=ax.scatter(X[0],Y[0],Z[0],c=T[i][0],cmap=plt.hot())
        ax.set_xlabel(X[1])
        ax.set_ylabel(Y[1])
        ax.set_zlabel(Z[1])
        fig.colorbar(img,label=T[i][1])
    plt.show()

oxides=['SiO2','MgO','Na2O','Al2O3','CaO']
#NMIN,GAP,ALPHA,RMSE,R2,SIZE=evaluator(mlp_regressor,'density_table_tableonly_data_clean','Density',oxides,50, 100,5,0.1,0.2,5,0.1,0.3,5,5)
#graphe([NMIN,'nmin'],[GAP,'gap'],[ALPHA,'alpha'],[(RMSE,'rmse'),(R2,'r2'),(SIZE,'size')],combined=True)


def create_evaluate_model_polynomiale(index_fold,x_train,x_test,y_train,y_test,deg):
    polynomial_features=PolynomialFeatures(degree=deg)
    poly_regression_alg=LinearRegression()

    model=Pipeline([
    ("polynomial_features", polynomial_features),
    ("linear_regression", poly_regression_alg)])

    regression_alg=LinearRegression()
    model.fit(x_train,y_train)
    test_predictions=model.predict(x_test)
    rmse=np.sqrt(mean_squared_error(y_test,test_predictions))
    r2=r2_score(y_test,test_predictions)
    #print(f"Run {index_fold}:RMSE={round(rmse,2)}-R2_score={round(r2,2)}")
    return (rmse,r2)
def eval(df0,deg):
    grandeur_mesuree=df0.columns[-1]
    data=df0[df0.columns[0:-1]]
    y=df0[[grandeur_mesuree]]
    data=(data - data.mean()) / data.std(ddof = 0)
    nb_model=10
    kf=KFold(n_splits=nb_model, shuffle=True)
    index_fold=0
    average_rmse=0
    average_r2=0
    for train_index, test_index in kf.split(data):
        x_train, x_test=data.iloc[train_index], data.iloc[test_index]
        y_train, y_test=y.iloc[train_index], y.iloc[test_index]
        current_rmse,current_r2=create_evaluate_model_polynomiale(index_fold,x_train,x_test,y_train,y_test,deg)
        average_rmse=average_rmse+current_rmse
        average_r2=average_r2+current_r2
        index_fold= index_fold +1
    average_rmse=average_rmse/nb_model
    average_r2=average_r2/nb_model
    #print(f"Moyenne : RMSE={round(average_rmse,2)}-R2-score={round(average_r2,2)}")
    return average_rmse,average_r2