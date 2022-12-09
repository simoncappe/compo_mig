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
def eval_lin(df0,deg):
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
def eval_mlp(df0,hidden_layer,iter):
    grandeur_mesuree=df0.columns[-1]
    data=df0[df0.columns[0:-1]]
    y=df0[[grandeur_mesuree]]
    avg_rmse,avg_r2=0,0

    nb_tests=5

    kf=KFold(n_splits=nb_tests,shuffle=True)
    for train_index, test_index in kf.split(data):
        
        x_train, x_test=data.iloc[train_index], data.iloc[test_index]
        y_train, y_test=y.iloc[train_index], y.iloc[test_index]

        mlp=MLPRegressor(hidden_layer_sizes=(hidden_layer,),max_iter=iter,)
        mlp.fit(x_train,y_train)

        y_predict=mlp.predict(x_test)

        rmse=(mean_squared_error(y_test,y_predict))
        r2=r2_score(y_test,y_predict)



        avg_rmse+=rmse
        avg_r2+=r2


    avg_rmse,avg_r2=avg_rmse/nb_tests, avg_r2/nb_tests
    return avg_rmse,avg_r2

    
