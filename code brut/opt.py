import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from preparator_25 import preparator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures



def f_min(f,comp_possible):
    '''n=len(cube)
    L=[[cube[i][0],cube[i][1]] for i in range(n)]
    g=[]
    T=[]
    for i in range(n):
        N=int((cube[i][1]-cube[i][0])/step)
        Y=[cube[i][0]+i*(cube[i][1]-cube[i][0])/N for i in range(N)]
        if Y!=[]:
            L[i]=Y
    for i0 in L[0]:
        for i1 in L[1]:
            for i2 in L[2]:
                for i3 in L[3]:
                    for i4 in L[4]:
                        x=[i0,i1,i2,i3,i4]
                        x=np.array(x)
                        if (np.sum(x)-1)<10**(-):
                            print('a')
                            
                            T.append(np.array(x))
                    
                            g.append(f(x))'''
    g=[]
    T=[]
    for comp in comp_possible:
        g.append(f(comp))
        T.append(comp)


    g=np.array(g)
    res=np.min(g)
    indice=np.where(g==res)[0][0]
    return res,T[indice]
#for x in np.arange(a,b,delta)
def trouve_compos(cube,step):
    n=len(cube)
    compos=[]
    for x0 in np.arange(cube[0][0],cube[0][1],step):
        for x1 in np.arange(cube[1][0],max(cube[1][1],1-x0),step):
            for x2 in np.arange(cube[2][0],max(cube[2][1],1-x0-x1),step):
                for x3 in np.arange(cube[3][0],max(cube[3][1],1-x0-x1-x2),step):
                    x4=1-x0-x1-x2-x3
                    if x4<=cube[4][1] and x4>=cube[4][0]:
                        x=np.array([x0,x1,x2,x3,x4])
                        compos.append(x)
    return compos
                        

    