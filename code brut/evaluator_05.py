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
from sklearn.model_selection import KFold
from preparator_25 import preparator


oxides=['SiO2','MgO','Na2O','Al2O3','CaO']
def normalize_stat(data):
    data=(data - data.mean()) / data.std(ddof = 0)

#performances pour regressions polynomiales
def test_model(index_fold,x_train,x_test,y_train,y_test,deg):
    polynomial_features=PolynomialFeatures(degree=deg)
    poly_regression_alg=LinearRegression()
    model=Pipeline(
        [("polynomial_features", polynomial_features),
        ("linear_regression", poly_regression_alg)])
    model.fit(x_train,y_train)
    test_predictions=model.predict(x_test)
    rmse=np.sqrt(mean_squared_error(y_test,test_predictions))
    r2=r2_score(y_test,test_predictions)
    #print(f"Run {index_fold}:RMSE={round(rmse,2)}-R2_score={round(r2,2)}")
    return (rmse,r2)

def mean_perf(file_name,cols,prop,nb_tests=10,deg=2,show=False):
    index_fold, average_rmse, average_r2=0, 0, 0
    df=pd.read_csv(file_name+'.csv')
    data=df[cols] ; y=df[[prop]]
    normalize_stat(data) ; normalize_stat(y)
    kf=KFold(n_splits=nb_tests, shuffle=True)
    for train_index, test_index in kf.split(data):
        x_train, x_test=data.iloc[train_index], data.iloc[test_index]
        y_train, y_test=y.iloc[train_index], y.iloc[test_index]
        current_rmse,current_r2=test_model(index_fold,x_train,x_test,y_train,y_test,deg)
        average_rmse=average_rmse+current_rmse
        average_r2=average_r2+current_r2
        index_fold+=1
    average_rmse=average_rmse/nb_tests
    average_r2=average_r2/nb_tests
    if show:
        print(f"(file = {file_name}) Moyenne : RMSE={round(average_rmse,2)}-R2-score={round(average_r2,2)}")
    return average_r2,average_rmse,len(df)

def compilator(file_name,property,oxides,nmin_min, nmin_max,nb_nmin,alpha_min,alpha_max,nb_alpha,gap_min,gap_max,nb_gap,sg):
    Nmin=[nmin_min+i*(nmin_max-nmin_min)/nb_nmin for i in range(nb_nmin)]
    #Nmin=np.linspace(nmin_min,nmin_max,nb_nmin)
    Alpha=[alpha_min+i*(alpha_max-alpha_min)/nb_alpha for i in range(nb_alpha)]
    #Alpha=np.linspace(alpha_min,alpha_max,nb_alpha)
    Gap=[gap_min+i*(gap_max-gap_min)/nb_gap for i in range(nb_gap)]
    #Slice=[ns_min + i*(ns_max-ns_min)/nb_ns for i in range(nb_ns)
    names=[]
    n=[]
    X,Y,Z=[],[],[]
    for g,gap in enumerate(Gap):
        for nmin in Nmin:
            for a,alpha in enumerate(Alpha):
                
                names.append('nmin='+str(nmin)+','+'Alpha='+str(a)+','+'Gap='+str(g))
                
                
                if preparator(file_name,property,oxides,nmin,gap,alpha,add_to_name=names[-1],nb_slices_init=5):
                    df=pd.read_csv(file_name+names[-1]+'.csv')
                    if len(df)>=min_size:
                        n.append('nmin='+str(nmin)+','+'Alpha='+str(a)+','+'Gap='+str(g))
                        n.append('|')
                    
                        X.append(nmin)
                        Y.append(alpha)
                        Z.append(gap)
    f=open('names'+sg+'.csv','x')
    f.writelines(n)
    f.close()
    with open('x'+sg+'.csv','x') as s:
        for x in X:
            s.write('%s'%x)
            s.write(',')
    with open('y'+sg+'.csv','x') as s:
        for y in Y:
            s.write('%s'%y)
            s.write(',')
    with open('z'+sg+'.csv','x') as s:
        for z in Z:
            s.write('%s'%z)
            s.write(',')

def perf(file_name,oxides,proprety,sg):
    float_v=np.vectorize(float)
    with open('names'+sg+'.csv','r') as filestream:
        for line in filestream:
            c=line.split('|')
    names=c[0:-1]
    with open('x'+sg+'.csv','r') as s:
        for line in s:
            c=line.split(',')
    X=float_v(np.array(c[0:-1]))
    X=list(X)
    with open('y'+sg+ '.csv','r') as s:
        for line in s:
            c=line.split(',')
    Y=float_v(np.array(c[0:-1]))
    Y=list(Y)
    with open('z'+sg+'.csv','r') as s:
        for line in s:
            c=line.split(',')
    Z=float_v(np.array(c[0:-1]))
    Z=list(Z)

    rmses,scores,sizes=[],[],[]
    for name in names:
        score,rmse,size=mean_perf(file_name+name,oxides,proprety)
        if score<0:
            score =0
        scores.append(score)
        rmses.append(rmse)
        sizes.append(size)
    
    return scores,rmses,sizes,X,Y,Z

def graphe(scores,size_val,rmse_val,X,Y):  
    fig=plt.figure()
    
    ax1=plt.axes(projection='3d')
    ax1.scatter3D(X, Y, rmse_val,c='red', marker='x')
    ax1.set_xlabel('nmin')
    ax1.set_ylabel('alpha')
    ax1.set_zlabel('RMSE')
    plt.show()
    
    
    
    #ax2 = fig.add_subplot(312, projection='3d')
    ax2=plt.axes(projection='3d')
    ax2.scatter3D(X, Y, scores,c='red', marker='x')
    ax2.set_xlabel('nmin')
    ax2.set_ylabel('alpha')
    ax2.set_zlabel('R2_score')
    plt.show()


    #ax3 = fig.add_subplot(313, projection='3d')
    ax3=plt.axes(projection='3d')
    ax3.scatter3D(X, Y, size_val,c='red', marker='x')
    ax3.set_xlabel('nmin')
    ax3.set_ylabel('alpha')
    ax3.set_zlabel('size')
    plt.show()

def scatter_4D_score(scores,X,Y,Z):
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')

    img=ax.scatter(X,Y,Z,c=scores,cmap=plt.hot())
    ax.set_xlabel('nmin')
    ax.set_ylabel('alpha')
    ax.set_zlabel('gap')
    fig.colorbar(img)
    plt.show()
def scatter_4D_rmse(rmses,X,Y,Z):

    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')

    img=ax.scatter(X,Y,Z,c=rmses,cmap=plt.hot())
    ax.set_xlabel('nmin')
    ax.set_ylabel('alpha')
    ax.set_zlabel('gap')
    fig.colorbar(img)
    plt.show()
def scatter_4D_size(sizes,X,Y,Z):
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')

    img=ax.scatter(X,Y,Z,c=sizes,cmap=plt.hot())
    ax.set_xlabel('nmin')
    ax.set_ylabel('alpha')
    ax.set_zlabel('gap')
    fig.colorbar(img)
    plt.show()

def scatter_4D(scores,rmses,sizes,X,Y,Z,arg):
    if arg=='score':
        scatter_4D_score(scores,X,Y,Z)
    if arg=='rmse':
        scatter_4D_rmse(rmses,X,Y,Z)
    if arg=='size':
        scatter_4D_size(sizes,X,Y,Z)
    else:
        return 'error: arg doit être choisi parmis: score, rmse, size'

'''compilator('density','Density',oxides,
nmin_min=50,nmin_max=150,nb_nmin=10,
alpha_min=0.01,alpha_max=0.9,nb_alpha=10,
gap_min=0.01,gap_max=0.3,nb_gap=10,
sg='w')'''

scores,rmses,sizes,X,Y,Z=perf(file_name='density',oxides=oxides,proprety='Density',sg='w')

scores,rmses,sizes=np.array(scores),np.array(rmses),np.array(sizes)
    
def opti(scores,rmses,sizes,X,Y,Z):
    sc=scores/np.max(scores) +2
    rm=abs(rmses/np.max(rmses))    +2
    si=sizes/np.max(sizes)    +2
    a=0.4*sc-0.4*rm+0.2*si
    max_a=np.max(a)
    i=np.where(a==max_a)[0][0]
    return scores[i],rmses[i],sizes[i],X[i],Y[i],Z[i]

score,rmse,size,nmin,alpha,gap=opti(scores,rmses,sizes,X,Y,Z)
print('filtrage optimal: score=',score,';rmse=',rmse,';size=',size,'.pour les paramètres:nmin=',nmin,';alpha=',alpha,';gap=',gap )

def smart(file_name,property,oxides,nmin_min, nmin_max,nb_nmin,alpha_min,alpha_max,nb_alpha,gap_min,gap_max,nb_gap,sg,size_min):
    compilator(file_name,property,oxides,nmin_min, nmin_max,nb_nmin,alpha_min,alpha_max,nb_alpha,gap_min,gap_max,nb_gap,size_min,sg)
    scores,rmses,sizes,X,Y,Z=perf(file_name='density',oxides=oxides,proprety='Density',sg=sg)
    scores,rmses,sizes=np.array(scores),np.array(rmses),np.array(sizes)
    score,rmse,size,nmin,alpha,gap=opti(scores,rmses,sizes,X,Y,Z)
    print('filtrage optimal: score=',score,';rmse=',rmse,';size=',size,'.pour les paramètres:nmin=',nmin,';alpha=',alpha,';gap=',gap )
    return ['score:{score}','rmse={rmse}','size={size}','nmin={nmin}','alpha={alpha}','gap={gap}']
    
ox=['SiO2','Al2O3','MgO','CaO','Na2O','K2O','SO3']
    




#scatter_4D(scores,rmses,sizes,X,Y,Z,'score')
                



#graphe('density','Density',oxides,nmin_min=10,nmin_max=500,nb_nmin=10,alpha_min=0.01,alpha_max=0.9,nb_alpha=10,gap=0.6)
'''scores,sizes,rmses,X,Y,Z=evaluator('density','Density',oxides,
                                   nmin_min=10,nmin_max=500,nb_nmin=3,
                                    alpha_min=0.01,alpha_max=0.9,nb_alpha=5,
                                    gap_min=0.1,gap_max=0.9,nb_gap=2)'''
#scatter_4D_score(scores,X,Y,Z)
#scatter_4D_rmse(rmses,X,Y,Z)
#scatter_4D_size(sizes,X,Y,Z)


"""
#exemple utilisation
evaluate('density_table_tableonly_data_clean',['SiO2', 'Al2O3', 'MgO', 'CaO',
'Na2O'],'Density',deg=3)


evaluate('density_table_tableonly_data_clean_prepared',['SiO2', 'Al2O3', 'MgO', 'CaO',
'Na2O'],'Density',deg=3)

evaluate('density_table_tableonly_data_clean_prepared2',['SiO2', 'Al2O3', 'MgO', 'CaO',
'Na2O'],'Density',deg=3)

evaluate('density_table_tableonly_data_clean_prepared3',['SiO2', 'Al2O3', 'MgO', 'CaO',
'Na2O'],'Density',deg=3)
"""