import numpy as np ; import pandas as pd ; import matplotlib.pyplot as plt
ox=['SiO2','MgO','Na2O','Al2O3','CaO']
def classify(limits):
    def aux_classify(x):
        #classe x dans la classe de borne sup limits[b]
        a,b=-1,len(limits)-1
        while (b-a)>1:
            c=(b+a)//2
            if limits[c]<x: a=c
            else: b=c
        return b
    return aux_classify

def data_sliceor(df,oxides,nmin,nb_slices,return_data=False):
    data=df.copy()
    initial_cols=data.columns
    limits=np.arange(start=100/nb_slices,stop=100+100/nb_slices,step=100/nb_slices)
    for i,oxide in enumerate(oxides):
        data[i]=data[oxide].apply(classify(limits))
        cubes=data[i].unique()
        grby=data.groupby(i,sort=False)
        size_select_min=(grby.size()>nmin)
        for cube in cubes[~size_select_min]:
            data.drop(labels=grby.get_group(cube).index)
    data['cube']=data[[i for i in range(len(oxides))]].apply(tuple,axis=1)
    grby=data.groupby('cube',sort=False).size().reset_index()
    cubes_selected=grby[grby[0]>nmin] 
    if return_data:
        return cubes_selected['cube'].unique(),data
    else:
        nbtot=cubes_selected[0].sum()
        return (nbtot,len(cubes_selected),nbtot/(len(cubes_selected)*(1/nb_slices)**(len(oxides))))

def compare_slicings(df,oxides,nmin,n1,n2):
    I=[] ; D=[] ; S=[] ; C=[]
    for i in range(n1,n2+1):
        I.append(i)
        d=data_sliceor(df,oxides,nmin,i)
        D.append(d[2])
        C.append(d[1])
        S.append(d[0])
    plt.figure()
    plt.scatter(I,D, label='density moyenne')
    plt.legend()
    plt.figure()
    plt.scatter(I,S, label='nb donnees')
    plt.legend()
    plt.figure()
    plt.scatter(I,C,label='nb cubes')
    plt.legend()
    plt.show()

def cube_cleanor(data,oxides,property,alpha,acc_data,slices=5):
    def aux_cube_cleanor(cube):
        df=data[data['cube']==cube].copy()
        maxi=df[property].max() ; mini=df[property].min()
        try:
            df['auxprop']=(df[property]-mini)/(maxi-mini)
        except ZeroDivisionError:
            print('propriete constante, normalisation max-min impossible')
        limits=np.array([(i+1)/slices for i in range(slices)])
        #decoupe selon la colonne property
        df['property classes']=df['auxprop'].apply(classify(limits))
        grby=df.groupby('property classes',sort=False)
        prop_classes=grby.size().reset_index()
        prop_classes.sort_values(by=0,axis=0,ascending=False)
        ref,sizemax=prop_classes.iloc[0]
        prop_classes.sort_values(by='property classes',axis=0,ascending=True)
        ref_index=(prop_classes['property classes']<ref).sum()
        keepmin=ref
        keepmax=ref
        #selectionne les donnees voisines de ref et conformes en taille
        for i0 in range(ref_index,-1,-1):
            if ref-prop_classes['property classes'].iloc[i0]==ref_index-i0 and prop_classes[0].iloc[i0]/sizemax>=alpha:
                keepmin=prop_classes['property classes'].iloc[i0]
            else:
                break
        for i1 in range(ref_index+1,len(prop_classes)):
            if (prop_classes['property classes'].iloc[i1])-ref==i1-ref_index and prop_classes[0].iloc[i1]/sizemax>=alpha:
                keepmax=prop_classes['property classes'].iloc[i1]
            else:
                break
        acc_data[cube]=df[(df['property classes']>=keepmin) & (df['property classes']<=keepmax)][oxides+[property]]
    return aux_cube_cleanor

def cleanor(data,cubes,property,oxides,alpha):
    acc_data={}
    np.vectorize(cube_cleanor(data,oxides,property,alpha,acc_data))(np.array(cubes))
    return acc_data


def preparator(file_name,property,oxides,nmin=50,alpha=0.1,add_to_name='_prepared_',nb_slices=100):
    df=pd.read_csv(file_name+'.csv')
    #lance decoupage
    cubes,data=data_sliceor(df,oxides,nmin,nb_slices,return_data=True)
    if len(cubes)==0:
        print('donnees non conformes aux exigences')
    else:
        acc_data=cleanor(data,cubes,property,oxides,alpha)
        #fusionne les donnees nettoyees
        prepared=pd.concat(acc_data.values(),ignore_index=True, sort=False)
        prepared.to_csv(file_name+add_to_name+'.csv')
        #enregistre si besoin les domaines regroupant des donnees selectionees
        np.save(file_name+add_to_name+'domains_nbslices='+str(nb_slices),cubes)

"""
#exemple d utilisation
oxides=['SiO2','MgO','Na2O','Al2O3','CaO']
df=pd.read_csv('density.csv')
compare_slicings(df,oxides,50,1,30)
"""
ox=['SiO2','Al2O3','MgO','CaO','Na2O','K2O','SO3']
oxides=['SiO2','MgO','Na2O','Al2O3','CaO']

"""df=pd.read_csv('ym.csv')
compare_slicings(df,ox,25,1,30)"""

#preparator('ym','Young',ox,nmin=15,alpha=0.1,nb_slices=12,add_to_name='o')
