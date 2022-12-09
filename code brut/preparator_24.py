import numpy as np ; import pandas as pd

def classify(limits):
    def aux_classify(x):
        #classe x dans la classe de borne sup limits[b]
        a,b=0,len(limits)-1
        while (b-a)>1:
            c=(b+a)//2
            if limits[c]<x: a=c
            else: b=c
        return b
    return aux_classify

def data_sliceor(acc_data,acc_cubes,data,oxides,nmin,gap,limits):
    initial_cols=data.columns
    #colonne 'cube' : tuples identifiant chaque subdivision spatiale
    data['cube']=tuple(map(tuple,data[oxides].apply(np.vectorize(classify(limits)),axis=1)))
    cubes=data['cube'].unique()
    grby=data.groupby('cube',sort=False)
    size_select=grby.size()>nmin
    std_select=(1-(len(limits)/100)*grby[oxides].std(ddof=0).mean(axis=1))<gap    
    #conserve tout cube avec distribution spatiale de donnees conforme
    if np.any(size_select & std_select):
        for new in cubes[size_select & std_select]:
            acc_cubes.append((new,len(limits)))
            acc_data[(new,len(limits))]=grby.get_group(new)[initial_cols]
    select=size_select & ~std_select
    #relance recherche cubes de donnees plus petits sur donnees possibles 
    if np.any(select):
        for next_one in cubes[select]:
            data_sliceor(acc_data,acc_cubes,grby.get_group(next_one)[initial_cols],\
                oxides,nmin,gap,np.array([(i+1)/(len(limits)+1) for i in range(len(limits)+1)]))

def data_cleanor(acc_data,acc_cubes,property,alpha):
    for cube in acc_cubes:
        df=acc_data[cube] ; initial_cols=df.columns
        maxi=df[property].max() ; mini=df[property].min()
        try:
            df['auxprop']=(df[property]-mini)/(maxi-mini)
        except ZeroDivisionError:
            print('propriete constante, normalisation max-min impossible')
        scale=df['auxprop'].std(ddof=0) ;  slices=int(1/scale)+1
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
        select=df[(df['property classes']>=keepmin) & (df['property classes']<=keepmax)][initial_cols]
        acc_data[cube]=select

def preparator(file_name,property,oxides,nmin=100,gap=0.6,alpha=0.1,add_to_name='_prepared',domains=False):
    df=pd.read_csv(file_name+'.csv')
    acc_data={} ;  acc_cubes=[]
    limits=np.array([(i+1)/2 for i in range(2)])
    #lance decoupage
    data_sliceor(acc_data,acc_cubes,df,oxides,nmin,gap,limits)
    if acc_cubes!=[]:
        data_cleanor(acc_data,acc_cubes,property,alpha)
    else:
        print('donnees non conformes aux exigences')
    #fusionne les donnees nettoyees
    prepared=pd.concat(acc_data.values(),ignore_index=True, sort=False)
    prepared.to_csv(file_name+add_to_name+'.csv')
    #enregistre si besoin les domaines regroupant des donnees selectionees
    if domains:
        toxides=tuple(oxides)
        f=open(file_name+add_to_name+'domains.txt','a')
        f.write(str(acc_cubes))
        f.close()

"""
#exemple d utilisation
oxides=['SiO2','MgO','Na2O','Al2O3','CaO']
preparator('density_table_tableonly_data_clean','Density',oxides,nmin=100,gap=0.6,alpha=0.1,domains=True)
"""