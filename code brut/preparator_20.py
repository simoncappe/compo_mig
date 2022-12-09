import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def classify(x,slices):
    limits=np.array([(i+1)/slices for i in range(slices)])
    a,b=0,len(limits)-1
    while (b-a)>1:
        c=(b+a)//2
        if limits[c]<x: a=c
        else: b=c
    return b

def data_sliceor(G,data,oxides,nmin,gap,slices,father):
    initial_cols=data.columns
    data['group']=tuple(map(tuple,data[oxides].apply(np.vectorize(classify),axis=1,args=(slices,))))
    groups=data['group'].unique()
    grby=data.groupby('group',sort=False)
    size_select=grby.size()>nmin
    std_select=(1-(slices/100)*grby[oxides].std(ddof=0).mean(axis=1))<gap    
    if np.any(size_select & std_select):
        leaves=groups[size_select & std_select]
        for son in leaves:
            G.add_node((son,slices),data_node=grby.get_group(son)[initial_cols])
            G.add_edge(father, (son,slices))
    mask=size_select & ~std_select
    if np.any(mask):
        to_continue=groups[mask]
        for next_one in to_continue:
            data_sliceor(G,grby.get_group(next_one)[initial_cols],oxides,nmin,gap,slices+1,father)

def data_cleanor(G,nodes,property,alpha):
    for node in nodes:
        df=G.nodes[node]['data_node'] ; initial_cols=df.columns
        maxi=df[property].max() ; mini=df[property].min()
        try:
            df['auxprop']=(df[property]-mini)/(maxi-mini)
        except ZeroDivisionError:
            print('propriete constante, normalisation max-min impossible')
        scale=df['auxprop'].std(ddof=0) ;  slices=int(1/scale)+1
        df['group']=df['auxprop'].apply(classify,args=(slices,))
        grby=df.groupby('group',sort=False)
        groups=pd.DataFrame({'group':df['group'].unique()})
        groups['size']=list(grby.size())
        groups.sort_values(by='size',axis=0,ascending=False,inplace=True)
        ref,sizemax=groups.iloc[0]
        groups.sort_values(by='group',axis=0,ascending=True,inplace=True)
        ref_index=(groups['group']<ref).sum()
        keepmin=ref
        keepmax=ref
        for i0 in range(ref_index,-1,-1):
            if ref-groups['group'].iloc[i0]==ref_index-i0 and groups['size'].iloc[i0]/sizemax>=alpha:
                keepmin=groups['group'].iloc[i0]
            else:
                break
        for i1 in range(ref_index+1,len(groups)):
            if (groups['group'].iloc[i1])-ref==i1-ref_index and groups['size'].iloc[i1]/sizemax>=alpha:
                keepmax=groups['group'].iloc[i1]
            else:
                break
        select=df[(df['group']>=keepmin) & (df['group']<=keepmax)][initial_cols]
        G.nodes[node]['data_node']=select

def preparator(file_name,property,oxides,nmin=100,gap=0.6,alpha=0.1):
    df=pd.read_csv(file_name+'.csv')
    G=nx.DiGraph()
    G.add_node('origin')
    data_sliceor(G,df,oxides,nmin,gap,1,'origin')
    nodes=list(G.nodes())[1:]
    if nodes!=[]:
        data_cleanor(G,nodes,property,alpha)
    else:
        print('donnees non conformes aux exigences')
    prepared=pd.concat(nx.get_node_attributes(G,'data_node').values(),ignore_index=True, sort=False)
    prepared.to_csv(file_name+'_prepared3.csv')


#exemple d utilisation
oxides=['SiO2','MgO','Na2O','Al2O3','CaO']
preparator('density_table_tableonly_data_clean','Density',oxides,nmin=50,gap=0.6,alpha=0.1)
