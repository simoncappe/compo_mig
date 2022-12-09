import numpy as np

def get_domain(class_id,nb_slices=10):
    classes=class_id
    domain=[]
    for id in classes:
        limits=np.arange(start=100/nb_slices,stop=100+100/nb_slices,step=100/nb_slices)
        domain.append((limits[id]-100/nb_slices,limits[id]))
    return tuple(domain)  

def domains(id_list):
    doms=[]
    for class_id in id_list:
        doms.append(get_domain(class_id))
    return doms

def is_in_domains(compo,domains):
    for domain in domains:
        for i in range(len(compo)):
            if not domain[i][1]<=compo[i]<=domain[i][1]:
                break
        if i==len(compo):
            return True
    return False
ox=['SiO2','Al2O3','MgO','CaO',     
'Na2O','K2O','SO3']       

#exemple
'''ox=['SiO2','Al2O3','MgO','CaO',     
'Na2O','K2O','SO3']       

f=open('ym_prepareddomains_nbslices=10.npy', 'rb')
id_list=np.load(f,allow_pickle=True)
doms=domains(id_list)''' 
#doms est une liste
#de domaines de validités. 
# Chaque élément de doms
# est de la forme
#((minSiO2,maxSiO2),(minAl2O3,maxAl2O3),(minMg0,maxMg0),...)

