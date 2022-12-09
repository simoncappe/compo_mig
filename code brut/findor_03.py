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

#exemple
'''oxides=['SiO2','MgO','Na2O',
'Al2O3','CaO']
f=open('densityodomains_nbslices=10.npy', 'rb')
id_list=np.load(f,allow_pickle=True)
doms=domains(id_list)
print(doms[3])'''

