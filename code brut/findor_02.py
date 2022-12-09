

def get_domain(class_id):
    classes,depth=class_id
    domain=[]
    for id in classes:
        domain.append((id/depth,(id+1)/(depth+1)))
    return tuple(domain)  

def domains(id_list):
    doms=[]
    for class_id in id_list:
        doms.append(get_domain(class_id))
    doms.sort()
    return doms

def is_in_domains(compo,domains):
    for domain in domains:
        for i in range(len(compo)):
            if not domain[i][0]<=compo[i]<=domain[i][1]:
                break
        if i==len(compo)-1:
            return True
    return False



#exemple
'''f=open('density_table_tableonly_data_clean_prepareddomains.txt','r')
id_list=eval(f.readline())
doms=domains(id_list)
print(doms[0])
print(is_in_domains((0.3,0.3,0.3,0.1,0.),doms))
f=open('densityodomains.txt','r')
id_list=eval(f.readline())
doms=domains(id_list)'''

