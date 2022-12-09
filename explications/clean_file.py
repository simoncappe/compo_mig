import numpy as np ; import matplotlib.pyplot as plt ; import pandas as pd ; import re

##outils traitement fichiers csv d INTERGLAD

def extract_table(file_name,oxides,property):
    """Description:
    Enleve preambule genant pour ne conserver que la table, puis enregistre la table extraite.

    Arguments:
    - file_name (str): nom du fichier SANS le ".csv". Le fichier final est renregistre en ajoutant au nom "_tableonly"
    - oxides (str list): liste des noms des oxydes  (ex : ['SiO2','MgO']).
    - property (str): nom de la propriete etudiee (ex : 'Density').

    Working conditions:
    - les str en argument apparaissent dans les noms des colonnes ciblees.
    - property cible la colonne dont le nom admet la premiere occurence de property.
    - le nom du fichier final n est pas deja utilise."""
    file=open(file_name+'.csv','r')
    b=1;i=0
    while b:
        i+=1
        line=file.readline()
        regex="|".join(oxides)
        if re.search(regex,line) and re.search(property,line):
            aux=open(file_name+'_tableonly.csv','a')
            aux.write(line) ; aux.write(file.read())
            aux.close()
            b=0
    file.close()

def select_data(file_name,oxides,property):
    """Selectionne les colonnes oxides et celle de property.
    Les noms des colonnes initiales peuvent ne pas parfaitement correspondre aux arguments : elles sont donc renommees selon ces arguments.
    Enregistre la sous-table selectionnee.

    Arguments:
    - file_name (str): nom du fichier SANS le ".csv". La table finale est renregistree en ajoutant au nom "_data".
    - oxides (str list): liste des noms des oxydes  (ex : ['SiO2','MgO']).
    - property (str): nom de la propriete etudiee (ex : 'Density').

    Working conditions:
    - les str en argument apparaissent dans les noms des colonnes ciblees.
    - le nom du fichier final n est pas deja utilise."""
    data=pd.read_csv(file_name+'.csv')
    rename_dict={}
    for col in data.columns:
        for ox in oxides:
            if re.search(ox,col):
                rename_dict[col]=ox
        if re.search(property,col):
            rename_dict[col]=property
            break
    data=data.rename(columns=rename_dict)
    data=data[oxides+[property]]
    data.to_csv(file_name+'_data.csv',index=False)

def str_to_float(x):
    """Convertit un str en float.
    Si le str contient un chiffre, on suppose que c est deja un float et on laisse python faire la conversion, que l on renvoie.
    Sinon, on renvoie un zero."""
    if type(x)==str:
        if not re.search('\d',x):
            return 0.
        else:
            return float(x)
    else:
        return x

def cleanup_data(file_name,error,ref=100):
    """Nettoie la table importee et l enregistre :
    - enleve les doublons.
    - convertit les str en float avec str_to_float.
    - conserve uniquement les verres dont la somme des pourcentages des composants vaut 100.

    Arguments:
    - file_name (str): nom du fichier SANS le ".csv". La table finale est renregistree en ajoutant au nom "_clean".
    - ref (float): dans le cas des compositions en pourcentages vaut par defaut 100.
    - error (float): ecart a ref admissible.

    Working conditions:
    - la derniere colonne correspond a la propriete etudiee.
    - le nom du fichier final n est pas deja utilise."""
    clean_data=pd.read_csv(file_name+'.csv')
    clean_data=clean_data.drop_duplicates()
    clean_data=clean_data.apply(np.vectorize(str_to_float),axis=0,raw=True)
    clean_data=clean_data[abs(np.sum(clean_data[clean_data.columns[:-1]],axis=1)-ref)<error]
    clean_data.to_csv(file_name+'_clean.csv',index=False)

def do_all(file_name,oxides,property):
    """Execute extract_table, puis select_data, puis cleanup_data"""
    extract_table(file_name,oxides,property)
    select_data(file_name+'_tableonly',oxides,property)
    cleanup_data(file_name+'_tableonly_data',5,100)

"""#exemple d utilisation
oxides=['SiO2','MgO','Na2O','Al2O3','CaO']
do_all('fracture_toughness_100',oxides,'Toughness')
"""
