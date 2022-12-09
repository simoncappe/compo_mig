from preparator_33 import preparator,compare_slicings
import pandas as pd
ox=['SiO2','Al2O3','MgO','CaO','Na2O','K2O','SO3']
df=pd.read_csv('tenace.csv')
#compare_slicings(df,ox,nmin=30,n1=1,n2=10)
preparator('tenace','Fracture Toughness',ox,nmin=50,alpha=0.1,add_to_name='o',nb_slices=6)
