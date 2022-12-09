Pour faire des prédictions à partir d'un fichier Interglad:
-Tout d'abord, le nettoyer (convertir les str, enlever ceux dont la somme ne fait pas 1,...). Utiliser 'clean_file.py', avec la fonction do_all (cf exemple d'utilisation)
-Ensuite, le préparer (l'histoire de découpage en cube,...). Utiliser 'preparator.py' (cf exemple d'utilisation)
-Il en résulte un fichier .csv final contenant mes données et un .npy qui contient les domaines de validité 
-Au choix, faire des régressions linéaires ('prediction_lin.py') ou du machine learning ('prediction_mlp.py'). Les fonctions que j'ai nommé 'f' dans chacun des ces fichiers fait office de fonction de prédiction.
-Si on veut, on peut aller jeter un oeil aux domaines de validités dans 'doms.py'

Commentaires:
-Les traitements effectués créent plusieurs fichiers automatiquement dans l'ordinateur. Attention a ne pas se perdre dans tout cela. Ne pas hésiter à effacer ceux qui ont été créé si on veut refaire un essai
-Ce programme n'est pas parfait, il se peut preparator ou do_all réduise beaucoup trop le nombre de données.Un temps de tests des différents paramètres est nécéssaire avant d'obtenir un fichier satisfaisant. 