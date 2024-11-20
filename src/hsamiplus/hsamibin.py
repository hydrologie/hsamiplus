
# Le modèle HSAMI a été originalement développé par J.L. Bisson, et F. Roberge 
# en Matlab, 1983. HSAMI a été modifié et bonifié par Catherine Guay, Marie Minville,
# Isabelle Chartier et Jonathan Roy, 2013-2017 pour devenir HSAMI+. Le code a été
# traduit en Python par Didier Haguma, 2024.
 

from __future__ import annotations

import os
import json
import datetime
from hsami2 import hsami2


def hsamibin(path,filename):
    '''
    Fonction qui lit un projet HSAMI+ en format JSON, exécute HSAMI+, et
    sauvegarde les sorties d'HSAMI+ en format JSON dans le méme répertoire
    que le projet. La fonction peut étre compilée avec le makefile disponible
    dans le répertoire. 
    
    
    INPUT
     -----
    - path         Emplacement du fichier de projet 
                   ex.: ./DATA
    - filename     Nom du fichier projet
                   ex.: projet.json
    
    OUTPUT
    ------
    Fichier output_<date>.json
    
    
    '''

    # Load json files and convert to Python format
    with open(os.path.join(path, filename), 'r') as file:
        projet = json.load(file)

    # Execute hsami2
    date = datetime.date.today()
    
    S, etats, deltas = hsami2(projet)

    # Write output file
    output = {
        'S': S,
        'etats': etats,
        'deltas': deltas
    }
    output_json = json.dumps(output)
    
    with open(os.path.join(path, 'output_' + date.strftime('%d_%m_%Y') + '.json'), 'w') as file:
        file.write(output_json)
        
    return S, etats, deltas
            

if __name__ == "__main__":
    
    #import sys
    import time
    start_time = time.time()
    
    path = r'../../Data'
    filename = 'projet.json'
    #filename = 'input_bassin1045_20241029.json'
          
    S, etats, deltas = hsamibin(path, filename)
    
    print("Fin, après {:.2f} secondes !!! ".format(time.time() - start_time))
