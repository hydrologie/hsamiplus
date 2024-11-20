from __future__ import annotations

import numpy as np

def hsami_hydrogramme(mode,forme,pas_temps_par_jour,memoire):
    '''
     function h = hsami_hydrogramme(mode,forme,pas_temps_par_jour,memoire)
     Calcule les valeurs d'un hydrogramme
     qui pointe aprés "mode" jours,
     suivant une loi béta de paramétre de forme nommé "forme"
     et tronqué aprés "memoire" jours
     ===========================================================================
    '''
    n = int(memoire * pas_temps_par_jour)
    if isinstance(mode, np.ndarray):        
        t = np.tile(np.arange(n+1), (mode.shape[1], mode.shape[0], 1))
    else:
        t = np.arange(1,n+1)
        
    mode = np.tile(mode, (n, 1)).transpose(1, 0)
    forme = np.tile(forme, (n, 1)).transpose(1, 0)
    h = t ** (mode * forme) * np.exp(-forme / pas_temps_par_jour * t)
    h = h / np.tile(np.sum(h, axis=1), n)

    return h
