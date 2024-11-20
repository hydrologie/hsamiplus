from __future__ import annotations

import numpy as np
from copy import copy
from hsami2_noyau import hsami2_noyau

def hsami2(projet):
    '''
    ============================================
     Champs obligatoires dans projet
     -------------------------------
     projet :  projet['superficie']
               projet['param'] 
               projet['memoire']
               projet['physio'] (peut être vide)
               projet['modules'] (peut être vide)
               projet['meteo']
               projet['dates']
               projet['nb_pas_par_jour']
    ============================================
    '''

    # Extraction de variables de la structure projet
    # ----------------------------------------------

    superficie = projet['superficie']
    if len(superficie) == 1:
        superficie.append(0)
    
    param = projet['param']
    physio = projet['physio']

    # Valeurs par défaut dans modules
    # -----------------------------------

    modules = projet['modules']
    
    if 'etp_bassin' not in modules:
        modules['etp_bassin'] = 'hsami'
    
    if 'etp_reservoir' not in modules:
        modules['etp_reservoir'] = 'hsami'
    
    if 'een' not in modules:
        modules['een'] = 'hsami'
    
    if 'infiltration' not in modules:
        modules['infiltration'] = 'hsami'
    
    if 'qbase' not in modules:
        modules['qbase'] = 'hsami'
    
    if 'sol' not in modules:
        modules['sol'] = 'hsami'
    
    if 'radiation' not in modules:
        modules['radiation'] = 'hsami'
    
    if 'reservoir' not in modules:
        modules['reservoir'] = 0
    
    if 'mhumide' not in modules:
        modules['mhumide'] = 0
    
    if 'glace_reservoir' not in modules:
        modules['glace_reservoir'] = 0

    # ------------------------
    # Initialisation des etats
    # ------------------------

    # Dictionnaire états entrants
    # ------------------------
    etat = {}
    
    etat['eau_hydrogrammes'] = np.zeros((int(projet['memoire']), 3))
    
    if modules['een'] in ['mdj', 'alt']:
        if modules['een'] == 'mdj':
            n = len(physio['occupation'])
        if modules['een'] == 'alt':
            n = len(physio['occupation_bande'])
            
        etat['modules'] = {}
        
        etat[modules['een']] = {
            'couvert_neige': [0]*n,
            'densite_neige': [0]*n,
            'albedo_neige': [0.9]*n,
            'neige_au_sol': [0]*n,
            'fonte': [0]*n,
            'gel': [0]*n,
            'sol': [0]*n,
            'energie_neige': [0]*n,
            'energie_glace': 0
        }
    
    etat['neige_au_sol'] = 0
    etat['fonte'] = 0
    etat['nas_tot'] = 0
    etat['fonte_tot'] = 0
    etat['derniere_neige'] = 0
    etat['gel'] = 0
    etat['nappe'] = param[13]
    etat['reserve'] = 0
    
    if modules['sol'] == 'hsami':
        # Initialisation du sol à sol_min.
        etat['sol'] = np.array([param[11], np.nan])
    elif modules['sol'] == '3couches':
        # Initialisation du sol à la capacité au champ.
        etat['sol'] = np.array([param[42] * param[39], param[43] * param[40]])
    
    if modules['mhumide'] == 1:
        if physio['samax'] == 0:
            raise ValueError('La superficie maximale du milieu humide équivalent est égale à 0.')
        
        etat['mh_surf'] = param[48] * physio['samax'] * 100                       # On considère la surface initiale égale à la surface normale (en hectars)
        etat['mh_vol'] = param[48] * (param[47] * physio['samax'] * 100 * 10000)  # On considère le volume initial au volume normal (en m^3)
        etat['ratio_MH'] = etat['mh_surf'] / (superficie[0] * 100)
    
    if modules['mhumide'] == 0:
        etat['mh_vol'] = 0
        etat['ratio_MH'] = 0
        etat['mh_surf'] = 1
    
    etat['mhumide'] = etat['mh_vol'] * etat['ratio_MH'] / (etat['mh_surf'] * 100)
    etat['ratio_qbase'] = 0

    # Glace/réservoir
    etat['cumdegGel'] = 0
    etat['obj_gel'] = -200
    etat['dernier_gel'] = 0
    etat['reservoir_epaisseur_glace'] = 0
    etat['reservoir_energie_glace'] = 0
    etat['reservoir_superficie'] = superficie[1]
    etat['reservoir_superficie_glace'] = 0
    etat['reservoir_superficie_ref'] = etat['reservoir_superficie']
    etat['eeg'] = np.zeros(5000)
    etat['ratio_bassin'] = 1
    etat['ratio_reservoir'] = 0
    etat['ratio_fixe'] = 1

    # Structure états sortants
    # ------------------------

    nb_pas_total = len(projet['meteo']['bassin'])
    
    etats = {}

    f = list(etat.keys())
    
    for i_f in range(len(f)):
        
        etats[f[i_f]] = []
        

    # ----------------------
    # Structure des sorties
    # ----------------------

    S = {'Qtotal': [], 'Qbase': [], 'Qinter': [], 'Qsurf': [], 'Qreservoir': [], 'Qglace': [], 
         'ETP': [],'ETRtotal': [], 'ETRsublim': [], 'ETRPsurN': [], 'ETRintercept': [],
         'ETRtranspir':[],'ETRreservoir': [], 'ETRmhumide': [], 'Qmh': []}
    
    deltas = {'total': [], 'glace': [], 'interception': [], 'ruissellement': [], 'vertical': [],
              'mhumide': [], 'horizontal': []}

    # ----------------------
    # Tour de chauffe (1 an)
    # ----------------------

    pas = 1
    for i_pas in range(365):

        # Construction du projet pour hsami_noyau
        p = {}

        if 'hu_surface' in projet:
            p['hu_surface'] = projet['hu_surface']
        if 'hu_inter' in projet:
            p['hu_inter'] = projet['hu_inter']

        p['date'] = projet['dates'][i_pas]
        p['nb_pas_par_jour'] = projet['nb_pas_par_jour']
        p['superficie'] = superficie
        p['memoire'] = projet['memoire']
        p['param'] = param
        p['meteo'] = {'bassin': projet['meteo']['bassin'][i_pas],
                      'reservoir': projet['meteo']['reservoir'][i_pas]}
        p['modules'] = modules
        p['physio'] = copy(physio)
        p['pas'] = pas
        if 'niveau' in physio.keys():
            p['physio']['niveau'] = physio['niveau'][i_pas]

        # Simulation
        _, etat, _ = hsami2_noyau(p, etat)

        # On avance d'un pas de temps
        if pas == projet['nb_pas_par_jour']:
            pas = 1
        else:
            pas = pas + 1

    # ----------
    # Simulation
    # ----------

    pas = 1
    for i_pas in range(nb_pas_total):

        # Construction du projet pour hsami_noyau
        p = {}

        if 'hu_surface' in projet:
            p['hu_surface'] = projet['hu_surface']
        if 'hu_inter' in projet:
            p['hu_inter'] = projet['hu_inter']

        p['date'] = projet['dates'][i_pas]
        p['nb_pas_par_jour'] = projet['nb_pas_par_jour']
        p['superficie'] = superficie
        p['memoire'] = projet['memoire']
        p['param'] = param
        p['meteo'] = {'bassin': projet['meteo']['bassin'][i_pas],
                      'reservoir': projet['meteo']['reservoir'][i_pas]}
        p['modules'] = modules
        p['physio'] = copy(physio)
        if 'niveau' in physio.keys():
            p['physio']['niveau'] = physio['niveau'][i_pas]
        p['pas'] = pas

        # Simulation
        s, etat, delta = hsami2_noyau(p, etat)

        # Sauvegarde des sorties
        f = list(s.keys())
        
        for i_f in range(len(f)):
            S[f[i_f]].append(s[f[i_f]])
       
        # Sauvegarde des états
        f = list(etat.keys())
        for i_f in range(len(f)):
            if isinstance(etat[f[i_f]], np.ndarray): 
                if f[i_f] == 'eeg':
                    etats[f[i_f]].append(np.nansum(etat[f[i_f]]).tolist())
                else:
                    etats[f[i_f]].append(etat[f[i_f]].tolist())
            else:
                etats[f[i_f]].append(etat[f[i_f]])
                

        # Sauvegarde du bilan de masse
        f = list(delta.keys())
        for i_f in range(len(f)):
            deltas[f[i_f]].append(delta[f[i_f]])

        # On avance d'un pas de temps
        if pas == projet['nb_pas_par_jour']:
            pas = 1
        else:
            pas = pas + 1

    return S, etats, deltas

