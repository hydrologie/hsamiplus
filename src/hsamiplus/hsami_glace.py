from __future__ import annotations

import numpy as np


def hsami_glace(modules, superficie, etats, *varargin):
    
    '''
     ENTRÉES
     -------
     modules:        Voir hsami2_noyau (projet['modules'])
     superficie:     Voir hsami2_noyau (projet['superficie'])
     varargin{1}:    meteo, voir hsami2_noyau (projet['meteo'])
     varargin{2}:    physio, voir hsami2_noyau (projet['physio'])
     varargin{3}:    param, hsami2_noyau (projet['param'])
    
     SORTIES
     -------
     glace_vers_reservoir  : Lame d'eau transitant de la glace de rive vers le
                             réservoir pour le pas de temps (cm)
     bassin_vers_reservoir : Lame d'eau transitant du réservoir vers la glace 
                             de rive pour le pas de temps (cm)                       
     etats                 : Voir hsami2_noyau (etat)
     
     EXEMPLE
     -------
     test_glace
     Assigner modules['reservoir'] = 1
     Tester les deux valeurs possibles pour modules['glace_reservoir']. Pour
     tester l'option 'mylake', utiliser modules['een'] = 'mdj'.
     Tous les autres modules sont PAR DÉFAUT.
    
    '''

    # Densité de la glace normale à 0 deg et pression atmosphérique.
    densite_glace = 0.916

    if modules['reservoir'] == 1:
        
        if len(varargin) > 2:
            
            meteo = varargin[0]
            physio = varargin[1]
            param = varargin[2]
            if modules['glace_reservoir'] == 'stefan':
                k = param[46]
                superficie_glace, superficie_reservoir, etats = Stefan(meteo['reservoir'][:2], k, physio, etats)
                # Ex. : superficie_glace = [0, 0]
                #       superficie_reservoir = [339.1769, 339.6125]
                #       etats.reservoir_superficie_glace = 0
                #       etats.reservoir_epaisseur_glace = 0.1272
                #       etats.dernier_gel = 0
                #       etats.reservoir_superficie_ref = 338.7413
                #       etats.obj_gel = -200
                #       etats.cumdegGel = -530.2250
                
            elif modules['glace_reservoir'] == 'mylake':
                if modules['een'] == 'mdj' or modules['een'] == 'alt':
                    superficie_glace, superficie_reservoir, etats = MyLake(meteo['reservoir'], physio, etats, param, modules)
                    # Ex. : superficie_glace = [99, 98]
                    #       superficie_reservoir = [339.1769, 339.6125]
                    #       etat.reservoir_superficie = 339.6125
                    #       etat.reservoir_superficie_glace = 98
                    #       etat.reservoir_epaisseur_glace = 0.7488
                    #       etat.reservoir_superficie_ref = 438
                    #       etat.reservoir_energie_glace = -1.7324E07
                    
                else:
                    raise ValueError("Le module 'mylake' pour la glace de réservoir doit être utilisé obligatoirement avec le module 'mdj' ou 'alt' pour la neige")
            else:
                raise ValueError("modules.glace_reservoir doit être 'stefan' ou 'mylake'")
            
            # Épaisseur de la glace flottante en cm
            etats['reservoir_epaisseur_glace'] = etats['reservoir_epaisseur_glace'] * 100
            
            # Fraction du bassin versant occupée par le réservoir
            etats['ratio_reservoir'] = superficie_reservoir[1] / superficie[0]
            etats['ratio_bassin'] = 1 - etats['ratio_reservoir']
            etats['ratio_fixe'] = 1 - (superficie[1] / superficie[0])
            
            # Variations des superficies de berges et de réservoir
            delta_glace = superficie_glace[1] - superficie_glace[0]
            delta_reservoir = (superficie_reservoir[1] - superficie_reservoir[0]) / superficie[0]
            
        else:
            # On considère la superficie du réservoir fixe
            etats['reservoir_epaisseur_glace'] = 0
            etats['reservoir_superficie_glace'] = 0
            etats['ratio_reservoir'] = superficie[1] / superficie[0]
            etats['ratio_bassin'] = 1 - etats['ratio_reservoir']
            etats['ratio_fixe'] = 1 - (superficie[1] / superficie[0])
            delta_glace = 0
            delta_reservoir = 0
            
    else:
        # Cas où on ne considère pas de réservoir
        etats['reservoir_epaisseur_glace'] = 0
        etats['reservoir_superficie_glace'] = 0
        etats['ratio_reservoir'] = 0
        etats['ratio_bassin'] = 1
        etats['ratio_fixe'] = 1
        delta_glace = 0
        delta_reservoir = 0
        
    # Dépôt de glace flottante et restitution par ennoiement
    if delta_glace > 0: # Il y a dépôt
        
        ind1 = int(superficie_glace[0]) + 1
        ind2 = int(superficie_glace[1])
        etats['eeg'][ind1:ind2] = etats['reservoir_epaisseur_glace'] * densite_glace
        glace_vers_reservoir = -np.sum(etats['eeg'][ind1:ind2])
        
    elif delta_glace < 0: # Il y a restitution
        
        ind1 = int(superficie_glace[1]) + 1
        ind2 = int(superficie_glace[0])
        glace_vers_reservoir = np.sum(etats['eeg'][ind1:ind2])
        etats['eeg'][ind1:ind2] = 0
        
    else:
        glace_vers_reservoir = 0
        
    # Retrait ou restitution de neige dû à la fluctuation du réservoir
    # - : retrait     (reservoir -> bassin)
    # + : restitution (bassin -> réservoir)
    change = delta_reservoir * etats['neige_au_sol']
    bassin_vers_reservoir = change

    return glace_vers_reservoir, bassin_vers_reservoir, etats



# --------------------
# Fonctions de soutien
# --------------------

def Stefan(meteo, k, physio, etats):
    """
    Calcule la superficie de glace et la superficie du réservoir en fonction des conditions météorologiques, 
    des paramètres physiologiques et des états précédents.

    Entreées :
        meteo (array-like): Les conditions météorologiques.
        k (float): Le coefficient de conversion des degrés-jours en épaisseur de glace.
        physio (dict): Les paramètres physiologiques.
        etats (dict): Les états précédents.

    Sorties :
        tuple: Un tuple contenant la superficie de glace, la superficie du réservoir et les nouveaux états.
    
    """

    # Entrées
    niveau = physio['niveau']
    coeff = physio['coeff']

    # Variables locales
    HIVERGLACIO = -200 # Début de l'hiver glaciologique aprés HIVERGLACIO degrés-jours négatifs
    NBJ = 21           # Nombre de jours de stagnation pour declencher le dégel

    # Calcul des degrés jours de gel
    moyenneGel = np.mean([meteo[0], meteo[1]/2], axis=0)
    moyenneGel = 0 if moyenneGel >= 0 else moyenneGel #moyenneGel[moyenneGel>=0] = 0
    cumdegGel = etats['cumdegGel'] + moyenneGel

    # Récupération des variables d'états du pas de temps précédent
    superficie_reservoir = [0,0]
    superficie_glace = [0,0]
    tot_epaisseur = [0,0]
    superficie_reservoir[0] = etats['reservoir_superficie']
    superficie_glace[0] = etats['reservoir_superficie_glace']
    tot_epaisseur[0] = etats['reservoir_epaisseur_glace']
    dernier_gel = etats['dernier_gel']
    supref = etats['reservoir_superficie_ref']
    obj_gel = etats['obj_gel']

    # Calcul de la superficie du réservoir
    if np.isnan(niveau):
        superficie_reservoir[1] = superficie_reservoir[0]
    else:
        superficie_reservoir[1] = coeff[0]*niveau**2 + coeff[1]*niveau + coeff[2]

    # Si le cumul deg-j atteind l'objectif de gel
    if cumdegGel < obj_gel:
        
        # Calcul de l'épaississement de la glace en cm
        tot_epaisseur[1] = (k*np.abs(cumdegGel-obj_gel)**(1/2))/100
        # Ex.: tot_epaisseur_2 = 0.1272
        
        # Si c'est le premier cas de l'hiver
        if tot_epaisseur[0] == 0:
            # On fixe les références
            supref = superficie_reservoir[0]
        
        # Vérification de sortie de la période de gel
        if moyenneGel == 0:
            dernier_gel = dernier_gel + 1
        else:
            dernier_gel = 0
            
        if dernier_gel >= NBJ:
            # On fixe le prochain objectif deg-jour à atteindre
            # pour démarrer la prochain période de gel
            obj_gel = HIVERGLACIO + cumdegGel
        
        # Calcul de la superficie du dépôt de glace. Si le niveau du
        # réservoir remonte au-delà du niveau de référence, il n'y a plus
        # de glace déposée
        superficie_glace[1] = max(0, supref - superficie_reservoir[1])
        
    else:
        superficie_glace[1] = 0
        tot_epaisseur[1] = 0

    # Arrondi pour la discrétisation spatiale par km2
    superficie_glace[1] = np.round(superficie_glace[1])

    # Sauvegarde des états
    etats['reservoir_superficie'] = superficie_reservoir[1]
    etats['reservoir_superficie_glace'] = superficie_glace[1]
    etats['reservoir_epaisseur_glace'] = tot_epaisseur[1]
    etats['dernier_gel'] = dernier_gel
    etats['reservoir_superficie_ref'] = supref
    etats['obj_gel'] = obj_gel
    etats['cumdegGel'] = cumdegGel

    return superficie_glace, superficie_reservoir, etats



def MyLake(meteo, physio, etat, param, modules):
    """
    Fonction qui simule le comportement d'un lac gelé en utilisant le modèle MYLake.

    Entreées :
    - meteo : dict
        Dictionnaire contenant les données météorologiques.
    - physio : dict
        Dictionnaire contenant les paramètres physiques du lac.
    - etat : dict
        Dictionnaire contenant les états du lac à un pas de temps précédent.
    - param : dict
        Dictionnaire contenant les paramètres du modèle.
    - modules : dict
        Dictionnaire contenant les modules activés dans le modèle.

    Sorties :
    - superficie_glace : float
        Superficie de la glace sur le lac en km2.
    - superficie_reservoir : float
        Superficie totale du lac en km2.
    - etat : dict
        Dictionnaire contenant les nouveaux états du lac.

    Remarque :
    Cette fonction est basée sur le modèle MYLake (Saloranta et Andersen, 2004) 
    et le modèle mixte degrés-jour implanté.
    """
    
    # Constante physiques
    k_i    = 2.24      # Conductivité thermique de la glace, W/m-degC
    rho_i  = 916       # Densité de la glace, kg/m3
    rho_w  = 1000      # Densité de l'eau, kg/m3
    Lf     = 3.34e5    # Chaleur latente de fusion, J/kg
    c_i    = 2093.4    # Chaleur spécifique de l'eau solide à 0degC (J/(kg*degC))
    c_w    = 4216      # Chaleur spécifique de l'eau liquide à 0 degC (J/(kg*degC))

    # Courbe d'emmagasinnement
    niveau = physio['niveau']
    coeff = physio['coeff']

    # Récupération des variables d'états du pas de temps précédent
    superficie_reservoir = [0,0]
    superficie_glace = [0,0]
    epaisseur_glace = [0,0]
    superficie_reservoir[0] = etat['reservoir_superficie']         # km2
    superficie_glace[0] = etat['reservoir_superficie_glace']       # km2
    epaisseur_glace[0] = etat['reservoir_epaisseur_glace']/100     # m
    supref = etat['reservoir_superficie_ref']                      # km2
    couvert = etat[modules['een']]['couvert_neige'][-1]            # m
    dennei = etat[modules['een']]['densite_neige'][-1]             # fraction

    # Calcul de la superficie du réservoir
    if np.isnan(niveau):
        superficie_reservoir[1] = superficie_reservoir[0]
    else:
        superficie_reservoir[1] = coeff[0]*niveau**2 + coeff[1]*niveau + coeff[2]

    # Si c'est le premier cas de l'hiver
    if epaisseur_glace[0] == 0:
        # On fixe les références
        supref = superficie_reservoir[0]

    # Calcul de la température moyenne de l'air
    Ta = np.mean([meteo[0], meteo[1]/2])

    if Ta <= 0:
        
        if epaisseur_glace[0] > 0:
            
            # Estimation de la température de la glace
            if couvert > 0:
                k_s = ConductiviteNeige(dennei*rho_w)
                p = k_i*couvert/(k_s*epaisseur_glace[0])
            else:
                p = 1/(10*epaisseur_glace[0])
            Ti = Ta/(1+p)
            
        else:
            
            Ti = Ta
            
        # Calcul de la croissance thermique de la glace flottante en
        # fonction de sa température
        epaisseur_glace[1] = np.sqrt(epaisseur_glace[0]**2 + (2*k_i*86400/(rho_i*Lf))*(-Ti))
        # Ex. : epaisseur_glace[1] = 0.7488
        
        if not np.isreal(epaisseur_glace[1]) or epaisseur_glace[1] == 0:
            # Si le terme sous la racine carrée est négatif,on obtient un nombre
            # complexe. Cette situation correspond à la disparition de la glace.
            epaisseur_glace[1] = 0
            energie = 0
            
        else:
            # Mise à jour de l'énergie contenue dans la glace selon sa température estimée
            energie = Ti * epaisseur_glace[1]* rho_i * c_i
            
    else: # S'il fait "chaud"
        
        if epaisseur_glace[0] > 0:
            # Estimation de la température de la glace
            if couvert > 0:
                k_s = ConductiviteNeige(dennei*rho_w)
                p = k_i*couvert/(k_s*epaisseur_glace[0])
            else:
                p = 1/(10*epaisseur_glace[0])
            Ti = Ta/(1+p)

            # Mise à jour de l'énergie contenue dans la glace selon sa température estimée
            energie = Ti * epaisseur_glace[0]* rho_i * c_i

            if couvert == 0:  # Si toute la neige est fondue, la glace peut accumuler de l'énergie et fondre
                
                # Chaleur de la pluie
                energie = energie + (meteo[2]/100) * rho_w * (Lf + c_w * Ta)

                # Radiation
                indice_radiation = (1.15 - 0.4 * np.exp(-0.38*etat['derniere_neige'])) * (meteo[4]/0.52)**0.33
                albedo = 0.33
                
                if modules['een'] == 'alt':
                    taux_fonte = param[2]/100
                elif modules['een'] == 'mdj':
                    n = len(physio['occupation'][physio['occupation'] !=0])
                    taux_fonte = 1.5 * param[27 + n]/100
                
                potentiel_fonte = taux_fonte * Ta * indice_radiation * (1-albedo)
                energie = energie + (potentiel_fonte * rho_w * Lf)
                etat['beg_fonte'] = potentiel_fonte * rho_w * Lf
                
                # Flux de chaleur à l'interface eau-glace
                energie = energie + 0.5*86400  # Leppäranta (2010)

                # Fonte si le couvert est mûr
                if energie > 0:
                    fonte = energie / (Lf*rho_w)
                    epaisseur_glace[1] = max(0,(epaisseur_glace[0])-fonte)
                else:
                    epaisseur_glace[1] = epaisseur_glace[0]
                
            else:
                epaisseur_glace[1] = epaisseur_glace[0]
                
        else:
            energie = 0
            epaisseur_glace[1] = 0
            
        
    # Calcul de la superficie du dépôt de glace. Si le niveau du
    # réservoir remonte au-delà du niveau de référence, il n'y a plus
    # de glace déposée
    superficie_glace[1] = max(0, supref - superficie_reservoir[1])

    # Arrondi pour la discrétisation spatiale par km2
    superficie_glace = np.around(superficie_glace)

    # Sauvegarde des états
    etat['reservoir_superficie']       = superficie_reservoir[1]
    etat['reservoir_superficie_glace'] = superficie_glace[1]
    etat['reservoir_epaisseur_glace']  = epaisseur_glace[1]
    etat['reservoir_superficie_ref']   = supref
    etat['reservoir_energie_glace']    = energie

    return superficie_glace, superficie_reservoir, etat



def ConductiviteNeige(densite):
    """
    Calcule la conductivité de la neige en fonction de sa densité.

    Entreées:
    densite (float): La densité de la neige en kg/m^3.

    Sorties:
    float: La conductivité de la neige en W/(m*K).
    """

    d0 = 0.36969
    d1 = 1.58688e-03
    d2 = 3.02462e-06
    d3 = 5.19756e-09
    d4 = 1.56984e-11
    p0 = 1.0

    p1 = densite - 329.6
    p2 = ((densite - 260.378) * p1) - (21166.4 * p0)
    p3 = ((densite - 320.69) * p2) - (24555.8 * p1)
    p4 = ((densite - 263.363) * p3) - (11739.3 * p2)

    conductivite =  d0 * p0 + d1 * p1 + d2 * p2 + d3 * p3 + d4 * p4

    return conductivite

