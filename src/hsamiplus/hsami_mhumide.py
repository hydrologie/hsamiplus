from __future__ import annotations

import numpy as np

def hsami_mhumide(apport,param,etat,demande,etr,physio,superficie):
    
    '''
    ENTRÉE
    -------
    apport        : Lames d'eau verticales (cm, voir hsami_interception)
    param         : Voir hsami2_noyau (projet['param'])
    etat          : Voir hsami2_noyau
    demande       : Demande évaporative de l'atmosphére (cm)
    etr           : Composantes de l'évapotranspiration (cm, voir hsami_interception)
    physio        : Voir hsami2_noyau (projet['physio'])
    superficie    : Voir hsami2_noyau (projet['superficie'])
     SORTIE
    -------
    apport : Lames d'eau verticales (cm, voir hsami_interception)
    etat   : Voir hsami2_noyau
    etr    : Composantes de l'évapotranspiration (cm, voir hsami_interception)

    EXEMPLE
    -------
    test_mhumide
    Tous les modules sont par DÉFAUT, sauf le milieu humide qui est à 1.
    '''

    # -----------------------------
    # Identification des paramétres
    # -----------------------------
    hmax = param[47]                     # Coefficient pour calcul du volume max du MHE (hmax)
    p_norm = param[48]                   # Coefficient pour détermination de la surface normale (30# dans HYDROTEL)(p_norm)
    ksat = 10**param[49]                 # Puissance de la conductivité hydraulique é saturation é la base du MHE (cm/j)
                                    

    # -----------------------------------
    # Identification des variables d'état
    # -----------------------------------
    Vinit = etat['mh_vol']
    SA = etat['mh_surf']                 # Superficie du MHE au début du pas de temps (hectares)

    SupBV = superficie[0]*100            # Surface totale du BV (en hectares)- NE VARIE PAS PDT LA SIMULATION 
    SAmax = (physio['samax'])*100        # Surface max du MHE (en hectares)- NE VARIE PAS PDT LA SIMULATION
    SAnorm = p_norm*SAmax                # Surface normale du MHE (30# de Smax dans HYDROTEL) (en hectares) - NE VARIE PAS PDT LA SIMULATION

    # Calcul de Vmax et Vnorm
    Vmax = hmax*(SAmax*10000)            # Volume d'eau max dans le MHE (m^3)   
    Vnorm = p_norm*Vmax                  # Volume d'eau normal dans le MHE (m^3)
    Vmin = 0.5 * Vnorm                   # Volume d'eau minimal dans le MHE (m^3)
    

    # Calcul des coefficients Alpha et Beta
    Alpha = (np.log10(SAmax) - np.log10(SAnorm))/(np.log10(Vmax) - np.log10(Vnorm)) # Ex.: Alpha = 1.000
    Beta = SAmax/(Vmax**Alpha)                                                      # Ex.: 1.0000e-04

    # ===================
    # Ecoulement vertical
    # ===================

    Qb = apport[0]  # écoulement de base vers le MH en cm
    Qi = apport[1]  # écoulement latéral vers le MH en cm
    Qs = apport[2]  # écoulement de surf. vers le MH en cm

    #--------------------------------------------------
    # Calcul du volume d'eau qui entre dans le MH - Vin
    #--------------------------------------------------
    
    Vb = Qb*SA*100  # en m^3 
    Vi = Qi*SA*100  # en m^3
    Vs = Qs*SA*100  # en m^3

    #------------------------------------------
    # Calcul du volume de ruissellement - Vsurf
    #------------------------------------------
    # Le volume et débit de ruissellement sont calculés en prenant é la base un
    # Vsurf de 0 et en recalculant le nouveau volume du MH. En fonction de la
    # valeur du MH et sa comparaison avec les seuil Vnorm et Vmax, on établit la
    # valeur de Vsurf et ainsi le débit et volume de ruissellement.

    Vactuel = Vinit + Vb + Vi + Vs  # Ex.: Vactuel = 2.4645e+07

    if Vactuel <= Vnorm:
        Vsurf = 0
        
    elif Vactuel <= Vmax:
        Vsurf = (Vactuel - Vnorm) / 10  # Ex.: Vsurf = 3.4845e+04
        
    elif Vactuel > Vmax:
        Vsurf = (Vactuel - Vmax) + (Vmax - Vnorm) / 10
        
    Vactuel = Vactuel - Vsurf
    
    #--------------------------------------
    # Calcul du volume d'eau évaporé - Vevap
    #---------------------------------------
    # La demande en evaporation est comblée é cette étape. étant donnée que
    # c'est un MH assimilé é un lac non connecté, la demande devrait toujours
    # étre comblée.
    # on n'offre en evap que le Vinitial - Vnormal + Vsurface)

    offre_evap = (Vactuel-Vmin)/(SA*100)

    if offre_evap > demande:
        Vevap = demande*SA*100  # Ex.: Vevap = 2.2023e+04
    else:
        Vevap = offre_evap*SA*100

    Vactuel = Vactuel - Vevap

    #-------------------------------------------------
    # Calcul du volume sortant à la base du MH - Vseep
    #-------------------------------------------------
    # é cette étape, on calcule le débit et volume de base
    # offre_seep = ce qu'il reste dans le MHE aprés l'évap.

    demande_seep = ksat*SA*100

    offre_seep = (Vactuel - Vmin)

    if offre_seep > demande_seep:
        Vseep = demande_seep  # Ex.: Vseep = 2.4633e+03
    else:
        Vseep = offre_seep

    Vactuel = Vactuel - Vseep


    # ----------------------------------------
    # Calcul de la surface et du volume du MHE
    # ----------------------------------------
    # é partir du nouveau volume, la nouvelle surface du MH peut-étre déterminée
    # Cette surface sera donc réutilisée au prochain pas de temps

    etat['mh_surf'] = Beta*(Vactuel**Alpha)
    etat['mh_vol'] = Vactuel

    # -------------------------------------------------------
    # Calcul des sorties pondérées au bassin versant et au MH
    # -------------------------------------------------------

    # Sorties du MH

    qbase_mh = np.round(Vseep*etat['ratio_MH']/(SA*100), 10)  # Ex.: qbase_mh = 9.3306e-05
    qsurf_mh = Vsurf * etat['ratio_MH']/(SA*100)           # Ex.: qsurf_mh = 0.0013
    etr_mh = np.round(Vevap*etat['ratio_MH']/(SA*100), 10)    # Ex.: etr_mh = 8.3422e-04

    # Sorties du BV pondérées

    qbase_bv = apport[0]*(1-etat['ratio_MH'])
    qintr_bv = apport[1]*(1-etat['ratio_MH'])
    qsurf_bv = apport[2]*(1-etat['ratio_MH'])

    # Sorties totales

    apport = [qbase_mh + qbase_bv, qintr_bv, qsurf_bv, apport[3], apport[4], qsurf_mh]  # Ex.: apport = [0.0507, 0, 0, -0.0894, 0, 0.0013]
    etr = np.append(etr, etr_mh) 
    etat['ratio_qbase'] = qbase_mh/(qbase_bv + qbase_mh)                             # Ex.: etat.ratio_qbase = 0.0018
    
    #if  (qbase_bv + qbase_mh) == 0 :                                                     # Ex.: etr = [0, 0, 0, 0.0711, 0.0894, 0.0008]
    #    etat['ratio_qbase'] = qbase_mh 
    #else: 
    #    etat['ratio_qbase'] = qbase_mh/(qbase_bv + qbase_mh)                             # Ex.: etat.ratio_qbase = 0.0018

    # Recalcul des ratios

    etat['ratio_MH'] = etat['mh_surf']/SupBV                          # Ex.: 0.0093
    etat['mhumide'] = etat['mh_vol']*etat['ratio_MH']/(etat['mh_surf']*100)  # Ex.: 0.9313


    return apport, etat, etr