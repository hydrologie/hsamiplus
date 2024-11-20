from __future__ import annotations


def hsami_ruissellement_surface(nb_pas, param, etat, eau_surface, modules):
    '''
     ENTRÉES
     -------
     nb_pas        : Voir hsami2_noyau (projet['nb_pas_par_jour'])
     param         : Voir hsami2_noyau (pojet['param'])
     etat          : Voir hsami2_noyau 
     eau_surface   : Quantité d'eau disponible en surface (cm)
     modules       : Voir hsami2_noyau (projet['modules'])
    
     SORTIES
     -------
     ruissellement_surface   : Quantité d'eau qui ruisselle (entre 0 et eau_surface, cm)
     infiltration            : Quantité d'eau qui pourra s'infiltrer (entre 0 et eau_surface, cm)
    
     EXEMPLE
     -------
     test_ruissellement_surface
     Tous les modules sont par DéFAUT, sauf les modules d'infiltration et de sol à
     tester ('hsami','green_ampt' et 'scs_cn' pour l'infiltration, et 'hsami' et '3couches' pour le sol)
     
    '''

    # Formulation de l'infiltration
    if modules['infiltration'] in ['green_ampt', 'scs_cn']:
        # L'eau en surface est passée dans infiltration (qui deviendra "offre") pour étre
        # traitée selon différentes formulations d'infiltration dans la fonction
        # ecoulement_vertical
        ruissellement_surface = 0.0
        infiltration = eau_surface
            
    elif modules['infiltration'] == 'hsami':
        # Contréle de l'infiltration et du ruissellement de surface
        effet_gel = param[8]    # effet du gel sur l'infiltration, adimensionnel
        effet_sol = param[9]    # effet du niveau de la réserve d'eau non saturée sur l'infiltration (cm)
        seuil_min = param[10]   # seuil minimal (sur 24h) é partir duquel le ruissellement de surface devient important (cm)
            
        # Niveau maximal de la réserve d'eau dans le sol (cm)
        if modules['sol'] == 'hsami':
            sol_max = param[12]
            
        elif modules['sol'] == '3couches':
            # Porosité totale * épaisseur de la couche 1
            sol_max = param[44] * param[39]
            
        # Eau gelée dans le sol (cm)
        gel = etat['gel']                            # eau gelée dans le sol
                    
        # Réserve d'eau non saturée (cm)
        sol = etat['sol'][0]
                
        # Calcul du seuil é partir duquel le ruissellement devient important (cm)
        # Lorsque l'eau en surface est inférieure é ce seuil, la grande majorité de l'eau s'infiltre
        seuil = effet_sol / nb_pas * (1 - sol / sol_max) - effet_gel * gel
        # Ex.: modules.sol = 'hsami'   , seuil = 1.0257
        #      modules.sol = '3couches', seuil = 1.2971
            
        # On s'assure de conserver un seuil minimal
        seuil = max(seuil, seuil_min / nb_pas)
            
        # On calcule le ruissellement de surface
        if eau_surface >= seuil:
            ruissellement_surface = eau_surface - seuil / 2
        else:
            ruissellement_surface = eau_surface ** 2 / (2 * seuil)
            
        # Le reste s'infiltre
        infiltration = eau_surface - ruissellement_surface
            
                
    return ruissellement_surface, infiltration