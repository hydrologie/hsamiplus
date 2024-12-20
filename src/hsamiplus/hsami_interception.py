"""The function simulates the interception of water in HSAMI+ model."""

from __future__ import annotations

import numpy as np
from numba import jit


def hsami_interception(nb_pas, jj, param, meteo, etp, etat, modules, physio):
    """
    Compute interception.

    Parameters
    ----------
    nb_pas : float
        Nombre de pas de temps.
    jj : int
        Jour julien.
    param : list
        Paramètres pour la simulation.
    meteo : dict
        Données météorologiques pour la simulation.
    etp : float
        Évapotranspiration potentielle du pas de temps (bassin et réservoir).
    etat : dict
        États du bassin versants et du réservoir.
    modules : dict
        Les modules pour la simulation.
    physio : dict
        Les données physiographiques.

    Returns
    -------
    eau_surface : float
        Eau disponible à la surface pour évaporation, ruissellement et infiltration.
    demande_eau : float
        Demande en eau restante.
    etat :  dict
        États du bassin versants et du réservoir.
    apport_vertical : list
        Lames d'eau à moduler par les hydrogrammes unitaires.
    """
    # --------------------
    # Paramétres temporels
    # --------------------
    # La durée est la fraction d'une journée correspondant à un pas de temps
    duree = 1 / nb_pas
    pas_de_temps = 24 / nb_pas
    # Pas de temps en secondes
    pdts = pas_de_temps * 60 * 60

    # --------------------------------------
    # Initialisation des variables de sortie
    # --------------------------------------
    apport_vertical = np.zeros(5)
    etr = np.zeros(5)

    # -----------------------------
    # Identification des Paramétres
    # -----------------------------
    efficacite_evapo_ete = param[0]
    efficacite_evapo_hiver = param[1]
    taux_fonte_jour = param[2]  # en cm/degre C/jour
    taux_fonte_nuit = param[3]  # en cm/degre C/jour
    temp_fonte_jour = param[4]  # C
    temp_fonte_nuit = param[5]  # C
    temp_ref_pluie = param[6]  # C
    effet_redoux_sur_aire_enneigee = param[7]  # adimensionnel
    sol_min = param[11]

    # -----------------------------------
    # Variables d'Args :  météorologiques
    # -----------------------------------
    t_min = meteo["bassin"][0]  # Valeur extréme (observée ou prévue) sur 24h (Celcius)
    t_max = meteo["bassin"][1]  # Valeur extréme (observée ou prévue) sur 24h (Celcius)
    pluie = meteo["bassin"][2]  # Total pour le pas de temps (cm)
    neige = meteo["bassin"][3]  # Total pour le pas de temps (cm)

    if len(meteo["bassin"]) >= 5:
        # Ensoleillement (observé ou prévu) pour la journée (entre 0 et 1)
        soleil = meteo["bassin"][4]
    else:
        soleil = 0.5

    demande_eau = etp[0] * efficacite_evapo_ete
    demande_reservoir = etp[1] * efficacite_evapo_ete

    # -----------------------------------
    # Identification des variables d'état
    # -----------------------------------
    neige_au_sol = etat[
        "neige_au_sol"
    ]  # équivalent en eau de la neige au sol incluant l'eau de fonte
    fonte = etat["fonte"]  # eau liquide stockée dans la neige
    neige_au_sol_totale = etat["nas_tot"]  # total des chutes de neige pendant l'hiver
    fonte_totale = etat["fonte_tot"]  # total de la fonte de neige pendant l'hiver
    derniere_neige = etat["derniere_neige"]  # nombre de jours depuis la derniere neige
    eeg = etat["eeg"]  # équivalent en eau de la glace
    gel = etat["gel"]  # eau gelée dans la zone non saturée

    if modules["sol"] == "hsami":
        sol = etat["sol"][0]  # reserve d'eau dans la zone non-saturée
    elif modules["sol"] == "3couches":
        sol = etat["sol"][0]  # reserve d'eau dans la zone non-saturée
        sol_min = (
            param[41] * param[39]
        )  # Le gel de l'eau dans le sol est permis jusqu'au point de flétrissement permanent

    # ----------
    # SIMULATION
    # ----------
    # Modules hsami et dj
    if modules["een"] == "hsami" or modules["een"] == "dj":
        eau_surface, demande_eau, etat, etr, apport_vertical = dj_hsami(
            modules,
            meteo,
            etat,
            apport_vertical,
            etr,
            duree,
            efficacite_evapo_hiver,
            taux_fonte_jour,
            taux_fonte_nuit,
            temp_fonte_jour,
            temp_fonte_nuit,
            temp_ref_pluie,
            effet_redoux_sur_aire_enneigee,
            sol_min,
            sol,
            t_min,
            t_max,
            pluie,
            neige,
            soleil,
            demande_eau,
            demande_reservoir,
            neige_au_sol,
            fonte,
            neige_au_sol_totale,
            fonte_totale,
            derniere_neige,
            eeg,
            gel,
        )

    # Module mdj et alt
    elif modules["een"] == "mdj" or modules["een"] == "alt":
        eau_surface, demande_eau, etat, etr, apport_vertical = mdj_alt(
            param,
            modules,
            meteo,
            physio,
            etat,
            apport_vertical,
            etr,
            duree,
            pdts,
            jj,
            pas_de_temps,
            efficacite_evapo_hiver,
            temp_fonte_jour,
            sol_min,
            sol,
            t_min,
            t_max,
            pluie,
            neige,
            soleil,
            demande_eau,
            demande_reservoir,
            neige_au_sol,
            fonte,
            derniere_neige,
            eeg,
            gel,
        )

    return eau_surface, demande_eau, etat, etr, apport_vertical


def dj_hsami(  # noqa: C901
    modules,
    meteo,
    etat,
    apport_vertical,
    etr,
    duree,
    efficacite_evapo_hiver,
    taux_fonte_jour,
    taux_fonte_nuit,
    temp_fonte_jour,
    temp_fonte_nuit,
    temp_ref_pluie,
    effet_redoux_sur_aire_enneigee,
    sol_min,
    sol,
    t_min,
    t_max,
    pluie,
    neige,
    soleil,
    demande_eau,
    demande_reservoir,
    neige_au_sol,
    fonte,
    neige_au_sol_totale,
    fonte_totale,
    derniere_neige,
    eeg,
    gel,
):
    """
    Module "hsami" et  "dj" pour calculer "een".

    Parameters
    ----------
    modules : dict
        Les modules pour la simulation.
    meteo : dict
        Données météorologiques pour la simulation.
    etat : dict
        États du bassin versants et du réservoir.
    apport_vertical : list
        Lames d'eau à moduler par les hydrogrammes unitaires.
    etr : list
        Évapotranspiration et évaporation.
    duree : float
        Fraction d'une journée correspondant à un pas de temps.
    efficacite_evapo_hiver : float
        Param[1].
    taux_fonte_jour : float
        Param[2]  en cm/degre C/jour.
    taux_fonte_nuit : float
        Param[3]  en cm/degre C/jour.
    temp_fonte_jour : float
        Param[4]  en C.
    temp_fonte_nuit : float
        Param[5]  en C.
    temp_ref_pluie : float
        Param[6]  en C.
    effet_redoux_sur_aire_enneigee : float
        Pparam[7].
    sol_min : float
        Param[11].
    sol : float
        Reserve d'eau dans la zone non-saturée.
    t_min : float
        Valeur extréme (observée ou prévue) sur 24h (Celcius).
    t_max : float
        Valeur extréme (observée ou prévue) sur 24h (Celcius).
    pluie : float
        Total pour le pas de temps (cm).
    neige : float
        Total pour le pas de temps (cm).
    soleil : int
        Ensoleillement (observé ou prévu) pour la journée (entre 0 et 1).
    demande_eau : float
        Demande en eau restante.
    demande_reservoir : float
        Demande en eau restante pour le reservoir.
    neige_au_sol : float
        Équivalent en eau de la neige au sol incluant l'eau de fonte.
    fonte : float
        Eau liquide stockée dans la neige.
    neige_au_sol_totale : float
        Total des chutes de neige pendant l'hiver.
    fonte_totale : float
        Total de la fonte de neige pendant l'hiver.
    derniere_neige : int
        Nombre de jours depuis la derniere neige.
    eeg : flpoat
        Équivalent en eau de la glace.
    gel : float
        Eau gelée dans la zone non saturée.

    Returns
    -------
    eau_surface : float
        Eau disponible à la surface pour évaporation, ruissellement et infiltration.
    demande_eau : float
        Demande en eau restante.
    etat : dict
        États du bassin versants et du réservoir.
    etr : list
        Évapotranspiration et évaporation.
    apport_vertical : list
        Lames d'eau à moduler par les hydrogrammes unitaires.
    """
    # Modéle degré-jour
    # -----------------

    # -----------------------------------------------
    # Gestion de la portion en eau libre du réservoir
    # -----------------------------------------------
    # La pluie et la neige tombent dans le réservoir

    apport_vertical[3] = meteo["reservoir"][2] + meteo["reservoir"][3]

    # On tient le compte du nombre de jours sans neige
    seuil_neige_modifiant_albedo = 0

    if (neige_au_sol > 0) and (neige <= seuil_neige_modifiant_albedo):
        derniere_neige = derniere_neige + duree
    else:
        derniere_neige = 0

    # Sur la premiere ligne de la sixiéme colonne, on peut retrouver un relevé de neige
    if (len(meteo["bassin"]) == 6) and (meteo["bassin"][5] >= 0):
        # Si c'est le cas, on met é jour la neige au sol en fonction du relevé
        delta_neige = neige_au_sol_totale - neige_au_sol
        neige_au_sol = meteo["bassin"][5]

        # On conserve l'écart
        neige_au_sol_totale = neige_au_sol + delta_neige

    # Ajout de la précipitation neigeuse
    neige_au_sol = neige_au_sol + neige
    neige_au_sol_totale = neige_au_sol_totale + neige

    # ===================================================
    # Fonte ou gel en fonction de la température maximale
    # ===================================================

    dt_max = t_max - temp_fonte_jour
    dt_min = t_min - temp_fonte_nuit

    if dt_max < 0:
        # -------------
        # Il fait froid
        # -------------
        demande_eau = demande_eau * efficacite_evapo_hiver
        demande_reservoir = demande_reservoir * efficacite_evapo_hiver

        # évaporation de l'eau du réservoir au taux hivernal
        etr[4] = demande_reservoir
        apport_vertical[3] = apport_vertical[3] - etr[4]

        # Ajout de la précipitation liquide au bassin
        neige_au_sol = neige_au_sol + pluie
        neige_au_sol_totale = neige_au_sol_totale + pluie
        fonte = fonte + pluie
        fonte_totale = fonte_totale + pluie

        # Sublimation
        if demande_eau < neige_au_sol:
            neige_au_sol = neige_au_sol - demande_eau
            etr[0] = demande_eau
        else:
            etr[0] = neige_au_sol
            neige_au_sol = 0
            neige_au_sol_totale = 0

        demande_eau = 0

        # gel de l'eau dans le sol
        sol, gel = gel_sol(duree, dt_max, sol_min, sol, gel, neige_au_sol)
        # Ex1. modules['een'] = 'hsami' : sol = 2.6924
        #                                 gel = 0.0337
        #      modules['een'] = 'dj'    : sol = 2.9964
        #                                 gel = 0.0368

        eau_surface = 0

        # Pour eviter des problémes numériques, on ne fait pas évoluer
        # un stock de neige de moins de un centiéme de pouce par temps
        # froid
        if neige_au_sol > 0.0254:
            # gel de l'eau libre dans la neige
            fonte, fonte_totale = gel_neige(
                duree, dt_max, neige_au_sol, fonte, fonte_totale
            )
            # Ex1. : fonte = 0 ('hsami' et 'dj')
            #        fonte_totale = 0 ('hsami et 'dj')

            # s'il y a de l'eau libre dans la neige, elle peut percoler
            if fonte > 0:
                (
                    eau_fonte,
                    neige_au_sol,
                    neige_au_sol_totale,
                    fonte,
                    fonte_totale,
                ) = percolation_eau_fonte(
                    neige_au_sol, neige_au_sol_totale, fonte, fonte_totale
                )
                eau_surface = eau_fonte

    else:  # dt_max >= 0
        # -------------
        # Il fait chaud
        # -------------
        # Effet de la température sur l'eau gelée
        if gel > 0:
            # une partie de l'eau gelée dans le sol va retourner dans les réserves liquides du sol
            # une autre va ruisseler (ruissellement hypodermique)
            sol, gel = degel_sol(duree, dt_max, sol, gel, neige_au_sol)

        # L'évaporation du réservoir se produit au rythme estival
        etr[4] = demande_reservoir
        apport_vertical[3] = apport_vertical[3] - etr[4]

        # On fait d'abord fondre la neige, ensuite la glace s'il n'y a
        # plus de neige

        if neige_au_sol > 0:
            # On estime la proportion du bassin qui est couverte de neige
            aire_enneigee = effet_redoux_sur_aire_enneigee * (
                1 - fonte_totale / neige_au_sol_totale
            )
            aire_enneigee = max(0.1, min(aire_enneigee, 1))

            # Estimation de l'accélération de la fonte causée par la radiation solaire
            effet_radiation = (1.15 - 0.4 * np.exp(-0.38 * derniere_neige)) * (
                soleil / 0.52
            ) ** 0.33

            # On estime la fonte pour le jour et la nuit
            fonte_jour = (
                dt_max * aire_enneigee * taux_fonte_jour * effet_radiation * duree
            )
            fonte_nuit = dt_min * aire_enneigee * taux_fonte_nuit * duree

            neige_fondue = fonte_jour + fonte_nuit

            # On accentue la fonte en tenant compte de la chaleur de la pluie
            t_moy = 2 / 3 * t_max + 1 / 3 * t_min
            if t_moy > temp_ref_pluie:
                effet_chaleur_pluie = (
                    0.0126 * (t_moy - temp_ref_pluie) * aire_enneigee * pluie
                )
                neige_fondue = neige_fondue + effet_chaleur_pluie

            if modules["een"] == "hsami":
                # On ajoute la pluie au stock de neige et on retire l'évaporation
                pluie_moins_evaporation = (
                    pluie - efficacite_evapo_hiver * demande_eau
                ) * aire_enneigee

                if neige_au_sol + pluie_moins_evaporation < 0:
                    etr[1] = neige_au_sol + pluie * aire_enneigee
                else:
                    etr[1] = efficacite_evapo_hiver * demande_eau * aire_enneigee

                neige_fondue = neige_fondue + pluie_moins_evaporation

                nas_avant_pme = neige_au_sol
                neige_au_sol = neige_au_sol + pluie_moins_evaporation
                neige_au_sol_totale = neige_au_sol_totale + pluie_moins_evaporation

                if neige_au_sol < 0:
                    neige_au_sol = 0
                    etr[1] = nas_avant_pme

                # On ajoute la neige fondue à l'eau de fonte dans la neige
                if neige_fondue > 0:
                    fonte = fonte + neige_fondue
                    fonte_totale = fonte_totale + neige_fondue

            elif modules["een"] == "dj":
                # Fonte de la neige solide disponible ou gel de
                # l'eau dans la neige si potentiel_fonte est négatif
                potentiel_fonte = neige_fondue
                neige_solide = neige_au_sol - fonte

                if potentiel_fonte < 0:
                    potentiel_gel = (
                        -potentiel_fonte
                    )  # potentiel_fonte est en réalité un potentiel de gel

                    if fonte - potentiel_gel >= 0:  # s'il y a assez d'eau de fonte
                        fonte = fonte - potentiel_gel  # le potentiel de gel est comblé
                        neige_solide = (
                            neige_solide + potentiel_gel
                        )  # l'eau est transférée à la phase solide
                    else:
                        neige_solide = (
                            neige_solide + fonte
                        )  # sinon toute l'eau de fonte disponible est gelée
                        fonte = 0

                elif neige_solide - potentiel_fonte >= 0:
                    fonte = fonte + potentiel_fonte  # le potentiel de fonte est comblé
                    neige_solide = (
                        neige_solide - potentiel_fonte
                    )  # la neige solide est réduite
                else:
                    fonte = fonte + neige_solide  # toute la neige solide fond
                    neige_solide = 0

                # évaporation réelle
                demande = demande_eau * efficacite_evapo_hiver * aire_enneigee

                # On satisfait d'abord la demande evaporative avec la pluie
                pluie_sur_neige = pluie * aire_enneigee

                if demande > 0:
                    if pluie_sur_neige - demande >= 0:
                        etr[1] = demande  # la demande est satisfaite
                        pluie_sur_neige = (
                            pluie_sur_neige - demande
                        )  # la pluie est réduite
                    else:
                        etr[1] = pluie_sur_neige  # toute la pluie est évaporée
                        demande = demande - pluie_sur_neige  # la demande est réduite
                        pluie_sur_neige = 0

                        # On utilise ensuite la fonte
                        if fonte - demande >= 0:
                            etr[1] = etr[1] + demande  # la demande est comblée
                            fonte = fonte - demande  # l'eau de fonte est réduite
                        else:
                            etr[1] = etr[1] + fonte  # toute l'eau de fonte est évaporée
                            demande = demande - fonte  # la demande est réduite
                            fonte = 0

                            # On sublime ensuite la neige solide
                            if neige_solide - demande >= 0:
                                etr[0] = etr[0] + demande  # la demande est comblée
                                neige_solide = (
                                    neige_solide - demande
                                )  # la neige est réduite
                            else:
                                etr[0] = (
                                    etr[0] + neige_solide
                                )  # toute la neige est sublimée
                                neige_solide = 0

                # Mise à jour du couvert de neige
                fonte = fonte + pluie_sur_neige
                neige_au_sol = neige_solide + fonte

            # L'eau qui tombe sur sol nu est disponible pour infiltration ou ruissellement
            eau_surface = pluie * (1 - aire_enneigee)

            # On corrige l'évapotranspiration pour qu'elle s'applique seulement é la portion de sol nu
            demande_eau = demande_eau * (1 - aire_enneigee)

            if fonte < neige_au_sol:
                # une partie de l'eau dans la neige peut percoler
                (
                    eau_fonte,
                    neige_au_sol,
                    neige_au_sol_totale,
                    fonte,
                    fonte_totale,
                ) = percolation_eau_fonte(
                    neige_au_sol, neige_au_sol_totale, fonte, fonte_totale
                )
                eau_surface = eau_surface + eau_fonte
            else:
                eau_surface = eau_surface + neige_au_sol
                neige_au_sol = 0
                neige_au_sol_totale = 0
                fonte = 0
                fonte_totale = 0

            # On vérifie si toute la neige a fondue, si oui, on fait
            # fondre la glace (s'il y en a)

            # for i_g in range(len(eeg)):
            #     if neige_au_sol == 0 and eeg[i_g] > 0:
            #         # Estimation de l'accélération de la fonte causée par la radiation solaire
            #         effet_radiation = (1.15 - 0.4 * np.exp(-0.38 * derniere_neige)) * (
            #             soleil / 0.52
            #         ) ** 0.33

            #         # On estime la fonte pour le jour et la nuit.
            #         # Les taux de fonte de la neige sont multipliés
            #         # par 1.5 pour la glace selon Braithwaite (1995)
            #         # et Singh et al (1999).
            #         fonte_jour = (
            #             dt_max * 1.5 * taux_fonte_jour * effet_radiation * duree
            #         )
            #         fonte_nuit = dt_min * 1.5 * taux_fonte_nuit * duree

            #         potentiel_fonte = fonte_jour + fonte_nuit

            #         # On accentue la fonte en tenant compte de la chaleur de la pluie
            #         t_moy = 2 / 3 * t_max + 1 / 3 * t_min

            #         if t_moy > temp_ref_pluie:
            #             effet_chaleur_pluie = (
            #                 0.0126 * (t_moy - temp_ref_pluie) * meteo.reservoir(3)
            #             )
            #             potentiel_fonte = potentiel_fonte + effet_chaleur_pluie

            #         # Fonte réelle en fonction de la glace disponible
            #         # (Si le potentiel de fonte est inférieur é 0, on ne
            #         # fait pas geler la glace puisque la glace ne contient pas d'eau libre é geler)
            #         if potentiel_fonte > 0:
            #             if potentiel_fonte >= eeg[i_g]:
            #                 apport_vertical[4] = apport_vertical[4] + eeg[i_g]
            #                 eeg[i_g] = 0
            #             else:
            #                 apport_vertical[4] = apport_vertical[4] + potentiel_fonte
            #                 eeg[i_g] = eeg[i_g] - potentiel_fonte

            mask_eeg = (eeg > 0) & (neige_au_sol == 0)
            if np.any(mask_eeg):
                # Estimation de l'accélération de la fonte causée par la radiation solaire
                effet_radiation = (1.15 - 0.4 * np.exp(-0.38 * derniere_neige)) * (
                    soleil / 0.52
                ) ** 0.33

                # On estime la fonte pour le jour et la nuit.
                # Les taux de fonte de la neige sont multipliés
                # par 1.5 pour la glace selon Braithwaite (1995)
                # et Singh et al (1999).
                fonte_jour = dt_max * 1.5 * taux_fonte_jour * effet_radiation * duree
                fonte_nuit = dt_min * 1.5 * taux_fonte_nuit * duree

                potentiel_fonte = fonte_jour + fonte_nuit

                # On accentue la fonte en tenant compte de la chaleur de la pluie
                t_moy = 2 / 3 * t_max + 1 / 3 * t_min

                if t_moy > temp_ref_pluie:
                    effet_chaleur_pluie = (
                        0.0126 * (t_moy - temp_ref_pluie) * meteo.reservoir(3)
                    )
                    potentiel_fonte = potentiel_fonte + effet_chaleur_pluie

                # Fonte réelle en fonction de la glace disponible
                # (Si le potentiel de fonte est inférieur é 0, on ne
                # fait pas geler la glace puisque la glace ne contient pas d'eau libre é geler)
                if potentiel_fonte > 0:
                    mask_pf_eeg = potentiel_fonte >= eeg
                    apport_vertical[4] = apport_vertical[4] + np.sum(eeg[mask_pf_eeg])
                    apport_vertical[4] = apport_vertical[4] + potentiel_fonte * np.sum(
                        mask_pf_eeg
                    )
                    eeg[mask_pf_eeg] = 0
                    eeg[~mask_pf_eeg] = eeg[~mask_pf_eeg] - potentiel_fonte

        else:
            eau_surface = pluie
            # Il n'y a pas de neige, mais il peut y avoir de la glace é fondre

            # for i_g in range(len(eeg)):
            #     if eeg[i_g] > 0:
            #         # Estimation de l'accélération de la fonte causée par la radiation solaire
            #         effet_radiation = (1.15 - 0.4 * np.exp(-0.38 * derniere_neige)) * (
            #             soleil / 0.52
            #         ) ** 0.33

            #         # On estime la fonte pour le jour et la nuit.
            #         # Les taux de fonte de la neige sont multipliés
            #         # par 1.5 pour la glace selon Braithwaite (1995)
            #         # et Singh et al (1999).
            #         fonte_jour = (
            #             dt_max * 1.5 * taux_fonte_jour * effet_radiation * duree
            #         )
            #         fonte_nuit = dt_min * 1.5 * taux_fonte_nuit * duree

            #         potentiel_fonte = fonte_jour + fonte_nuit

            #         # On accentue la fonte en tenant compte de la chaleur de la pluie
            #         t_moy = 2 / 3 * t_max + 1 / 3 * t_min

            #         if t_moy > temp_ref_pluie:
            #             effet_chaleur_pluie = (
            #                 0.0126 * (t_moy - temp_ref_pluie) * meteo["reservoir"][2]
            #             )
            #             potentiel_fonte = potentiel_fonte + effet_chaleur_pluie

            #         # Fonte réelle en fonction de la glace disponible
            #         # (Si le potentiel de fonte est inférieur é 0, on ne
            #         # fait pas geler la glace puisque la glace ne
            #         # contient pas d'eau libre é geler)
            #         if potentiel_fonte > 0:
            #             if potentiel_fonte >= eeg[i_g]:
            #                 apport_vertical[4] = apport_vertical[4] + eeg[i_g]
            #                 eeg[i_g] = 0
            #             else:
            #                 apport_vertical[4] = apport_vertical[4] + potentiel_fonte
            #                 eeg[i_g] = eeg[i_g] - potentiel_fonte

            mask_eeg = eeg > 0
            if np.any(mask_eeg):
                if t_moy > temp_ref_pluie:
                    effet_chaleur_pluie = (
                        0.0126 * (t_moy - temp_ref_pluie) * meteo.reservoir(3)
                    )
                    potentiel_fonte = potentiel_fonte + effet_chaleur_pluie

                if potentiel_fonte > 0:
                    mask_pf_eeg = potentiel_fonte >= eeg
                    apport_vertical[4] = apport_vertical[4] + np.sum(eeg[mask_pf_eeg])
                    apport_vertical[4] = apport_vertical[4] + potentiel_fonte * np.sum(
                        mask_pf_eeg
                    )
                    eeg[mask_pf_eeg] = 0
                    eeg[~mask_pf_eeg] = eeg[~mask_pf_eeg] - potentiel_fonte

    # ====================
    # Sauvegarde de l'état
    # ====================
    #  'hsami'   'dj'
    etat["neige_au_sol"] = neige_au_sol  # Ex1. : 5.7187    5.0373
    etat["fonte"] = fonte  # Ex1. : 0         0
    etat["nas_tot"] = neige_au_sol_totale  # Ex1. : 10.3942   6.7000
    etat["fonte_tot"] = fonte_totale  # Ex1. : 0         0
    etat["derniere_neige"] = derniere_neige  # Ex1. : 1         1
    etat["eeg"] = eeg  # Ex1. : (vecteur de 0)
    etat["gel"] = gel  # Ex1. : 0.0337    0.0368

    if modules["sol"] == "hsami":
        etat["sol"][0] = sol

    elif modules["sol"] == "3couches":
        etat["sol"][0] = sol

    return eau_surface, demande_eau, etat, etr, apport_vertical


def mdj_alt(  # noqa: C901
    param,
    modules,
    meteo,
    physio,
    etat,
    apport_vertical,
    etr,
    duree,
    pdts,
    jj,
    pas_de_temps,
    efficacite_evapo_hiver,
    temp_fonte_jour,
    sol_min,
    sol,
    t_min,
    t_max,
    pluie,
    neige,
    soleil,
    demande_eau,
    demande_reservoir,
    neige_au_sol,
    fonte,
    derniere_neige,
    eeg,
    gel,
):
    """
    Module "mdj" et "alt" pour calculer "een".

    Parameters
    ----------
    param : list
        Paramètres pour la simulation.
    modules : dict
        Les modules pour la simulation.
    meteo : dict
        Données météorologiques pour la simulation.
    physio : dict
        Les données physiographiques peuvent être vides.
    etat : dict
        États du bassin versants et du réservoir.
    apport_vertical : list
        Lames d'eau à moduler par les hydrogrammes unitaires.
    etr : list
        Évapotranspiration et évaporation.
    duree : float
        Fraction d'une journée correspondant à un pas de temps.
    pdts : float
        Pas de temps en secondes.
    jj : int
        Jour julien.
    pas_de_temps : int
        Pas de temps.
    efficacite_evapo_hiver : float
        Param[1].
    temp_fonte_jour : float
        Param[4]  en C.
    sol_min : float
        Param[11].
    sol : float
        Reserve d'eau dans la zone non-saturée.
    t_min : float
        Valeur extréme (observée ou prévue) sur 24h (Celcius).
    t_max : float
        Valeur extréme (observée ou prévue) sur 24h (Celcius).
    pluie : float
        Total pour le pas de temps (cm).
    neige : float
        Total pour le pas de temps (cm).
    soleil : int
        Ensoleillement (observé ou prévu) pour la journée (entre 0 et 1).
    demande_eau : float
        Demande en eau restante.
    demande_reservoir : float
        Demande en eau restante pour le reservoir.
    neige_au_sol : float
        Équivalent en eau de la neige au sol incluant l'eau de fonte.
    fonte : float
        Eau liquide stockée dans la neige.
    derniere_neige : int
        Nombre de jours depuis la derniere neige.
    eeg : flpoat
        Équivalent en eau de la glace.
    gel : float
        Eau gelée dans la zone non saturée.

    Returns
    -------
    eau_surface : float
        Eau disponible à la surface pour évaporation, ruissellement et infiltration.
    demande_eau : float
        Demande en eau restante.
    etat : dict
        États du bassin versants et du réservoir.
    etr : list
        Évapotranspiration et évaporation.
    apport_vertical : list
        Lames d'eau à moduler par les hydrogrammes unitaires.
    """
    # Modéle mixte-degré-jour
    if modules["een"] == "mdj":
        occupation = physio["occupation"]

    elif modules["een"] == "alt":
        occupation = physio["occupation_bande"]

    # Nombre de milieux
    n = len([m for m in occupation if m != 0])

    # Paramétres du modéle de fonte Mixte degré-jour
    taux_de_fonte = np.zeros(n)
    temperature_de_fonte = np.zeros(n)

    if modules["een"] == "mdj":
        for i_z in range(n):
            taux_de_fonte[i_z] = (
                param[27 + i_z] / 100
            )  # Taux de fonte - milieu(i_z) # /100 pour cm --> m/degC/jour
            temperature_de_fonte[i_z] = param[
                30 + i_z
            ]  # Température de fonte - milieu(i_z) # degC

    elif modules["een"] == "alt":
        taux_de_fonte[:] = param[2] / 100
        temperature_de_fonte[:] = param[4]

    taux_fonte_ns = 0.0005  # Taux de fonte neige-sol. # /100 pour cm --> m/jour
    capacite_retenue = param[
        35
    ]  # varie entre 0 et 15 # selon Singh et Singh (2001), p.108, disponible é http://books.google.ca/books?id=0VW6Tv0LVWkC&pg=PA108&lpg=PA108&dq=maximum+liquid+water+capacity+snow&source=bl&ots=8UjCZB0H2u&sig=zeC1iSHR3pcBLUrF2qC1hYhindQ&hl=fr&sa=X&ei=-G_6Ut2sIIa9yAH-34GAAg&ved=0CCgQ6AEwAA#v=onepage&q=maximum#20liquid#20water#20capacity#20snow&f=false.

    # ----------
    # Constantes
    # ----------
    rho_w = 1000  # Masse volumique de léeau (kg/m3)
    chaleur_latente_fusion = (
        335000  # Chaleur de fusion de l'eau (J/kg) solide-->liquide
    )
    chaleur_latente_evaporation = (
        2500000  # Chaleur de vaporisation de l'eau (J/kg) liquide-->gaseux
    )
    chaleur_latente_sublimation = (
        2834000  # Chaleur de sublimation de l'eau (J/kg) solide-->gaseux
    )
    capacite_thermique_massique_eau_solide = (
        2093.4  # Chaleur spécifique de l'eau solide é 0degC (J/(kg*degC))
    )
    capacite_thermique_massique_eau_liquide = (
        4216  # Chaleur spécifique de l'eau liquide é 0degC (J/(kg*degC))
    )
    constante_tassement = 0.1  # Pour le calcul de la compaction
    densite_maximale = (
        466  # Densité maximale d'un couvert de neige (kg/m3) - Turcotte et al. (2007)
    )
    conductivite_glace = 2.24  # Conductivité thermique de la glace, (W/(m*degC))

    # -----------------------------------------------------
    # Conversion des cm aux m pour le modéle mixte degré-jour
    # -----------------------------------------------------
    demande_eau = demande_eau / 100
    demande_eau = np.tile(demande_eau, n)
    demande_reservoir = demande_reservoir / 100
    pluie = pluie / 100
    neige = neige / 100
    eeg = eeg / 100

    nas_moy = np.sum(
        [a * b for a, b in zip(etat[modules["een"]]["neige_au_sol"][0:n], occupation)]
    )  # nas_moy sert seulement pour la maj de l'een.

    # Ex1. :  modules['een'] = 'mdj', nas_moy = 0.0653
    #                          'alt', nas_moy = 0.1023

    # -----------------------------------------------------------------------
    # Détermination du nombre de jour depuis la derniére neige pour le calcul
    # de la radiation si l'orientation et la pente sont inconnues
    # -----------------------------------------------------------------------
    # On tient le compte du nombre de jours sans neige.

    seuil_neige_modifiant_albedo = 0
    if neige_au_sol > 0 and neige <= seuil_neige_modifiant_albedo:
        derniere_neige = derniere_neige + duree
    else:
        derniere_neige = 0

    # ========================================================================
    # Détermination de l'eau disponible pour le ruissellement de surface selon
    # les 3 zones d'occupation du sol
    # ========================================================================
    eau_surface_zones = np.zeros(n)
    sublimation = np.zeros(n)
    evapo_eau_neige = np.zeros(n)

    # -----------------------------------------------
    # Gestion de la portion en eau libre du réservoir
    # -----------------------------------------------
    # La pluie et la neige tombent au réservoir
    apport_vertical[3] = meteo["reservoir"][2] / 100 + meteo["reservoir"][3] / 100

    # évaporation de l'eau du réservoir au taux hivernal ou estival
    # selon dt_max comme dans le modéle degré-jour
    dt_max = t_max - temp_fonte_jour
    if dt_max < 0:
        etr[4] = demande_reservoir * efficacite_evapo_hiver
        apport_vertical[3] = apport_vertical[3] - etr[4]
    else:
        etr[4] = demande_reservoir
        apport_vertical[3] = apport_vertical[3] - etr[4]

    # On calcule la neige_au_sol et la fonte pour chaque zone d'occupation
    for i_z in range(n):
        # ------------------------------------------------------
        # Récupération des états propres aux zones d'occupation
        # ------------------------------------------------------

        # Récupération des variables d'états propres aux zones qui ne sont pas
        # remoyennées au bassin versant et qui ne sont pas recalculées dans une
        # autre fonction d'HSAMI

        neige_au_sol = etat[modules["een"]]["neige_au_sol"][i_z]
        couvert_neige = etat[modules["een"]]["couvert_neige"][i_z]
        dennei = etat[modules["een"]]["densite_neige"][i_z]
        fonte = etat[modules["een"]]["fonte"][i_z]
        albedo_neige = etat[modules["een"]]["albedo_neige"][i_z]
        energie_neige = etat[modules["een"]]["energie_neige"][i_z]

        if i_z in range(n):  # La glace évolue avec le milieu le plus ouvert
            energie_glace = etat[modules["een"]]["energie_glace"]

        sol = etat["sol"][0]
        gel = etat["gel"]
        demande = demande_eau[i_z]

        # Initialisation de la sublimation et de l'etr pluie sur neige
        # pour le milieu i_z, sinon les valeurs du milieu précédent sont
        # réutilisées.
        etr[0:1] = 0

        # Calcul des températures par bande d'altitude et partition de
        # la précipitation
        # ------------------------------------------------------------
        if modules["een"] == "alt":
            # Récupération des températures à l'échelle du bassin
            t_min = meteo["bassin"][0]
            t_max = meteo["bassin"][1]

            # Récupération des altitudes
            alt_milieu = physio["altitude_bande"][int(n / 2)]
            alt_bande = physio["altitude_bande"][i_z]

            # Application du gradient de température de 0.6°C/100m (E. Paquet, 2004)
            t_max = t_max - 0.6 * (alt_bande - alt_milieu) / 100
            t_min = t_min - 0.6 * (alt_bande - alt_milieu) / 100

            # Partage de phase de la précipitation
            pluie = meteo["bassin"][2] / 100
            neige = meteo["bassin"][3] / 100
            pluie, neige = pluie_neige(t_min, t_max, pluie + neige)

        # ------------------------------
        # Mise é jour de la neige au sol
        # ------------------------------
        # À la sixiéme colonne de la météo, on peut retrouver un
        # relevé de neige. Le relevé est une valeur moyenne pour le bassin.
        # Lors d'une mise é jour avec le modéle mdj, toutes les occupations se
        # retrouvent avec la méme valeur moyenne.

        if len(meteo["bassin"]) == 6 and meteo["bassin"][5] >= 0:
            # Si c'est le cas, on met à jour la neige au sol en fonction du relevé
            # Mise à jour en pondérant selon les quantités présentes dans les milieux
            # avant la maj.
            if nas_moy != 0:
                facteur_maj = (meteo["bassin"][5] / 00) / nas_moy
            else:
                facteur_maj = 1
            neige_au_sol = neige_au_sol * facteur_maj

            # Hypothèse : la densité de la neige est la même qu'avant la mise à
            # jour. S'il n'y avait plus de neige simulée avant la maj, la
            # densité est estimée à 300 kg/m3, qui est une valeur moyenne vers
            # la mi et fin de l'hiver.
            if dennei <= 0:
                dennei = 0.3
            couvert_neige = neige_au_sol / dennei

        # =====================================================
        # Gel du sol et dégel du sol selon un modéle degré-jour
        # =====================================================
        # Calcul du dt_max pour le gel du sol
        dt_max = t_max - temperature_de_fonte[i_z]

        if dt_max <= 0:
            sol, gel = gel_sol(
                duree, dt_max, sol_min, sol, gel, neige_au_sol * 100
            )  # *100 car Neige au sol doit étre en cm.
            # Ex1. : modules['een'] = 'mdj', i_z = 1, sol = 2.4932, gel = 0.0351
            #                                i_z = 2, sol = 2.4888, gel = 0.0395
            #                                i_z = 3, sol = 2.4888, gel = 0.0395
            #
            #        modules['een'] = 'alt', i_z = 1, sol = 0.7829, gel = 0.0226
            #                                i_z = 2, sol = 0.7835, gel = 0.0220
            #                                i_z = 3, sol = 0.7840, gel = 0.0215
            #                                i_z = 4, sol = 0.7844, gel = 0.0211
            #                                 i_z = 5, sol = 0.7847, gel = 0.0208

        else:  # dt_max > 0
            if gel > 0:
                # L'eau_degelee est toujours nulle dans la version 2 d'HSAMI
                # car toute l'eau dégelée va dans l'état sol au lieu de
                # contribuer au ruissellement intermédiaire dans
                # hsami_meteo_apport

                sol, gel = degel_sol(duree, dt_max, sol, gel, neige_au_sol * 100)

        # Calcul de la température moyenne
        tmoy = (t_min + t_max) / 2
        tneige = tmoy

        # ========================================================
        # Évolution de la neige (een), du couvert et de la densité
        # ========================================================

        if neige_au_sol > 0 or neige > 0:
            # ------------------------------------------------------
            # Ajout de la précipitation neigeuse et estimation de la
            # densité du couvert de neige
            # ------------------------------------------------------

            # Densite relative de la précipitation neigeuse
            drel = calcul_densite_neige(tmoy) / rho_w
            # Ex1. : modules['een'] = 'mdj', i_z = 1, drel = 0.0500
            #                                i_z = 2, drel = 0.0500
            #                                i_z = 3, drel = 0.0500
            #        modules['een'] = 'alt', 0.0500 pour les 5 zones

            # Ajout de la précipitation neigeuse
            neige_au_sol = neige_au_sol + neige
            couvert_neige = couvert_neige + neige / drel

            # Densite du couvert de neige
            dennei = neige_au_sol / couvert_neige

            # --------------------------------------------------------
            # Ajustement du bilan énergétique selon l'eau retenue dans
            # le couvert de neige au pas de temps précédent
            # --------------------------------------------------------
            energie_neige = energie_neige + (fonte * rho_w * chaleur_latente_fusion)

            # -------------------------------------------------
            # Ajustement du bilan énergétique par l'ajout de la
            # précipitation neigeuse
            # -------------------------------------------------
            energie_neige = (
                energie_neige
                + neige * rho_w * capacite_thermique_massique_eau_solide * tmoy
            )
            # -------------------------------------------------------
            # Ajustement du bilan énergétique par la convection selon
            # la température de la neige
            # -------------------------------------------------------
            if tmoy < temperature_de_fonte[i_z]:
                # Estimation de la température de la neige
                tneige = energie_neige / (
                    neige_au_sol * capacite_thermique_massique_eau_solide * rho_w
                )

                # Estimation temporaire de la hauteur de neige
                if couvert_neige < 0.4:
                    hneige = 0.5 * couvert_neige
                else:
                    hneige = 0.2 + 0.25 * (couvert_neige - 0.4)

                # Estimation de l'erreur pour le calcul de la température de la neige
                alpha = conductivite_neige(dennei * rho_w) / (
                    dennei * rho_w * capacite_thermique_massique_eau_solide
                )
                erf = erf(hneige / (2 * np.sqrt(alpha * pdts)))
                # Ex1. : modules['een'] = 'mdj', i_z = 1, alpha = 3.2224e-07, erf = 0.5806
                #                             i_z = 2, alpha = 3.2492e-07, erf = 0.5843
                #                             i_z = 3, alpha = 3.2492e-07, erf = 0.5843
                #        modules['een'] = 'alt', i_z = 1, alpha = 3.5620e-07, erf = 0.6356
                #                             i_z = 2, alpha = 3.5824e-07, erf = 0.6308
                #                             i_z = 3, alpha = 3.6034e-07, erf = 0.6260
                #                             i_z = 4, alpha = 3.6199e-07, erf = 0.6201
                #                             i_z = 5, alpha = 3.6307e-07, erf = 0.6136

                # Température de la neige corrigée
                tneige = tmoy + (tneige - tmoy) * erf

                # Mise à jour de l'énergie contenue dans la neige selon sa température estimée
                energie_neige = (
                    tneige
                    * neige_au_sol
                    * rho_w
                    * capacite_thermique_massique_eau_solide
                )

            # --------------------------------------------------------
            # Ajustement du bilan énergétique selon les précipitations
            # pluvieuses sur le couvert de neige
            # --------------------------------------------------------
            neige_au_sol = neige_au_sol + pluie
            fonte = fonte + pluie
            energie_neige = energie_neige + pluie * rho_w * (
                chaleur_latente_fusion + capacite_thermique_massique_eau_liquide * tmoy
            )

            # -------------------------------------------------
            # Ajustement du bilan énergétique selon le gradient
            # géothermique
            # -------------------------------------------------
            energie_neige = (
                energie_neige + (taux_fonte_ns * duree) * rho_w * chaleur_latente_fusion
            )

            # --------------------------------------------------------
            # Ajustement du bilan énergétique selon la radiation et la
            # température moyenne
            # --------------------------------------------------------
            # Indice de radiation et albedo de la neige
            # if isfield(physio,'latitude') && isfield(physio,'i_orientation_bv') && isfield(physio,
            # 'pente_bv') && duree == 1. Si le pas de temps est inférieur à 24, il faudrait que
            # l'indice de radiation connaisse l'heure de la journée. à implanter éventuellement.

            if modules["radiation"] == "mdj":
                # Calcul d'un indice de radiation sophistiqué qui tient compte
                # de la pente du bassin et de l'orientation
                indice_radiation = indice_radiation(
                    jj,
                    physio["latitude"],
                    physio["i_orientation_bv"],
                    pas_de_temps,
                    physio["pente_bv"],
                )

            elif modules["radiation"] == "hsami":
                # Si les caractéristiques physiographiques du bassins ne sont
                # pas Args :  dans la fonction, l'indice de radiation est
                # calculé comme dans le hsami original
                indice_radiation = (1.15 - 0.4 * np.exp(-0.38 * derniere_neige)) * (
                    soleil / 0.52
                ) ** 0.33
                # Ex1. : modules['een'] = 'mdj', i_z = 1, indice_radiation = 0.8652
                #                                i_z = 2, indice_radiation = 0.8652
                #                                i_z = 3, indice_radiation = 0.8652
                #        modules['een'] = 'alt', méme chose que pour 'mdj'

            albedo_neige = albedo_een(
                albedo_neige, drel, neige_au_sol, neige, pas_de_temps, pluie, tneige
            )
            # Ex1. : modules['een'] = 'mdj', i_z = 1, albedo_neige = 0.7453
            #                                i_z = 2, albedo_neige = 0.7453
            #                                i_z = 3, albedo_neige = 0.7453
            #        modules['een'] = 'alt', 0.7453 pour les 5 zones

            # -----------------------------------------------------------
            # Détermination de la fonte potentielle si le couvert est mûr
            # -----------------------------------------------------------
            if (
                tmoy > temperature_de_fonte[i_z]
            ):  # Limitation: tmoy est le méme pour toutes les zones
                potentiel_fonte = (
                    taux_de_fonte[i_z]
                    * duree
                    * (tmoy - temperature_de_fonte[i_z])
                    * indice_radiation
                    * (1 - albedo_neige)
                )
            else:
                potentiel_fonte = 0

            # --------------------------------
            # Mise é jour du bilan énergétique
            # --------------------------------
            energie_neige = energie_neige + (
                potentiel_fonte * rho_w * chaleur_latente_fusion
            )

            # =======================
            # Calcul de la compaction
            # =======================

            # Hauteur du couvert nival et de sa densite aprés compaction.
            compaction = (
                couvert_neige
                * constante_tassement
                * duree
                * (1 - dennei / densite_maximale * 1000)
            )
            if compaction < 0:
                compaction = 0

            couvert_neige = couvert_neige - compaction
            dennei = (
                neige_au_sol / couvert_neige
            )  # cette valeur peut étre tres élevée quand de la pluie a été ajoutée à
            #  la neige_au_sol, la dennei est réajustée aprés.

            # Correction de la densité si elle dépasse la densité maximale (survient principalement lorsque de la pluie
            # a été ajoutée au couvert de neige)
            if dennei * rho_w > densite_maximale:
                dennei = densite_maximale / rho_w
                couvert_neige = (
                    neige_au_sol / dennei
                )  # couvert_neige est é densité max.

            # =======================================
            # Calcul de la fonte selon le mûrissement
            # =======================================
            if energie_neige > 0:  # Le couvert est mûr
                # ==================================
                # Le couvert est mûr, il peut fondre
                # ==================================

                # Estimation de la neige pouvant fondre selon son niveau
                # d'énergie
                potentiel_fonte = energie_neige / chaleur_latente_fusion / rho_w

                # ----------------------------------
                # Ajustement de l'eau de fonte selon
                # la capacité de retenue du couvert
                # ----------------------------------

                # La neige_fondue est déjé dans la neige_au_sol et fait déjé
                # partie de la fonte. La fonte est le maximum entre ces 2
                # valeurs.
                fonte = max(fonte, potentiel_fonte)

                # Par contre, il ne peut pas y avoir plus de fonte que de
                # neige_au_sol... Cela peut survenir dans le code é la premiére
                # neige é l'automne.
                fonte = min(fonte, neige_au_sol)

                # Eau en trop dans la neige
                eau_excedentaire = fonte - (capacite_retenue * dennei * neige_au_sol)
                if eau_excedentaire <= 0:
                    # la neige peut contenir toute l'eau de fonte
                    percolation = 0

                elif eau_excedentaire >= neige_au_sol:
                    # il n'y a plus de neige solide : tout le couvert s'écoule
                    percolation = neige_au_sol
                    neige_au_sol = 0
                    couvert_neige = 0
                    fonte = 0
                else:
                    # l'eau en trop devient ruissellement
                    percolation = eau_excedentaire
                    fonte = fonte - eau_excedentaire
                    neige_au_sol = neige_au_sol - eau_excedentaire
                    couvert_neige = couvert_neige - eau_excedentaire / dennei

                # -------------------------------
                # Ajustement du bilan énergétique
                # -------------------------------
                energie_neige = energie_neige - (
                    percolation * rho_w * chaleur_latente_fusion
                )

                # ===========================================
                # évaporation de l'eau contenue dans la neige
                # ===========================================
                # Ajustement pour un couvert mur aprés avoir vérifié la
                # capacité de retenue.
                # Aprés la fonte du couvert, ce dernier peut sublimer. La sublimation est calculée aprés la fonte
                # car l'énergie requise pour sublimer est plus grande que pour la fonte (source: Gaétan Roberge)

                # Le couvert a maintenant une capacité de retenue, l'eau qui est retenue peut s'évaporer au
                # taux hivernal selon la disponibilité.
                etr[1] = demande * efficacite_evapo_hiver  # m
                if (
                    fonte > etr[1]
                ):  # il y a assez d'eau retenue dans le couvert pour satisfaire la demande en eau de l'atmosphére.
                    fonte = fonte - etr[1]
                    neige_au_sol = neige_au_sol - etr[1]
                    couvert_neige = couvert_neige - etr[1] / dennei

                else:
                    etr[1] = fonte  # m
                    neige_au_sol = neige_au_sol - fonte
                    couvert_neige = couvert_neige - fonte / dennei
                    fonte = 0  # toute l'eau retenue est évaporée.

                    demande = 0  #

                    # Condition déplacée le 2016-08-12. Elle était avant la boucle précédente.
                    # é l'automne, lors de la formation du couvert, il
                    # peut y a des instabilités numériques en raison de
                    # l'estimation de la densité d'une quantité de
                    # neige infinitésimale. Dans ce cas le couvert de
                    # neige peut devenir négatif (genre -1x10-15) quand on soustrait fonte/dennei.
                    if couvert_neige < 0:
                        # Précaution ajoutée pour éviter la neige nulle. N'arrive
                        # jamais pour les bassins testés. Conditions mises au cas
                        # oé. (2016-08-11: C'est arrivé avec un scénario climatique dans le lot2)
                        couvert_neige = 0
                        neige_au_sol = 0

                    # -------------------------------
                    # Ajustement du bilan énergétique
                    # -------------------------------
                    energie_neige = energie_neige - (
                        etr[1] * rho_w * chaleur_latente_evaporation
                    )

            else:
                # ===============================================
                # Le couvert n'est pas mûr, il ne peut pas fondre
                # ===============================================

                percolation = 0
                fonte = 0  # Il n'y a pas d'eau de fonte dans un couvert (pas?) mûr.

                # Note: Le gel de la neige et la percolation de l'eau de fonte
                # ont été ajouté lorsque le couvert n'est pas mur. Ces
                # fonctions ne servaient é rien. La fonte ne dépasse jamais
                # la capacité de retenue quand le couvert est mûr. Et le gel de
                # la neige n'excéde jamais son gel potentiel, ce qui engendre
                # toujours une fonte nulle.

                # ------------------
                # Le couvert sublime
                # ------------------
                demande = demande * efficacite_evapo_hiver  # m
                if demande < neige_au_sol:
                    neige_au_sol = neige_au_sol - demande
                    etr[0] = demande
                else:  # demande_eau >= neige_au_sol: toute la neige disparaét
                    etr[0] = neige_au_sol
                    neige_au_sol = 0
                    couvert_neige = 0

                demande = 0
                # -------------------------------
                # Ajustement du bilan énergétique
                # -------------------------------
                energie_neige = energie_neige - (
                    etr[0] * rho_w * chaleur_latente_sublimation
                )

            # ===================================================
            # Ajustements pour éviter les instabilités numériques
            # ===================================================

            # Réinitialisation du couvert de neige si l'equivalent en
            # eau est presque nul.
            if neige_au_sol > 0 and neige_au_sol < 0.0001:
                # Méme si ce sont des poussiéres, il faut quand méme l'ajouter pour que le bilan ferme.
                percolation = percolation + neige_au_sol
                neige_au_sol = 0
                couvert_neige = 0
                energie_neige = 0
                fonte = 0
                dennei = 0

            # =======================================================
            # détermination de l'eau disponible pour le ruissellement
            # =======================================================
            eau_surface = percolation

            # -------------------------------------------------------------------------
            # Si toute la neige a fondue sur la derniére zone, on fait évoluer la glace
            # -------------------------------------------------------------------------
            if i_z == n:
                if neige_au_sol == 0:
                    for i_g in range(len(eeg)):
                        if eeg[i_g] > 0:
                            # Calcul de la température moyenne
                            tmoy_glace = (
                                meteo.reservoir[0] + meteo["reservoir"][0]
                            ) / 2

                            # Calcul du bilan d'énergie pour la glace
                            # -------------------------------------------------------
                            # Ajustement du bilan énergétique par la convection selon
                            # la température de la glace
                            # -------------------------------------------------------
                            if tmoy_glace < temperature_de_fonte[i_z]:
                                # Estimation de la température de la glace
                                tglace = energie_glace / (
                                    eeg[i_g]
                                    * capacite_thermique_massique_eau_solide
                                    * rho_w
                                )

                                # Estimation de l'erreur pour le calcul de la
                                # température de la glace
                                denglace = 0.917  # densité fixée à 0.917, glace normale à 0 degré

                                alpha = conductivite_glace / (
                                    denglace
                                    * rho_w
                                    * capacite_thermique_massique_eau_solide
                                )
                                erf = erf(
                                    (eeg[i_g] / denglace) / (2 * np.sqrt(alpha * pdts))
                                )

                                # Température de la glace corrigée
                                tglace = tmoy_glace + (tglace - tmoy_glace) * erf

                                # Mise à jour de l'énergie contenue dans la glace selon sa température estimée
                                energie_glace = (
                                    tglace
                                    * eeg[i_g]
                                    * rho_w
                                    * capacite_thermique_massique_eau_solide
                                )

                            # --------------------------------------------------------
                            # Ajustement du bilan énergétique selon la radiation et la
                            # température moyenne
                            # --------------------------------------------------------
                            indice_radiation = (
                                1.15 - 0.4 * np.exp(-0.38 * derniere_neige)
                            ) * (soleil / 0.52) ** 0.33
                            albedo_glace = 0.6

                            # -------------------------------------------------
                            # Ajustement du bilan énergétique selon le gradient
                            # géothermique
                            # -------------------------------------------------
                            energie_glace = (
                                energie_glace
                                + (taux_fonte_ns * duree)
                                * rho_w
                                * chaleur_latente_fusion
                            )

                            # -----------------------------------------------------------
                            # Détermination de la fonte potentielle si le couvert est mûr
                            # -----------------------------------------------------------
                            # Les taux de fonte de la neige sont
                            # multipliés par 1.5 pour la glace selon Braithwaite
                            # (1995) et Singh et al (1999).
                            if tmoy_glace > temperature_de_fonte[i_z]:
                                potentiel_fonte = (
                                    1.5
                                    * taux_de_fonte[i_z]
                                    * duree
                                    * (tmoy_glace - temperature_de_fonte[i_z])
                                    * indice_radiation
                                    * (1 - albedo_glace)
                                )
                            else:
                                potentiel_fonte = 0

                            # --------------------------------
                            # Mise à jour du bilan énergétique
                            # --------------------------------
                            energie_glace = energie_glace + (
                                potentiel_fonte * rho_w * chaleur_latente_fusion
                            )

                            # =======================================
                            # Calcul de la fonte selon le mûrissement
                            # =======================================
                            if energie_glace > 0:  # La glace est mûre
                                # Estimation de la glace pouvant fondre selon son niveau
                                # d'énergie
                                potentiel_fonte = (
                                    energie_glace / chaleur_latente_fusion / rho_w
                                )

                                # La glace n'a aucune capacité de rétention de
                                # l'eau, alors la fonte dépend de la disponibilité
                                # de la glace
                                if potentiel_fonte >= eeg[i_g]:
                                    fonte_glace = eeg[i_g]
                                    eeg[i_g] = 0
                                else:
                                    fonte_glace = potentiel_fonte
                                    eeg[i_g] = eeg[i_g] - potentiel_fonte

                                apport_vertical[4] = apport_vertical[4] + fonte_glace

                                # -------------------------------
                                # Ajustement du bilan énergétique
                                # -------------------------------
                                energie_glace = energie_glace - (
                                    fonte_glace * rho_w * chaleur_latente_fusion
                                )

        else:  # S'il n'y a ni neige au sol ni précipitations neigeuses sur ce pas de temps.
            eau_surface = pluie
            energie_neige = 0
            neige_au_sol = 0
            couvert_neige = 0
            fonte = 0
            dennei = 0
            albedo_neige = 0.15

            # Fonte de la glace
            if i_z == n:
                for i_g in range(len(eeg)):
                    if eeg[i_g] > 0:
                        # Calcul de la température moyenne
                        tmoy_glace = (meteo["reservoir"][0] + meteo["reservoir"][1]) / 2

                        # Calcul du bilan d'énergie pour la glace
                        # -------------------------------------------------------
                        # Ajustement du bilan énergétique par la convection selon
                        # la température de la glace
                        # -------------------------------------------------------
                        if tmoy_glace < temperature_de_fonte[i_z]:
                            # Estimation de la température de la glace
                            tglace = energie_glace / (
                                eeg[i_g]
                                * capacite_thermique_massique_eau_solide
                                * rho_w
                            )

                            # Estimation de l'erreur pour le calcul de la
                            # température de la glace
                            denglace = (
                                0.917  # densité fixée à 0.917, glace normale à 0 degré
                            )
                            alpha = conductivite_glace / (
                                denglace
                                * rho_w
                                * capacite_thermique_massique_eau_solide
                            )
                            erf = erf(
                                (eeg[i_g] / denglace) / (2 * np.sqrt(alpha * pdts))
                            )

                            # Température de la glace corrigée
                            tglace = tmoy_glace + (tglace - tmoy_glace) * erf

                            # Mise é jour de l'énergie contenue dans la glace selon sa température estimée
                            energie_glace = (
                                tglace
                                * eeg[i_g]
                                * rho_w
                                * capacite_thermique_massique_eau_solide
                            )

                        # --------------------------------------------------------
                        # Ajustement du bilan énergétique selon la radiation et la
                        # température moyenne
                        # --------------------------------------------------------
                        indice_radiation = (
                            1.15 - 0.4 * np.exp(-0.38 * derniere_neige)
                        ) * (soleil / 0.52) ** 0.33
                        albedo_glace = 0.6

                        # -------------------------------------------------
                        # Ajustement du bilan énergétique selon le gradient
                        # géothermique
                        # -------------------------------------------------
                        energie_glace = (
                            energie_glace
                            + (taux_fonte_ns * duree) * rho_w * chaleur_latente_fusion
                        )

                        # -----------------------------------------------------------
                        # Détermination de la fonte potentielle si le couvert est mûr
                        # -----------------------------------------------------------
                        # Les taux de fonte de la neige sont
                        # multipliés par 1.5 pour la glace selon Brathwaite
                        # (1995) et Singh et al (1999).
                        if tmoy_glace > temperature_de_fonte[i_z]:
                            potentiel_fonte = (
                                1.5
                                * taux_de_fonte[i_z]
                                * duree
                                * (tmoy_glace - temperature_de_fonte[i_z])
                                * indice_radiation
                                * (1 - albedo_glace)
                            )
                        else:
                            potentiel_fonte = 0

                        # --------------------------------
                        # Mise é jour du bilan énergétique
                        # --------------------------------
                        energie_glace = energie_glace + (
                            potentiel_fonte * rho_w * chaleur_latente_fusion
                        )

                        # =======================================
                        # Calcul de la fonte selon le mûrissement
                        # =======================================
                        if energie_glace > 0:  # La glace est mûre
                            # Estimation de la glace pouvant fondre selon son niveau
                            # d'énergie
                            potentiel_fonte = (
                                energie_glace / chaleur_latente_fusion / rho_w
                            )

                            # La glace n'a aucune capcaité de rétention de
                            # l'eau, alors la fonte dépend de la disponibilité
                            # de la glace
                            if potentiel_fonte >= eeg[i_g]:
                                fonte_glace = eeg[i_g]
                                eeg[i_g] = 0
                            else:
                                fonte_glace = potentiel_fonte
                                eeg[i_g] = eeg[i_g] - potentiel_fonte

                            apport_vertical[4] = apport_vertical[4] + fonte_glace

                            # -------------------------------
                            # Ajustement du bilan énergétique
                            # -------------------------------
                            energie_glace = energie_glace - (
                                fonte_glace * rho_w * chaleur_latente_fusion
                            )

        # ====================================================
        # Récupération des variables simulées pour chaque zone
        # ====================================================

        # -------------------------------------------------
        # Variables devant étre moyennées au bassin versant
        # -------------------------------------------------
        #                                              'mdj'       'alt'
        eau_surface_zones[i_z] = eau_surface  # Ex1.: [0, 0, 0]      idem
        sublimation[i_z] = etr[1]  # Ex1.: [0, 0, 0]      idem
        evapo_eau_neige[i_z] = etr[2]  # Ex1.: [0, 0, 0]      idem
        demande_eau[i_z] = demande  # Ex1.: [0, 0, 0]      idem
        # --------------------------------------------
        # Variables propres é chaque zone d'occupation
        # --------------------------------------------
        #                                                               'mdj'                                 'alt'
        etat[modules["een"]]["neige_au_sol"][
            i_z
        ] = neige_au_sol  # Ex1.: [0.0635 0.0655 0.0655]           [0.1044 0.1044 0.1044 0.1035 0.1019]
        etat[modules["een"]]["couvert_neige"][
            i_z
        ] = couvert_neige  # Ex1.: [0.3566 0.3612 0.3612]           [0.4725 0.4667 0.4609 0.4528 0.4429]
        etat[modules["een"]]["densite_neige"][
            i_z
        ] = dennei  # Ex1.: [0.1780 0.1813 0.1813]           [0.2210 0.2236 0.2264 0.2286 0.2300]
        etat[modules["een"]]["fonte"][
            i_z
        ] = fonte  # Ex1.: [0 0 0]                          [0 0 0 0 0]
        etat[modules["een"]]["albedo_neige"][
            i_z
        ] = albedo_neige  # Ex1.: [0.7453 0.7453 0.7453]           [0.7453 0.7453 0.7453 0.7453 0.7453]
        etat[modules["een"]]["energie_neige"][
            i_z
        ] = energie_neige  # Ex1.: [-9.85e+05 -1.0128e+06 -1.08e+06] [-1.88e+06 -1.82e+06 -1.76e+06 -1.69e+06 -1.63e+06]
        etat[modules["een"]]["sol"][
            i_z
        ] = sol  # (en cm)            # Ex1.: [2.4932 2.4888 2.4888]           [0.7829 0.7835 0.7840 0.7844 0.7847]
        etat[modules["een"]]["gel"][
            i_z
        ] = gel  # (en cm)            # Ex1.: [0.0351 0.0395 0.0395]            [0.0226 0.0220 0.0215 0.0211 0.0208]

    # ----------------------------------------------------------------------
    # Moyennes pondérées au bassin selon les proportions occupées par chaque
    # zone
    # ----------------------------------------------------------------------
    #                                                                                'mdj'    'alt'
    eau_surface = np.sum(eau_surface_zones[:] * occupation[:])  # Ex1.:  0        0
    etr[0] = np.sum(sublimation[:] * occupation[:])  # Ex1.:  0        0
    etr[1] = np.sum(evapo_eau_neige[:] * occupation[:])  # Ex1.:  0        0
    demande_eau = np.sum(demande_eau[:] * occupation[:])  # Ex1.:  0        0

    neige_au_sol = np.sum(
        [a * b for a, b in zip(etat[modules["een"]]["neige_au_sol"][0:n], occupation)]
    )  # Ex1.:  0.0653   0.1023
    fonte = np.sum(
        [a * b for a, b in zip(etat[modules["een"]]["fonte"][0:n], occupation)]
    )  # Ex1.:  0        0

    sol = np.sum(
        [a * b for a, b in zip(etat[modules["een"]]["sol"][0:n], occupation)]
    )  # Ex1.:  2.4892   0.7846
    gel = np.sum(
        [a * b for a, b in zip(etat[modules["een"]]["gel"][0:n], occupation)]
    )  # Ex1.:  0.0392   0.0209

    # ----------------------------------------------------------------------
    # Ajustements des unités pour assurer une cohérence avec HSAMI pour les
    # variables utilisées dans d'autres fonctions du code
    # ----------------------------------------------------------------------
    eau_surface = eau_surface * 100  # m-->cm
    demande_eau = demande_eau * 100
    apport_vertical = apport_vertical * 100  # m-->cm
    neige_au_sol = neige_au_sol * 100  # m-->cm
    fonte = fonte * 100  # m-->cm
    etr = etr * 100  # m-->cm
    eeg = eeg * 100  # m-->cm

    etat["neige_au_sol"] = neige_au_sol
    etat["fonte"] = fonte
    etat["derniere_neige"] = derniere_neige
    etat["eeg"] = eeg
    etat["gel"] = gel
    etat["sol"][0] = sol

    return eau_surface, demande_eau, etat, etr, apport_vertical


# ==========================
# FIN DU PROGRAMME PRINCIPAL
# ==========================


def gel_sol(duree, dt_max, sol_min, sol, gel, neige_au_sol):
    """
    Gel du sol en fonction de la température maximale.

    Parameters
    ----------
    duree : float
        Nombre de pas de temps par jour.
    dt_max : float
        Températuret max - température de fonte.
    sol_min : float
        Point de flétrissement permanent du sol.
    sol : float
        Eau dans le sol.
    gel : float
        Eeau gelée dans le sol.
    neige_au_sol : float
        Neige au sol.

    Returns
    -------
    sol : float
        Eeau dans le sol.
    gel : float
        Eau gelée dans le sol.
    """
    # Gel potentiel
    delta = -(2.54**2) * 0.0036 * dt_max / (2.54 + gel + neige_au_sol) * duree

    # S'il y a assez d'eau libre dans le sol, on y puise l'eau gelee
    if sol - delta > sol_min:
        sol = sol - delta
        gel = gel + delta

        # S'il n'y a pas assez d'eau dans le sol pour combler le potentiel de gel,
        # on prend tout de méme l'eau disponible pour geler au lieu de ne rien geler du tout.
    else:
        gel = gel + (sol - sol_min)
        sol = sol_min

    return sol, gel


def degel_sol(duree, dt_max, sol, gel, neige_au_sol):
    """
    Degel de l'eau gelée dans le sol par temps doux.

    Parameters
    ----------
    duree : float
        Nombre de pas de temps par jour.
    dt_max : float
        Températuret max - température de fonte.
    sol : float
        Eau dans le sol.
    gel : float
        Eeau gelée dans le sol.
    neige_au_sol : float
        Neige au sol.

    Returns
    -------
    sol : float
        Eeau dans le sol.
    gel : float
        Eau gelée dans le sol.
    """
    # effet isolant de la neige_au_sol et du sol gelé (ralentit le dégel)
    isolation = 2.54 + gel + neige_au_sol

    # effet potentiel de la température douce
    infiltration = 2.54**2 * 0.072 * (dt_max + 40 / 9) / isolation * duree
    ruissellement = 2.54**2 * 0.036 * dt_max / isolation * duree

    infiltration = infiltration + ruissellement
    ruissellement = 0

    # effet réel en fonction de la réserve d'eau gelée
    if infiltration + ruissellement >= gel:
        # le sol est complétement dégelé: tout ce qui reste s'infiltre
        sol = sol + gel
        gel = 0
    else:
        # le sol est partiellement dégelé: une partie ruisselle, une partie s'infiltre
        sol = sol + infiltration
        gel = gel - infiltration - ruissellement

    return sol, gel


def gel_neige(duree, dt_max, neige_au_sol, fonte, fonte_totale):
    """
    Gel de la neige au sol en fonction de la température maximale.

    Parameters
    ----------
    duree : float
        Nombre de pas de temps par jour.
    dt_max : float
        Températuret max - température de fonte.
    neige_au_sol : float
        Neige au sol.
    fonte : float
        Eau liquide stockée dans la neige.
    fonte_totale : float
        Fonte totale.

    Returns
    -------
    fonte : float
        Eau liquide stockée dans la neige.
    fonte_totale : float
        Total de la fonte de neige pendant l'hiver.
    """
    # Gel potentiel
    delta = -(2.54**2) * 0.072 * dt_max / neige_au_sol * duree

    if fonte < delta:
        fonte = 0
        fonte_totale = 0

    else:
        fonte = fonte - delta  # Réduction de la fonte
        fonte_totale = fonte_totale - delta

    return fonte, fonte_totale


def percolation_eau_fonte(neige_au_sol, neige_au_sol_totale, fonte, fonte_totale):
    """
    Calcul la ercolation de l'eau de fonte dans la neige.

    Parameters
    ----------
    neige_au_sol : float
        Equivalent en eau de la neige au sol incluant l'eau de fonte.
    neige_au_sol_totale : float
        Total des chutes de neige pendant l'hiver.
    fonte : float
        Eau liquide stockée dans la neige.
    fonte_totale : float
        Total de la fonte de neige pendant l'hiver.

    Returns
    -------
    lame : float
        Eau qui s'écoule.
    neige_au_sol : float
        Neige au sol.
    neige_au_sol_totale : float
        Total des chutes de neige pendant l'hiver.
    fonte : float
        Eau liquide stockée dans la neige.
    fonte_totale : float
        Total de la fonte de neige pendant l'hiver.
    """
    # eau en trop dans la neige
    delta = (fonte - 0.1 * neige_au_sol) / 0.9

    if delta <= 0:
        # la neige séche peut contenir toute l'eau de fonte
        lame = 0
    elif delta >= neige_au_sol:
        # il n'y a plus de neige séche: tout le couvert s'écoule
        lame = neige_au_sol
        neige_au_sol = 0
        neige_au_sol_totale = 0
    else:
        # l'eau en trop s'écoule
        lame = delta
        fonte = fonte - delta
        neige_au_sol = neige_au_sol - delta

    return lame, neige_au_sol, neige_au_sol_totale, fonte, fonte_totale


def conductivite_neige(densite):
    """
    Calcul de la conductivite de la neige.

    Parameters
    ----------
    densite : float
        Densite de la neige.

    Returns
    -------
    float
        Conductivite de la neige.
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

    conductivite = d0 * p0 + d1 * p1 + d2 * p2 + d3 * p3 + d4 * p4

    return conductivite


def erf(x):
    """
    Approximation rationnelle.

    Parameters
    ----------
    x : float
        Arguement.

    Returns
    -------
    float
        Valeur la fonction.
    """
    if x < 0:
        raise Exception("Désolé, aucun nombre en dessous de zéro")

    t = 1 / (1 + 0.47047 * x)
    valeur = 1 - (0.3480242 * t - 0.0958798 * t * t + 0.7478556 * t * t * t) * np.exp(
        -x * x
    )

    return valeur


def indice_radiation(jour, latitude, i_orientation_bv, pas_de_temps, pente):
    """
    Calcul de l'indice de radiation pour une surface.

    Parameters
    ----------
    jour : int
        Jour julien.
    latitude : float
        Latitude du bassin versant.
    i_orientation_bv : float
        Indice d'orientantion du bassin versant.
    pas_de_temps : int
        Pas de temps.
    pente : float
        Pente du bassin versant.

    Returns
    -------
    float
        Indice de radiation.
    """
    heure = 24

    tan_orientation = [
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4,
        np.pi,
        5 * np.pi / 4,
        3 * np.pi / 2,
        7 * np.pi / 4,
        2 * np.pi,
    ]  # E, NE, N, NO, O, SO, S, SE
    orientation = tan_orientation[i_orientation_bv - 1]

    orientation = np.float32(orientation)

    i0 = 1376
    rad1 = 180 / np.pi
    deg1 = 58.1313429644  # "un jour en degré"
    w = 15 / rad1
    theta = latitude  # La latitude est entrée en radian via le call à HSAMI # latitude / rad1
    # jour = jour julien
    # heure = selon le pas de temps
    k = np.arctan(pente)
    h = np.mod(495 - orientation * 45, 360) / rad1

    ce1 = (
        np.arcsin(np.sin(k) * np.cos(h) * np.cos(theta) + np.cos(k) * np.sin(theta))
        * rad1
    )
    ce0 = (
        np.arctan(
            np.sin(h)
            * np.sin(k)
            / (np.cos(k) * np.cos(theta) - np.cos(h) * np.sin(k) * np.sin(theta))
        )
        * rad1
    )

    theta1 = ce1 / rad1
    alpha = ce0 / rad1

    # calcul du vecteur radian
    e2 = (1 - 0.01673 * np.cos((jour - 4) / deg1)) ** 2
    i_e2 = i0 / e2

    # calcul de la declinaison
    decli = 0.410152374218 * np.sin((jour - 80.25) / deg1)

    # demi-duree du jour sur une surface horizontale
    tampon = -np.tan(theta) * np.tan(decli)
    if tampon > 1:
        duree_hor = 0

    elif tampon < -1.0:
        duree_hor = 12.0
    else:
        duree_hor = np.arccos(tampon) / w

    # duree du jour sur une surface en pente
    tampon = -np.tan(theta1) * np.tan(decli)
    if tampon > 1:
        duree_pte = 0

    elif tampon < -1:
        duree_pte = 12

    else:
        duree_pte = np.arccos(tampon) / w

    # lever et coucher du soleil pour une surface en pente
    t1_pte = -duree_pte - alpha / w
    t2_pte = duree_pte - alpha / w

    if t1_pte < -duree_hor:
        t1_pte = -duree_hor

    if t2_pte > duree_hor:
        t2_pte = duree_hor

    # Si le pas de temps de la simulation (en heure) est inferieur a 24h
    # alors il ne suffit pas de calculer pour une surface en pente la duree du
    # jour, le leve et le couche du soleil. Mais il faut inclure seulement les
    # heures qu'on simule.

    t1_pte_sim = t1_pte
    t2_pte_sim = t2_pte
    t1_hor_sim = -duree_hor
    t2_hor_sim = duree_hor

    if pas_de_temps < 24:
        t1 = heure - 12
        t2 = heure + pas_de_temps - 12

        t1_pte_sim = max(t1, t1_pte)
        t2_pte_sim = min(t2, t2_pte)

        t1_hor_sim = max(t1, -duree_hor)
        t2_hor_sim = min(t2, duree_hor)

    # calcul de l'ensoleillement d'une surface horizontale
    if t1_hor_sim > t2_hor_sim:
        i_j1 = 0
    else:
        i_j1 = (
            3600
            * i_e2
            * (
                (t2_hor_sim - t1_hor_sim) * np.sin(theta) * np.sin(decli)
                + 1
                / w
                * np.cos(theta)
                * np.cos(decli)
                * (np.sin(w * t2_hor_sim) - np.sin(w * t1_hor_sim))
            )
        )

    # calcul de l'ensoleillement d'une surface en pente
    if t1_pte_sim > t2_pte_sim:
        i_j2 = 0
    else:
        i_j2 = (
            3600
            * i_e2
            * (
                (t2_pte_sim - t1_pte_sim) * np.sin(theta1) * np.sin(decli)
                + 1
                / w
                * np.cos(theta1)
                * np.cos(decli)
                * (np.sin(w * t2_pte_sim + alpha) - np.sin(w * t1_pte_sim + alpha))
            )
        )

    if i_j1 != 0:
        indice_radiation = abs(i_j2 / i_j1)
    else:
        indice_radiation = 1

    return indice_radiation


def albedo_een(albedo, drel, een, neige, pas_de_temps, pluie, tneige):
    """
    Calculer l'albedo de l'EEN.

    Parameters
    ----------
    albedo : float
        Albedo.
    drel : float
        Densite relative de la neige.
    een : float
        Équivalent en eau de la neige.
    neige : float
        Neige au sol.
    pas_de_temps : int
        Pas de temps.
    pluie : float
        Précipitations liquides.
    tneige : float
        Température de la neige.

    Returns
    -------
    float
        Albedo d'een.
    """
    eq_neige = neige * 1000
    st_neige = (een - neige / drel) * 1000

    if pluie > 0 or tneige >= 0:
        liquide = 1

    else:
        liquide = 0

    if st_neige > 0:  # // s'il y a deja de la neige au sol
        alb_t_plus_1 = (1 - np.exp(-0.5 * eq_neige)) * 0.8 + (
            1 - (1 - np.exp(-0.5 * eq_neige))
        ) * (0.5 + (albedo - 0.5) * np.exp(-0.2 * pas_de_temps / 24.0 * (1 + liquide)))

        if albedo < 0.5:
            beta2 = 0.2
        else:
            beta2 = 0.2 + (albedo - 0.5)

        albedo = (1 - np.exp(-beta2 * st_neige)) * alb_t_plus_1 + (
            1 - (1 - np.exp(-beta2 * st_neige))
        ) * 0.15

    else:
        albedo = (1 - np.exp(-0.5 * eq_neige)) * 0.8 + (
            1 - (1 - np.exp(-0.5 * eq_neige))
        ) * 0.15

    return albedo


def calcul_densite_neige(temperature):
    """
    Calculer la densite de la neige.

    Parameters
    ----------
    temperature : float
        Témperature en deg C.

    Returns
    -------
    float
        Densite de la neige.
    """
    if temperature < -17:
        densite = 50
    elif temperature > 0:
        densite = 150
    else:
        densite = 151 + 10.63 * temperature + 0.2767 * temperature**2

    return densite


def pluie_neige(tmin, tmax, prec):
    """
    Séparation de la précipitation en pluie et neige.

    Parameters
    ----------
    tmin : float
        Température minimale.
    tmax : float
        Température maximale.
    prec : float
        Précipitations.

    Returns
    -------
    pluie : float
        Pluie.
    neige float
        Neige.

    Notes
    -----
    Pluie_neige(tmin,tmax,prec) sépare la précipitation en pluie et neige
    selon l'algorithme suivant:
    Lorsque la valeur moyenne de tmin et tmax est inférieure à -2 deg C,
    la précipitation est complétement transformée en neige.

    Lorsque la valeur moyenne de la température est dans [-2,2]
    deg C, la précipitation est transformée en neige et pluie dans
    une proportion qui dépend linéairement de cette température i.e
    neige = alpha*prec et pluie = (1-alpha)*prec avec alpha = 0 à -2
    deg C et alpha = 1 à +2 deg C.

    Lorsque la valeur moyenne de tmin et tmax est supérieure à +2 deg
    C, la précipitation est complétement transformée en pluie.
    """
    if isinstance(prec, float):
        # Températures moyennes
        tmoy = (tmin + tmax) / 2

        # Proportion lpuie / neige
        alpha = (tmoy + 2) / 4
        pluie = alpha * prec
        neige = (1 - alpha) * prec

    elif isinstance(prec, list) | isinstance(prec, np.ndarray):
        # Températures moyennes
        tmoy = np.array([(tmin + tmax) / 2])

        # Indices entre -2 et 2 deg C
        indbetween = np.where((tmoy >= -2) & (tmoy <= 2))[0]
        indpluie = np.where(tmoy > 2)[0]
        indneige = np.where(tmoy < -2)[0]

        # Initialisation des vecteurs de sortie à 0
        prec = np.array(prec)
        pluie = np.zeros_like(prec)
        neige = np.zeros_like(prec)

        # écriture
        pluie[indpluie] = prec[indpluie]
        neige[indneige] = prec[indneige]
        alpha = (tmoy[indbetween] + 2) / 4
        pluie[indbetween] = alpha * prec[indbetween]
        neige[indbetween] = (1 - alpha) * prec[indbetween]

    return pluie, neige
