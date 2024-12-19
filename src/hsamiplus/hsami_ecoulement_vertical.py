"""The function simulates the infiltration in HSAMI+ model."""

from __future__ import annotations

import numpy as np
from scipy import optimize


def hsami_ecoulement_vertical(
    nb_pas,
    param,
    etat,
    offre,
    demande,
    modules,
    ruissellement_surface,
    apport_vertical,
    etr,
):
    """
    Écoulement vertical.

    Parameters
    ----------
    nb_pas : float
        Nombre de pas de temps.
    param : list
        Paramètres pour la simulation.
    etat : dict
        États du bassin versants et du réservoir.
    offre : float
        Quantité d'eau disponible pour l'évaporation et le ruissellement (cm).
    demande : float
        Demande évaporative de l'atmosphére (cm).
    modules : dict
        Les modules pour la simulation.
    ruissellement_surface : float
        Ruissellement (cm).
    apport_vertical : list
        Lames d'eau verticales (voir hsami_interception).
    etr : list
        Composantes de l'évapotranspiration (voir hsami_interception).

    Returns
    -------
    apport : list
        Lames d'eau verticales (cm, voir hsami_interception).
    etat : dict
        États du bassin versants et du réservoir.
    etr : list
        Composantes de l'évapotranspiration (cm, voir hsami_interception).
    """
    if modules["sol"] == "3couches":
        [apport, etat, etr] = ecoulement_3couches(
            nb_pas,
            param,
            etat,
            offre,
            demande,
            modules,
            ruissellement_surface,
            apport_vertical,
            etr,
        )

    elif modules["sol"] == "hsami":
        # ==============
        # INITIALISATION
        # ==============

        # -----------------------------
        # Identification des paramétres
        # -----------------------------
        # Niveaux minimums et maximums des réserves d'eau dans le sol
        sol_min = param[11]  # minimum zone non-saturée (cm)
        sol_max = param[12]  # maximum zone non-saturée (cm)
        nappe_max = param[13]  # maximum zone saturée (cm)

        # Taux de vidange des réserves (de 0 à 1)
        portion_ruissellement_surface = param[14]
        portion_ruissellement_sol_max = param[15]
        taux_vidange_sol_min = param[16] / nb_pas
        taux_vidange_nappe = param[17] / nb_pas

        # -----------------------------------
        # Identification des variables d'état
        # -----------------------------------
        # Niveau des réserves (cm)
        gel = etat["gel"]  # eau gelée dans le sol
        sol = etat["sol"][0]  # réserve d'eau dans la zone non-saturée # CG 2020-06-09
        nappe = etat["nappe"]  # niveau de la nappe phréatique
        neige_au_sol = etat["neige_au_sol"]

        # ===================
        # Ecoulement vertical
        # ===================
        apport = apport_vertical.copy()

        # -----------------------------------------------
        # Effet de la demande en eau (évapotranspiration)
        # -----------------------------------------------
        ecart_offre_demande = offre - demande

        if ecart_offre_demande > 0:
            evapo = demande
            offre = offre - demande

            # Calcul de l'infiltration
            if modules["infiltration"] == "green_ampt":
                ks = 10 ** param[34]
                psi = param[25]
                inf_potentielle, apport[2] = green_ampt(
                    offre, ks, psi, sol_max, sol, nb_pas, gel, neige_au_sol
                )
                # Ex.: inf_potentielle = 0.0917
                #      apport[2] = 0

            elif modules["infiltration"] == "hsami":
                # Si l'offre est plus forte que la demande, une partie de la différence
                # s'écoule, le reste est stocké
                inf_potentielle = offre
                apport[2] = ruissellement_surface
                # Ex.: inf_potentielle = 0.0813
                #      apport[2] = 0.0103

            elif modules["infiltration"] == "scs_cn":
                cn = param[23]
                inf_potentielle, apport[2] = scs_cn(offre, cn)
                # Ex.: inf_potentielle = 0
                #      apport[2] = 0.0917

            # Séparation de l'infiltration entre le sol et l'hydrogramme
            # intermédiaire

            apport[1] = inf_potentielle * portion_ruissellement_surface
            infiltration = inf_potentielle * (1 - portion_ruissellement_surface)
            sol = sol + infiltration
            pompage = 0

        else:
            # Si la demande est plus forte que l'offre, les racines vont prélever de l'eau dans le sol
            # de faéon plus ou moins efficace en fonction de l'humidité du sol
            # Il n'y a alors aucune infiltration
            evapo = offre
            pompage = min(sol - sol_min, -sol / sol_max * ecart_offre_demande)
            # Ex.: modules['infiltration'] = 'hsami'     , pompage = 0.0524
            #      modules['infiltration'] = 'green_ampt', pompage = 0.0582
            #      modules['infiltration'] = 'scs_cn',     pompage = 0.0118
            #
            sol = sol - pompage

            if modules["infiltration"] == "hsami":
                # Avec la formulation initiale du ruissellement dans
                # hsami, il faut passer le ruissellement en surface
                # méme si la demande est plus forte que l'offre.
                apport[2] = ruissellement_surface

        # ----------------------------------
        # Vidange et débordement de la nappe
        # ----------------------------------
        apport, nappe, sol = vidange_nappe(
            apport, nappe, taux_vidange_nappe, nappe_max, nb_pas, modules, param, sol
        )
        # Ex.: modules['infiltration'] = 'hsami',       apport = [0.0548; 0.0203; 0.0103; 0.0917; 0]
        #                                            nappe = 6.7919
        #                                            sol = 6.1920
        #      modules['infiltration']  = 'green_ampt', apport = [0.0568; 0; 0; -0.0831; 0]
        #                                            nappe = 7.0433
        #                                            sol = 6.9452
        #      modules['infiltration']  = 'scs_cn',     apport = [0.0173; 0; 0; -0.0831; 0]
        #                                            nappe = 2.1444
        #                                            sol = 1.4055

        # ----------------------------------
        # Debordement de la zone non-saturee
        # ----------------------------------
        # Si la réserve non-saturée déborde, une partie ruisselle, une partie s'en va dans la nappe
        # Modification pour s'assurer que sol+gel < sol_max
        debordement_sol = sol + gel - sol_max

        if debordement_sol > 0:
            # Valeur par défaut en absence de modules.inter
            apport[1] = apport[1] + debordement_sol * portion_ruissellement_sol_max
            nappe = nappe + debordement_sol * (1 - portion_ruissellement_sol_max)
            sol = sol - debordement_sol

            if sol < 0:
                gel = gel + sol
                sol = 0

        # --------------------------
        # Infiltration vers la nappe
        # --------------------------
        if sol > sol_min:
            sol_vers_nappe = (sol - sol_min) * taux_vidange_sol_min
            nappe = nappe + sol_vers_nappe
            sol = sol - sol_vers_nappe

        # ====================
        # Sauvegarde de l'état
        # ====================
        etat["gel"] = gel
        etat["sol"][0] = sol
        etat["nappe"] = nappe
        etr[2] = evapo
        etr[3] = pompage

    return apport, etat, etr


# -----------------------------
# FIN DE LA FONCTION PRINCIPALE
# -----------------------------


# Modélisation de 3 couches de sol
def ecoulement_3couches(
    nb_pas,
    param,
    etat,
    offre,
    demande,
    modules,
    ruissellement_surface,
    apport_vertical,
    etr,
):
    """
    Calcule l'écoulement vertical dans un système à trois couches de sol.

    Parameters
    ----------
    nb_pas : float
        Nombre de pas de temps.
    param : list
        Paramètres pour la simulation.
    etat :  dict
        États du bassin versants et du réservoir.
    offre : float
        L'offre en eau disponible.
    demande : float
        La demande en eau.
    modules : dict
        Les modules pour la simulation.
    ruissellement_surface : float
        Le ruissellement en surface.
    apport_vertical : list
        Liste des apports verticaux.
    etr : list
        Liste des évapotranspirations.

    Returns
    -------
    apport : list
        Liste d'apport.
    etat :  dict
        États du bassin versants et du réservoir.
    etr : list
        Évapotranspiration.
    """
    # ==============
    # INITIALISATION
    # ==============
    # -----------------------------------
    # Identification des variables d'état
    # -----------------------------------
    sol = [
        etat["sol"][0],
        etat["sol"][1],
        etat["nappe"],
    ]  # La nappe est ajoutée au bas de la colonne de sol pour simplifier les calculs
    gel = etat["gel"]
    neige_au_sol = etat["neige_au_sol"]

    # -----------------------------
    # Identification des paramétres
    # -----------------------------
    b = [param[36], param[37]]  # pore-size distribution index (adim.)
    z = [param[39], param[40]]  # épaisseur des couches (cm)
    cc = [param[42], param[43]]  # capacité au champ (cm/cm)
    n = [param[44], param[45]]  # porosité totale (cm/cm)
    ks = [10 ** param[24], 10 ** param[38]]  # cond. hyd. sat. (cm/j)
    pfp = param[41]  # point de flétrissement permanent (cm/cm)
    nappe_max = param[13]  # quantité d'eau maximale dans la nappe (cm)
    portion_ruissellement_surface = param[
        14
    ]  # séparation du ruissellement hypodermique
    taux_vidange_nappe = param[17] / nb_pas  # taux de vidange de la nappe
    c = 2.0 * np.array(b) + 3  # pore-disconnectedness index (adim.)

    # Calcul de sol_max à partir des porosités et des épaisseurs de couches
    sol_max = [n[0] * z[0], n[1] * z[1], nappe_max]

    # Calcul de sol_min à partir des capacités au champ et épaisseurs
    sol_min = [cc[0] * z[0], cc[1] * z[1]]

    # ===================
    # Ecoulement vertical
    # ===================
    apport = apport_vertical.copy()
    ecart_offre_demande = offre - demande

    if ecart_offre_demande > 0:
        evapo = demande
        offre = offre - demande

        # Calcul de l'infiltration potentielle
        # ------------------------------------
        if modules["infiltration"] == "green_ampt":
            psi = param[25]
            inf_potentielle, apport[2] = green_ampt(
                offre,
                param[34],
                psi,
                sol_max[0],
                sol[0],
                nb_pas,
                gel,
                neige_au_sol,
                n[0],
            )

        elif modules["infiltration"] == "hsami":
            # Si l'offre est plus forte que la demande, une partie de la différence
            # s'écoule, le reste est stocké
            inf_potentielle = ecart_offre_demande
            apport[2] = ruissellement_surface

        elif modules["infiltration"] == "scs_cn":
            cn = param[23]
            inf_potentielle, apport[:] = scs_cn(offre, cn)

        pompage = 0

    else:
        # Si la demande est plus forte que l'offre, les racines vont prélever de l'eau dans le sol
        # de faéon plus ou moins efficace en fonction de l'humidité du sol
        # Il n'y a alors aucune infiltration
        evapo = offre

        if modules["infiltration"] == "hsami":
            # Avec la formulation initiale du ruissellement dans
            # hsami, il faut passer le ruissellement en surface
            # méme si la demande est plus forte que l'offre.
            apport[2] = ruissellement_surface

        # Calcul de pompage_min é partir de l'épaisseur de la couche et du point de
        # flétrissement permanent
        limite_pompage = pfp * z[0]
        pompage = min(
            sol[0] - limite_pompage, -sol[0] / sol_max[0] * ecart_offre_demande
        )

        # Ex.: modules.qbase = 'hsami',   pompage = 0.0831
        #      modules.qbase = 'dingman', pompage = 0.0831
        sol[0] = sol[0] - pompage
        inf_potentielle = 0

    # Percolation de l'eau dans la colonne de sol
    # -------------------------------------------
    # Discrétisation en pas de 1h pour éviter les instabilités numériques de la
    # formulation de Black et al. (1970)
    pas_1h = int(24 / nb_pas)
    recharge = 0

    for i_p in range(0, pas_1h):
        # Calcul de K
        k = [
            ks[0] * (sol[0] / sol_max[0]) ** c[0],
            ks[1] * (sol[1] / sol_max[1]) ** c[1],
        ]

        # Drainage gravitaire potentiel de chaque couche
        drainage = [
            sol_max[0] * k[0] * (1 / 24) / z[0],
            sol_max[1] * k[1] * (1 / 24) / z[1],
        ]

        # Séparation de drainage[1] entre le
        # ruissellement intermédiaire et un drainage potentiel vers la nappe
        # é condition qu'il y ait assez d'eau dans la deuxiéme couche

        ecart_sol_min = sol[1] - sol_min[1]
        drainage[1] = min(ecart_sol_min, drainage[1])
        apport[1] = apport[1] + drainage[1] * portion_ruissellement_surface
        sol[1] = sol[1] - drainage[1] * portion_ruissellement_surface
        drainage[1] = drainage[1] * (1 - portion_ruissellement_surface)

        # Algorithme de percolation commençant au bas de la colonne
        for i_s in range(1, -1, -1):
            # 1ere condition : ne pas drainer en-deéa de sol_min
            # --------------------------------------------------
            # Condition particuliére pour la premiére couche qui peut se retrouver
            # entre le point de flétrissement permanent et la capacité au champ é
            # cause du gel ou du pompage.
            if i_s == 0:
                # Si la premiére couche est déjé en-deéa de la capacité au champ.
                if sol[0] < sol_min[0]:
                    # Pas de drainage.
                    drainage[i_s] = 0
                else:
                    ecart_sol_min = sol[i_s] - sol_min[i_s]
                    drainage[i_s] = min(ecart_sol_min, drainage[i_s])
            else:
                ecart_sol_min = sol[i_s] - sol_min[i_s]
                drainage[i_s] = min(ecart_sol_min, drainage[i_s])

            # 2e condition : ne pas dépasser sol_max é la couche inférieure
            # -------------------------------------------------------------
            ecart_sol_max = sol_max[i_s + 1] - sol[i_s + 1]

            if i_s == 1:
                # S'il existe un surplus de drainage dans la 2e couche,
                # celui-ci est envoyé en ruissellement intermédiaire
                surplus = max(drainage[i_s] - ecart_sol_max, 0)
                apport[1] = apport[1] + surplus
                sol[1] = sol[1] - surplus

            drainage[i_s] = min(ecart_sol_max, drainage[i_s])

            # Drainage de la couche
            # ---------------------
            sol[i_s] = sol[i_s] - drainage[i_s]
            sol[i_s + 1] = sol[i_s + 1] + drainage[i_s]

            if i_s == 1:
                recharge = recharge + drainage[i_s]

    # Ex. : modules.qbase = 'hsami'  , sol = [0.4169; 1.0000; 1.8694]
    #       modules.qbase = 'dingman', sol = [0.4169; 1.0000; 4.4037]

    # Infiltration é la surface (le pompage a déjà été retiré)
    # --------------------------------------------------------
    ecart_sol_max = sol_max[0] - sol[0]
    infiltration = min(ecart_sol_max, inf_potentielle)
    apport[2] = apport[2] + inf_potentielle - infiltration
    sol[0] = sol[0] + infiltration

    # Vidange et débordement de la nappe
    # ----------------------------------

    if modules["qbase"] == "hsami":
        apport[0] = sol[2] * taux_vidange_nappe
        sol[2] = sol[2] * (1 - taux_vidange_nappe)

    elif modules["qbase"] == "dingman":
        k = param[26]  # coeff. de récession
        sy = param[27]  # specific yield
        apport[0] = k / nb_pas * sy * sol[2] * np.exp(-k / nb_pas)
        sol[2] = sol[2] - apport[0]

    # Sauvegarde des états
    etat["sol"] = sol[0:2]
    etat["nappe"] = sol[2]  # (nappe)

    etr[2] = evapo
    etr[3] = pompage

    return apport, etat, etr


# Vidange et débordement de la nappe
def vidange_nappe(
    apport, nappe, taux_vidange_nappe, nappe_max, nb_pas, modules, param, sol
):
    """
    Effectue la vidange d'une nappe phréatique en fonction des paramètres donnés.

    Parameters
    ----------
    apport : list
        Liste contenant les apports d'eau.
    nappe : float
        Niveau de la nappe phréatique.
    taux_vidange_nappe : float
        Taux de vidange de la nappe phréatique.
    nappe_max : float
        Niveau maximum de la nappe phréatique.
    nb_pas : float
        Nombre de pas de temps.
    modules : dict
        Les modules pour la simulation.
    param : list
        Paramètres pour la simulation.
    sol : float
        Eau dans le sol.

    Returns
    -------
    apport : list
        Nouveaux apports d'eau.
    nappe : float
        Nouveau niveau de la nappe phréatique.
    sol: float
        Eau dans e sol.
    """
    # Vidange
    if modules["qbase"] == "hsami":
        apport[0] = nappe * taux_vidange_nappe
        nappe = nappe * (1 - taux_vidange_nappe)

    elif modules["qbase"] == "dingman":
        k = param[26]  # coeff. de récession
        sy = param[27]  # specific yield
        apport[0] = k / nb_pas * sy * nappe * np.exp(-k / nb_pas)
        nappe = nappe - apport[0]

    # Débordement
    if nappe > nappe_max:
        apport[2] = apport[2] + nappe - nappe_max
        nappe = nappe_max

    return apport, nappe, sol


# Green-Ampt
def green_ampt(eau_surface, ks, psi, sol_max, sol, nb_pas, gel, neige_au_sol, *args):
    r"""
    Modele de Green-Ampt.

    Parameters
    ----------
    eau_surface : float
        Eau disponible en surface aprés avoir comblé la demande évaporative (cm).
    ks : float
        Conductivité hydraulique saturée (cm/j).
    psi : float
        Pression matricielle au front mouillant, dérivée de Rawls (1993) (cm).
    sol_max : float
        Paramétre 13 correspondant au volume max d'eau ds le sol (cm).
    sol : float
        Variable d'état correspondant au volume d'eau dans le sol (cm).
    nb_pas : float
        Nombre de pas de temps dans une période de 24h (entier positif).
    gel : float
        Gel dans la premiére couche de sol (cm).
    neige_au_sol : float
        Équivalent en eau de la neige au sol (cm).
    \*args : list
        Porosité de la premiére couches de sol si 3couches est utilisé (cm3/cm3).

    Returns
    -------
    infiltration : float
        Infiltration selon Green-Ampt.
    ruissellement : float
        Eeau de ruissellement.

    Notes
    -----
    Fonction calculant l'infiltration et le ruissellement selon le modéle de Green-Ampt tel qu'implémenté dans SWAT.
    """
    k = ks / 2

    if eau_surface * nb_pas < ks:
        infiltration = eau_surface
        ruissellement = 0
    else:
        if len(args) > 0:
            n = args[0]
        else:
            n = 0.45

        m = n * (sol_max - sol) / sol_max

        # Si le sol est complétement saturé, le ruissellement se fait au
        # taux de la conductivité hydraulique (on suppose ainsi qu'il pleut
        # durant tout le pas de temps...)

        if m == 0:
            f = ks
        else:

            def fctobj(f):
                return abs(
                    -f + k / nb_pas + abs(psi) * m * np.log(1 + (f / (abs(psi) * m)))
                )

            f = optimize.fminbound(fctobj, 0, eau_surface * nb_pas)

        # S'il y a du gel et de la neige au sol, l'infiltration est calculée
        # avec Green-Ampt et la formulation de Granger et Pomeroy proporti-
        # onnellement au gel.

        if gel > 0 and neige_au_sol > 0:
            ratio_gel = gel / sol_max

            theta = n * sol / sol_max
            inf = (5 * (1 - theta) * (neige_au_sol * 10) ** 0.584) / 10
            infiltration_potentielle = inf * ratio_gel + f * (1 - ratio_gel)

            if infiltration_potentielle > eau_surface:
                infiltration = eau_surface
                ruissellement = 0
            else:
                infiltration = infiltration_potentielle
                ruissellement = eau_surface - infiltration_potentielle

        else:
            if f > eau_surface:
                infiltration = eau_surface
                ruissellement = 0
            else:
                infiltration = f
                ruissellement = eau_surface - f
    return infiltration, ruissellement


# SCS-CN
def scs_cn(eau_surface, cn):
    """
    Fonction calculant le ruissellement selon la méthode du Curve Number.

    Parameters
    ----------
    eau_surface : float
        Eau disponible en surface aprés avoir comblé la demande évaporative (cm).
    cn : floaat
        Curve Number (paramétre 24).

    Returns
    -------
    infiltration : float
        Infiltration selon Green-Ampt.
    ruissellement : float
        Eau de ruissellement.
    """
    s = (25400 / cn - 254) / 10  # en cm
    potentiel = (eau_surface - 0.2 * s) ** 2 / (eau_surface + 0.8 * s)
    ruissellement = min(potentiel, eau_surface)
    infiltration = eau_surface - ruissellement

    return infiltration, ruissellement
