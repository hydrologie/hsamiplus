"""The main function for HSAMI+ model simulation."""

from __future__ import annotations

from copy import copy

import numpy as np
from hsami2_noyau import hsami2_noyau


def hsami2(projet):
    """
    Simulation du modèle HSAMI.

    Parameters
    ----------
    projet : dict
        Dictionnaire contenant des données d'entrée.

    Returns
    -------
    s : dict
        Sorties de simulation.
    etats: dict
        États du bassin versants et du réservoir.
    deltas: dict
        Composants du bilan massique.

    Raises
    ------
    ValueError
        Si la superficie maximale de la zone humide équivalente est nulle lorsque le module « mhumide » est utilisé.

    Notes
    -----
    Fonction principale pour la simulation du modèle HSAMI.
    Simuler les processus hydrologiques en fonction des paramètres du projet donnés.

    projet : dict, un dictionnaire contenant des données d'entrée :
        - param : liste, 50 parametres du modèle
        - modules : dict, lchoix de modules
        - physio  : dict, qui contient d'information physiologique
        - superficie : liste, superfice du BV est de reservoir
        - meteo : ditc, données météo

    s : dict, un dictionnaire contenant les sorties de simulation avec les clés :
        - 'Qtotal': liste de float
        - 'Qbase': liste de float
        - 'Qinter': liste de float
        - 'Qsurf': liste de float
        - 'Qreservoir': liste de float
        - 'Qglace': liste de float
        - 'ETP': liste de float
        - 'ETRtotal': liste de float
        - 'ETRsublim': liste de float
        - 'ETRPsurN': liste de float
        - 'ETRintercept': liste de float
        - 'ETRtranspir': liste de float
        - 'ETRreservoir': liste de float
        - 'ETRmhumide': liste de float
        - 'Qmh': liste de float

    etats : dict
        Un dictionnaire contenant les états de la simulation à chaque pas de temps.

    deltas : dict
        Un dictionnaire contenant les composants du bilan massique avec les clés :
        - 'total': liste de float
        - 'glace': liste de float
        - 'interception': liste de float
        - 'ruissellement': liste de float
        - 'vertical': liste de float
        - 'mhumide': liste de float
        - 'horizontal': liste de float

    Développé par J.L. Bisson et F. Roberge dans Matlab, 1983.
    Modifié et bonifié par Catherine Guay, Marie Minville, Isabelle Chartier et Jonathan Roy, 2013-2017.
    Traduit en Python par Didier Haguma, 2024.
    """
    # Extraction de variables de la structure projet
    # ----------------------------------------------

    superficie = projet["superficie"]
    if len(superficie) == 1:
        superficie.append(0)

    param = projet["param"]
    physio = projet["physio"]

    # Valeurs par défaut dans modules
    # -----------------------------------
    modules = projet["modules"]

    modules_par_defaut(modules)

    # ------------------------
    # Initialisation des etats
    # ------------------------

    # Dictionnaire états entrants
    # ------------------------
    etat = {}

    etat["eau_hydrogrammes"] = np.zeros((int(projet["memoire"]), 3))

    if modules["een"] in ["mdj", "alt"]:
        if modules["een"] == "mdj":
            n = len(physio["occupation"])
        if modules["een"] == "alt":
            n = len(physio["occupation_bande"])

        etat["modules"] = {}

        etat[modules["een"]] = {
            "couvert_neige": [0] * n,
            "densite_neige": [0] * n,
            "albedo_neige": [0.9] * n,
            "neige_au_sol": [0] * n,
            "fonte": [0] * n,
            "gel": [0] * n,
            "sol": [0] * n,
            "energie_neige": [0] * n,
            "energie_glace": 0,
        }

    etat["neige_au_sol"] = 0
    etat["fonte"] = 0
    etat["nas_tot"] = 0
    etat["fonte_tot"] = 0
    etat["derniere_neige"] = 0
    etat["gel"] = 0
    etat["nappe"] = param[13]
    etat["reserve"] = 0

    if modules["sol"] == "hsami":
        # Initialisation du sol à sol_min.
        etat["sol"] = np.array([param[11], np.nan])
    elif modules["sol"] == "3couches":
        # Initialisation du sol à la capacité au champ.
        etat["sol"] = np.array([param[42] * param[39], param[43] * param[40]])

    if modules["mhumide"] == 1:
        if physio["samax"] == 0:
            raise ValueError(
                "La superficie maximale du milieu humide \
                             équivalent est égale à 0."
            )

        etat["mh_surf"] = (
            param[48] * physio["samax"] * 100
        )  # On considère la surface initiale égale à la surface normale (en hectars)
        etat["mh_vol"] = param[48] * (
            param[47] * physio["samax"] * 100 * 10000
        )  # On considère le volume initial au volume normal (en m^3)
        etat["ratio_MH"] = etat["mh_surf"] / (superficie[0] * 100)

    if modules["mhumide"] == 0:
        etat["mh_vol"] = 0
        etat["ratio_MH"] = 0
        etat["mh_surf"] = 1

    etat["mhumide"] = etat["mh_vol"] * etat["ratio_MH"] / (etat["mh_surf"] * 100)
    etat["ratio_qbase"] = 0

    # Glace/réservoir
    etat["cumdegGel"] = 0
    etat["obj_gel"] = -200
    etat["dernier_gel"] = 0
    etat["reservoir_epaisseur_glace"] = 0
    etat["reservoir_energie_glace"] = 0
    etat["reservoir_superficie"] = superficie[1]
    etat["reservoir_superficie_glace"] = 0
    etat["reservoir_superficie_ref"] = etat["reservoir_superficie"]
    etat["eeg"] = np.zeros(5000)
    etat["ratio_bassin"] = 1
    etat["ratio_reservoir"] = 0
    etat["ratio_fixe"] = 1

    # Structure états sortants
    # ------------------------

    nb_pas_total = len(projet["meteo"]["bassin"])

    etats = {}

    f = list(etat.keys())

    for i_f in range(len(f)):
        etats[f[i_f]] = []

    # ----------------------
    # Structure des sorties
    # ----------------------
    s = {
        "Qtotal": [],
        "Qbase": [],
        "Qinter": [],
        "Qsurf": [],
        "Qreservoir": [],
        "Qglace": [],
        "ETP": [],
        "ETRtotal": [],
        "ETRsublim": [],
        "ETRPsurN": [],
        "ETRintercept": [],
        "ETRtranspir": [],
        "ETRreservoir": [],
        "ETRmhumide": [],
        "Qmh": [],
    }

    deltas = {
        "total": [],
        "glace": [],
        "interception": [],
        "ruissellement": [],
        "vertical": [],
        "mhumide": [],
        "horizontal": [],
    }

    # Conditions initiales
    etat = hsami_etat_initial(projet, param, modules, physio, superficie, etat)

    # Simulation
    s, etats, deltas = hsami_simulation(
        projet, param, modules, physio, superficie, etat, nb_pas_total, s, etats, deltas
    )

    return s, etats, deltas


def set_default_module(modules, key, default_value):
    """
    Set module defaults values.

    Parameters
    ----------
    modules : dict
        Dictionary of modules.
    key : str
        Hydrological process.
    default_value : str
        HSAMI+ mudule name.
    """
    if key not in modules:
        modules[key] = default_value


def modules_par_defaut(modules):
    """
    Check projet modules definition.

    Parameters
    ----------
    modules : dict
        Dictionary of modules.
    """
    valeurs_default = {
        "etp_bassin": "hsami",
        "etp_reservoir": "hsami",
        "een": "hsami",
        "infiltration": "hsami",
        "qbase": "hsami",
        "sol": "hsami",
        "radiation": "hsami",
        "reservoir": 0,
        "mhumide": 0,
        "glace_reservoir": 0,
    }

    for key, value in valeurs_default.items():
        set_default_module(modules, key, value)


def hsami_etat_initial(projet, param, modules, physio, superficie, etat):
    """
    Tour de chauffe (1 an).

    Parameters
    ----------
    projet : dict
        Données du projet HSAMI+.
    param : list
        Paramètres pour la simulation.
    modules : dict
        Les modules pour la simulation.
    physio : dict
        Les données physiographiques.
    superficie : list
        La superficie du bassin versan et  la uuperficie moyenne du réservoir.
    etat : dict
        État du bassin versants et du réservoir.

    Returns
    -------
    dict
        État du bassin versants et du réservoir.
    """
    pas = 1
    for i_pas in range(365):
        # Construction du projet pour hsami_noyau
        p = {}

        if "hu_surface" in projet:
            p["hu_surface"] = projet["hu_surface"]
        if "hu_inter" in projet:
            p["hu_inter"] = projet["hu_inter"]

        p["date"] = projet["dates"][i_pas]
        p["nb_pas_par_jour"] = projet["nb_pas_par_jour"]
        p["superficie"] = superficie
        p["memoire"] = projet["memoire"]
        p["param"] = param
        p["meteo"] = {
            "bassin": projet["meteo"]["bassin"][i_pas],
            "reservoir": projet["meteo"]["reservoir"][i_pas],
        }
        p["modules"] = modules
        p["physio"] = copy(physio)
        p["pas"] = pas
        if "niveau" in physio.keys():
            p["physio"]["niveau"] = physio["niveau"][i_pas]

        # Simulation
        _, etat, _ = hsami2_noyau(p, etat)

        # On avance d'un pas de temps
        if pas == projet["nb_pas_par_jour"]:
            pas = 1
        else:
            pas = pas + 1

    return etat

    # ----------
    # Simulation
    # ----------


def hsami_simulation(
    projet, param, modules, physio, superficie, etat, nb_pas_total, s, etats, deltas
):
    """
    Simulation avec HASMAI+.

    Parameters
    ----------
    projet : dict
        Dictionnaire contenant des données d'entrée.
    param : list
        Paramètres pour la simulation.
    modules : dict
        Les modules pour la simulation.
    physio : dict
        Les données physiographiques.
    superficie : list
        La superficie du bassin versan et  la uuperficie moyenne du réservoir.
    etat : dict
        État du bassin versants et du réservoir à un pas de temps.
    nb_pas_total : float
        Nombre de pas des temps total.
    s : dict
        Sorties de simulation.
    etats : dict
        États du bassin versants et du réservoir pout tous les pas de temps.
    deltas : dict
        Composants du bilan massique.

    Returns
    -------
    s : dict
        Sorties de simulation.
    etats : dict
        États du bassin versants et du réservoir.
    deltas : dict
        Composants du bilan massique.

    Notes
    -----
    projet : dict, Un dictionnaire contenant les clés suivantes :
        - 'superficie' : liste des floats, la zone du projet. S'il ne contient qu'un seul élément,
           un deuxième élément de valeur 0 est ajouté.
        - 'param' : liste des float, Paramètres pour la simulation.
        - 'mémoire' : int, taille de la mémoire pour la simulation.
        - 'physio' : dict, les données physiographiques peuvent être vides.
        - 'modules' : dict, les modules pour la simulation peuvent être vides. Les valeurs par défaut
           sont définies si elles ne sont pas fournies.
        - 'meteo' : dict, données météorologiques pour la simulation.
        - 'dates' : liste des str, dates de simulation.
        - 'nb_pas_par_jour' : entier, nombre de pas de temps par jour.

    s : dict, un dictionnaire contenant les sorties de simulation avec les clés :
        - 'Qtotal': liste de float
        - 'Qbase': liste de float
        - 'Qinter': liste de float
        - 'Qsurf': liste de float
        - 'Qreservoir': liste de float
        - 'Qglace': liste de float
        - 'ETP': liste de float
        - 'ETRtotal': liste de float
        - 'ETRsublim': liste de float
        - 'ETRPsurN': liste de float
        - 'ETRintercept': liste de float
        - 'ETRtranspir': liste de float
        - 'ETRreservoir': liste de float
        - 'ETRmhumide': liste de float
        - 'Qmh': liste de float

    etats : dict
        Un dictionnaire contenant les états de la simulation à chaque pas de temps.

    deltas : dict
        Un dictionnaire contenant les composants du bilan massique avec les clés :
        - 'total': liste de float
        - 'glace': liste de float
        - 'interception': liste de float
        - 'ruissellement': liste de float
        - 'vertical': liste de float
        - 'mhumide': liste de float
        - 'horizontal': liste de float
    """
    pas = 1
    for i_pas in range(nb_pas_total):
        # Construction du projet pour hsami_noyau
        p = {}

        if "hu_surface" in projet:
            p["hu_surface"] = projet["hu_surface"]
        if "hu_inter" in projet:
            p["hu_inter"] = projet["hu_inter"]

        p["date"] = projet["dates"][i_pas]
        p["nb_pas_par_jour"] = projet["nb_pas_par_jour"]
        p["superficie"] = superficie
        p["memoire"] = projet["memoire"]
        p["param"] = param
        p["meteo"] = {
            "bassin": projet["meteo"]["bassin"][i_pas],
            "reservoir": projet["meteo"]["reservoir"][i_pas],
        }
        p["modules"] = modules
        p["physio"] = copy(physio)
        if "niveau" in physio.keys():
            p["physio"]["niveau"] = physio["niveau"][i_pas]
        p["pas"] = pas

        # Simulation
        s_sim, etat, delta = hsami2_noyau(p, etat)

        # Sauvegarde des sorties
        f = list(s_sim.keys())

        for i_f in range(len(f)):
            s[f[i_f]].append(s_sim[f[i_f]])

        # Sauvegarde des états
        f = list(etat.keys())
        for i_f in range(len(f)):
            if isinstance(etat[f[i_f]], np.ndarray):
                if f[i_f] == "eeg":
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
        if pas == projet["nb_pas_par_jour"]:
            pas = 1
        else:
            pas = pas + 1

    return s, etats, deltas
