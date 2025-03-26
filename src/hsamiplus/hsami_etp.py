"""The function simulates the evapotranspiration (ETP) in HSAMI+ model."""

from __future__ import annotations

import numpy as np


def hsami_etp(pas, nb_pas, jj, t_min, t_max, modules, physio):
    """
    Calcul de l'évapotranspiration potentielle.

    Parameters
    ----------
    pas : int
        Pas de temps courant à l'intérieur de la journée.
    nb_pas : float
        Nombre de pas de temps.
    jj : int
        Jour julien (entier positif).
    t_min : float
        Tmin journalière.
    t_max : float
        Tmax journalière.
    modules : dict
        Les modules pour la simulation.
    physio : dict
        Variables physiographiques du bassin.

    Returns
    -------
    float
        Estimation de l'évapotranspiration potentielle.

    Notes
    -----
    MODULES D'ÉVAPOTRANSPIRATION DISPONIBLES
    - hsami
    - blaney_criddle
    - hamon
    - linacre
    - kharrufa
    - mohyse
    - romanenko
    - makkink
    - turc
    - mcguinness_bordne
    - abtew
    - hargreaves
    - priestley_taylor

    Marie Minville, Catherine Guay, 2013
    Didier Haguma, 2024
    """
    # Poids horaires pour distribuer l'évapotranspiration potentielle
    poids = (
        np.array(
            [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.6,
                1.1,
                2.4,
                4,
                5.4,
                7,
                8.4,
                9.6,
                10.4,
                10.9,
                10.8,
                9.9,
                7.8,
                5,
                2,
                0.7,
                0.5,
                0.5,
                0.5,
            ]
        )
        / 100
    )

    # Calcul de l'ETP total pour la journée
    if modules == "hsami":  # Ex. : etp_total = 0.1788
        etp_total = (
            0.00065
            * 2.54
            * 9
            / 5
            * (t_max - t_min)
            * np.exp(0.019 * (t_min * 9 / 5 + t_max * 9 / 5 + 64))
        )

    elif modules == "blaney_criddle":  # Ex. : etp_total = 0.2799
        p = etp_p(physio["latitude"], jj)
        etp_total = etp_blaney_criddle(t_min, t_max, p)

    elif modules == "hamon":  # Ex. : etp_total = 0.1281
        etp_total = etp_hamon(jj, t_min, t_max, physio)

    elif modules == "linacre":  # Ex. : etp_total = 0.1043
        etp_total = etp_linacre(t_min, t_max, physio)

    elif modules == "kharrufa":  # Ex. : etp_total = 0.0757
        p = etp_p(physio["latitude"], jj)
        etp_total = etp_kharrufa(t_min, t_max, p)

    elif modules == "mohyse":  # Ex. : etp_total = 0.0812
        delta = etp_declinaison(jj)
        etp_total = etp_mohyse(t_min, t_max, delta, physio)

    elif modules == "romanenko":  # Ex. : etp_total = 0.2357
        etp_total = etp_romanenko(t_min, t_max)

    elif modules == "makkink":  # Ex. : etp_total = 0.2526
        re = etp_rayonnement_et(physio["latitude"], jj)
        rg = etp_rayonnement_g(re, physio["latitude"], jj, t_min, t_max)
        m = etp_m_courbe_pression(t_min, t_max)
        lamda = etp_chaleur_lat_vaporisation(t_min, t_max)
        etp_total = etp_makkink(rg, m, lamda)

    elif modules == "turc":  # Ex. : etp_total = 0.1988
        re = etp_rayonnement_et(physio["latitude"], jj)
        rg = etp_rayonnement_g(re, physio["latitude"], jj, t_min, t_max)
        etp_total = etp_turc(t_min, t_max, rg)

    elif modules == "mcguinness_bordne":  # Ex. : etp_total = 0.1274
        re = etp_rayonnement_et(physio["latitude"], jj)
        rg = etp_rayonnement_g(re, physio["latitude"], jj, t_min, t_max)
        lamda = etp_chaleur_lat_vaporisation(t_min, t_max)
        etp_total = etp_mcguinness_bordne(t_min, t_max, rg, lamda)

    elif modules == "abtew":  # Ex. : etp_total = 0.4884
        re = etp_rayonnement_et(physio["latitude"], jj)
        rg = etp_rayonnement_g(re, physio["latitude"], jj, t_min, t_max)
        lamda = etp_chaleur_lat_vaporisation(t_min, t_max)
        etp_total = etp_abtew(t_min, t_max, rg, lamda)

    elif modules == "hargreaves":  # Ex. : etp_total = 0.2566
        re = etp_rayonnement_et(physio["latitude"], jj)
        etp_total = etp_hargreaves(t_min, t_max, re)

    elif modules == "priestley_taylor":  # Ex. : etp_total = 0.0339
        re = etp_rayonnement_et(physio["latitude"], jj)
        rgo = etp_rayonnement_temps_clair(re, physio["altitude"])
        rg = etp_rayonnement_g(re, physio["latitude"], jj, t_min, t_max)
        rn = etp_rayonnement_net(t_min, t_max, rg, rgo, physio["albedo_sol"])
        m = etp_m_courbe_pression(t_min, t_max)
        lamda = etp_chaleur_lat_vaporisation(t_min, t_max)
        etp_total = etp_priestley_taylor(rn, m, lamda)

    etp_total = max(0, etp_total)

    # Aggrégation selon le pas de temps
    debut = int((pas - 1) * 24 / nb_pas)
    fin = int(pas * 24 / nb_pas)
    etp = etp_total * np.sum(poids[debut:fin])

    return etp


# -----------------------------
# FIN DE LA FONCTION PRINCIPALE
# -----------------------------

# ----------------------------
# FONCTIONS DE CALCUL DE L'ETP
# ----------------------------


def etp_blaney_criddle(t_min, t_max, p):
    """
    Calcul de l'évapotranspiration potentielle à partir de la formulation de Blaney-Criddle.

    Parameters
    ----------
    t_min : float
        Tmin journalière.
    t_max : float
        Tmax journalière.
    p : float
        Heures de clarté journalière sur le nombre d'heures de clarté annuelle.

    Returns
    -------
    float
        Estimation de l'evapotranspiration potentielle (cm).

    Notes
    -----
    Calcul de l'évapotranspiration potentielle en cm à partir de la formulation de Blaney-Criddle
    et répartition dans la journée selon la pondération proposée par Fortin, J.P.
    et Girard, G. (1970).
    """
    # température moyenne
    t_a = (t_min + t_max) / 2

    # Calcul de l'ETP total pour la journée
    k = 0.85  # Constante proposée par Xu et Singh (2001). Peut varier entre 0.5 et 1.2;
    etp_total = (
        k * p * (0.46 * t_a + 8.13) / 10
    )  # cm, formulation en mm selon Xu et Singh (2001)
    etp_total = max(0, etp_total)

    return etp_total


def etp_hamon(jj, t_min, t_max, physio):
    """
    Calcul de l'évapotranspiration potentielle selon la formulation de Hamon.

    Parameters
    ----------
    jj : int
        Jour julien.
    t_min : float
        Tmin journalière.
    t_max : float
        Tmax journalière.
    physio : dict
        Variables physiographiques du bassin.

    Returns
    -------
    float
        Estimation de l'évapotranspiration potentielle (cm).

    Notes
    -----
    Calcul de l'évapotranspiration potentielle en cm selon la formulation de Hamon
    et répartition dans la journée selon la pondération proposée par Fortin, J.P.
    et Girard, G. (1970).
    """
    dl = etp_duree_jour(jj, physio["latitude"])

    # température moyenne
    t_a = (t_min + t_max) / 2

    # Pression de vapeur
    es = etp_e(t_a)  # En émettant l'hypothése que Ts = Ta, Oudin (2004).

    # Calcul de l'ETP total pour la journée
    etp_total = 2.1 * dl**2 * es / (t_a + 273.3) / 10  # Haith et Shoemaker (1987).
    etp_total = max(0, etp_total)

    return etp_total


def etp_linacre(t_min, t_max, physio):
    """
    Calcul de l'évapotranspiration potentielle selon la formulaiton Linacre.

    Parameters
    ----------
    t_min : float
        Tmin journalière.
    t_max : float
        Tmax journalière.
    physio : dict
        Variables physiographiques du bassin.

    Returns
    -------
    float
        Estimation de l'évapotranspiration potentielle (cm).

    Notes
    -----
    Calcul de l'évapotranspiration potentielle en cm selon la formulaiton Linacre
    et répartition dans la journée selon la pondération proposée par Fortin, J.P.
    et Girard, G. (1970).
    """
    # température moyenne
    t_a = (t_min + t_max) / 2

    # température estimée é une altitude h
    t_h = t_a + 0.006 * physio["altitude"]

    # point de rosée
    t_d = etp_td_linacre(
        t_max, t_min
    )  # t_d = 0.38+t_max-0.018*t_max^2+1.4+t_min-5; Proposition de Linacre pour estimer t_d, pas applicable dans les zones trés maritimes.

    # Calcul de l'ETP total pour la journée
    if t_a < 0:  # le point de rosée ne peut pasêtre calculé avec une Ta négative.
        etp_total = 0
    else:
        # la latitude doitêtre en degré pour cette formulation
        lat = physio["latitude"] * 180 / np.pi
        etp_total = (
            (500 * t_h / (100 - lat) + 15 * (t_a - t_d)) / (80 - t_a) / 10
        )  # cm; Xu et Singh (2001)

    return etp_total


def etp_kharrufa(t_min, t_max, p):
    """
    Calcul de l'évapotranspiration potentielle selon la formulation Kharrufa.

    Parameters
    ----------
    t_min : float
        Tmin journalière.
    t_max : float
        Tmax journalière.
    p : float,
        Heures de clarté journalière sur le nombre déheures de clarté annuelle.

    Returns
    -------
    float
        Estimation de l'évapotranspiration potentielle (cm).

    Notes
    -----
    Calcul de l'évapotranspiration potentielle en cm selon la formulation Kharrufa
    et répartition dans la journée selon la pondération proposée par Fortin, J.P.
    et Girard, G. (1970).
    """
    # température moyenne
    t_a = (t_min + t_max) / 2

    t_a = max(
        0, t_a
    )  # MM20130712: Ta = 0 si elle est negative car sinon ETP = nbr imaginaire
    etp_total = (
        0.34 * p * t_a ** (1.3) / 10
    )  # cm #formulation originale en mm. Xu et Singh (2001)

    return etp_total


def etp_mohyse(t_min, t_max, delta, physio):
    """
    Calcul de l'évapotranspiration potentielle selon la formulation du modéle Mohyse.

    Parameters
    ----------
    t_min : float
        Tmin journalière.
    t_max : float
        Tmax journalière.
    delta : float
        Déclinaison solaire (radians).
    physio : dict
        Variables physiographiques du bassin.

    Returns
    -------
    float
        Estimation de l'évapotranspiration potentielle (cm).

    Notes
    -----
    Calcul de l'évapotranspiration potentielle en cm selon la formulation du modéle Mohyse
    et répartition dans la journée selon la pondération proposée par Fortin, J.P.
    et Girard, G. (1970).
    """
    # température moyenne
    t_a = (t_min + t_max) / 2

    # Calcul de l'ETP total pour la journée
    etp_total = (
        1
        / np.pi
        * np.arccos(-np.tan(physio["latitude"]) * np.tan(delta))
        * np.exp((17.3 * t_a) / (238 + t_a))
        / 10
    )  # cm, Fortin et Turcotte (2007)

    return etp_total


def etp_romanenko(t_min, t_max):
    """
    Calcul de l'évapotranspiration potentielle selon la formulation Romanenko.

    Parameters
    ----------
    t_min : float
        Tmin journalière.
    t_max : float
        Tmax journalière.

    Returns
    -------
    float
        Estimation de l'évapotranspiration potentielle (cm).

    Notes
    -----
    Calcul de l'évapotranspiration potentielle en cm selon la formulation
    Romanenko et répartition dans la journée selon la pondération proposée
    par Fortin, J.P. et Girard, G. (1970).
    """
    # température moyenne
    t_a = (t_min + t_max) / 2

    # Pression de vapeur
    ea = etp_e(t_a)
    ed = etp_e(
        t_min
    )  # on peut supposer que td=tmin pour les emplacements ou le couvert vegetal est bien humidifie (Kimball et al. (1997)

    # Calcul de l'ETP total pour la journée
    etp_total = (
        0.0045 * (1 + t_a / 25) ** 2 * (1 - ed / ea) * 100
    )  # cm, La version initiale est en m. Oudin (2004)

    return etp_total


def etp_makkink(rg, m, lamda):
    """
    Calcul de l'évapotranspiration potentielle selon la formulation de Makkink.

    Parameters
    ----------
    rg : float
        Rayonnement global MJ/m^2/j.
    m : float
        Pente de la courbe de pression.
    lamda : float
        Chaleur de vaporisation MJ/kg.

    Returns
    -------
    float
        Estimation de l'évapotranspiration potentielle (cm).

    Notes
    -----
    Calcul de l'évapotranspiration potentielle en cm selon la formulation de Makkink
    et répartition dans la journée selon la pondération proposée par Fortin, J.P.
    et Girard, G. (1970).
    """
    psi = 0.066  # Constante psychométrique (0,066 kPa/éC);

    # Calcul de l'ETP total pour la journée
    etp_total = ((m / (m + psi)) * (0.61 * rg / lamda) - 0.12) / 10

    return etp_total


def etp_turc(t_min, t_max, rg):
    """
    Calcul de l'évapotranspiration potentielle à partir des températures min et max.

    Parameters
    ----------
    t_min : float
        Tmin journalière.
    t_max : float
        Tmax journalière.
    rg : float
        Rayonnement global MJ/m^2/j.

    Returns
    -------
    float
        Estimation de l'évapotranspiration potentielle (cm).

    Notes
    -----
    Calcul de l'évapotranspiration potentielle en cm à partir des températures min et max
    en C journalière, selon la méthode empirique de Jean-Louis Bisson (Hydro-Québec)
    et répartition dans la journée selon la pondération proposée par Fortin,
    J.P. et Girard, G. (1970).
    """
    # température moyenne
    t_a = (t_min + t_max) / 2

    k = 0.35  # Constante de Turc

    # Calcul de l'ETP total pour la journée
    if t_a < 0:
        etp_total = 0
    else:
        etp_total = (
            k * (rg + 2.094) * (t_a / (t_a + 15)) / 10
        )  # cm; McGuiness et Bordne (1972), unité mise en SI

    return etp_total


def etp_mcguinness_bordne(t_min, t_max, rg, lamda):
    """
    Calcul de l'évapotranspiration potentielle selon la formulation de McGuiness et Bordne.

    Parameters
    ----------
    t_min : float
        Tmin journalière.
    t_max : float
        Tmax journalière.
    rg : float
        Rayonnement global MJ/m^2/j.
    lamda : float
        Chaleur de vaporisation MJ/kg.

    Returns
    -------
    float
        Estimation de l'évapotranspiration potentielle (cm).

    Notes
    -----
    Calcul de l'évapotranspiration potentielle en cm selon la formulation de McGuiness et Bordne
    et répartition dans la journée selon la pondération proposée par Fortin, J.P.
    et Girard, G. (1970).
    """
    # température moyenne
    t_a = (t_min + t_max) / 2

    rho_w = 1000  # Masse volumique de léeau (1000 kg/m3)

    # Calcul de l'ETP total pour la journée

    etp_total = (
        rg / (lamda * rho_w) * (t_a + 5) / 68
    ) * 100  # cm, version originale en m. Oudin (2004)

    return etp_total


def etp_abtew(t_min, t_max, rg, lamda):
    """
    Calcul de l'évapotranspiration potentielle selon la méthode empirique de Abtew.

    Parameters
    ----------
    t_min : float
        Tmin journalière.
    t_max : float
        Tmax journalière.
    rg : float
        Rayonnement global MJ/m^2/j.
    lamda : float
        Chaleur de vaporisation MJ/kg.

    Returns
    -------
    float
        Estimation de l'évapotranspiration potentielle (cm).

    Notes
    -----
    Calcul de l'évapotranspiration potentielle en cm selon la méthode empirique de Abtew
    et répartition dans la journée selon la pondération proposée par Fortin, J.P.
    et Girard, G. (1970).
    """
    # température moyenne
    t_a = (t_min + t_max) / 2

    # Calcul de l'ETP total pour la journée
    if t_a < 0:
        etp_total = 0
    else:
        etp_total = 0.53 * rg / lamda / 10  # Xu et Singh 2010

    return etp_total


def etp_hargreaves(t_min, t_max, re):
    """
    Calcul de l'évapotranspiration potentielle selon la formulation de Hargreaves.

    Parameters
    ----------
    t_min : float
        Tmin journalière.
    t_max : float
        Tmax journalière.
    re : float
        Rayonnement extraterrestre (MJ/m^2/j).

    Returns
    -------
    float
        Estimation de l'évapotranspiration potentielle (cm).

    Notes
    -----
    Calcul de l'évapotranspiration potentielle en cm selon la formulation de Hargreaves et Samani
    et répartition dans la journée selon la pondération proposée par Fortin, J.P.
    et Girard, G. (1970).
    """
    # température moyenne
    t_a = (t_min + t_max) / 2

    # Calcul de l'ETP total pour la journée
    if (
        t_max - t_min < 0
    ):  # il y a parfois des incohérence dans les séries observées. Cette condition pourraitêtre enlevée éventuellement.
        etp_total = 0
    else:
        etp_total = (
            0.0135 * (0.16 * re * np.sqrt(t_max - t_min)) * 0.4082 * (t_a + 17.8) / 10
        )  # Goyal et Harmsen (2014). Extrait du livre via Google book.

    return etp_total


def etp_priestley_taylor(rn, m, lamda):
    """
    Calcul de l'évapotranspiration potentielle selon la formulation de Priesley-Taylor.

    Parameters
    ----------
    rn : float
        Rayonnement net (MJ/m^2/j).
    m : float
        Pente de la courbe de pression.
    lamda : float
        Chaleur de vaporisation (MJ/kg).

    Returns
    -------
    float
        Estimation de l'évapotranspiration potentielle (cm).

    Notes
    -----
    Calcul de l'évapotranspiration potentielle en cm selon la formulation de Priesley-Taylor
    et répartition dans la journée selon la pondération proposée par Fortin, J.P. et Girard, G. (1970).
    """
    psi = 0.066  # Constante psychométrique (0,066 kPa/éC);
    rho_w = 1000  # Masse volumique de l'eau (kg/m3)

    ct = 1.26  # constante proposée par Priesley-Taylor

    # Calcul de l'ETP total pour la journée
    etp_total = (
        ct * m * rn / (lamda * rho_w * (m + psi)) * 100
    )  # cm; La formule proposée est en m. Oudin (2004)

    return etp_total


# --------------------
# FONCTIONS DE SOUTIEN
# --------------------
def etp_p(lat, jj):
    """
    Calcul du pourcentage de la durée du jour sur la somme des durées du jour annuelles.

    Parameters
    ----------
    lat : float
        Latitude moyenne du bassin versant.
    jj : int
        Jour julien.

    Returns
    -------
    float
        Heures de clarté journalière sur le nombre d'heures de clarté annuelle.
    """
    dl = np.zeros(366)

    for jj2 in range(366):
        dl[jj2] = etp_duree_jour(jj2, lat)

    p = 100 * (dl[jj] / np.sum(dl))  # Xu et Singh (2000)

    return p


def etp_duree_jour(jj, lat):
    """
    Calcul de la duree du jour jj à la latitude lat.

    Parameters
    ----------
    jj : int
        Jour julien.
    lat : float
        Latitude moyenne du bassin versant.

    Returns
    -------
    float
        Duree du jour au jour jj et à la latitude lat.
    """
    delta = etp_declinaison(jj)

    ws = np.arccos(-np.tan(lat) * np.tan(delta))  # angle de coucher de soleil (rad)
    dl = 24 / np.pi * ws

    return dl


def etp_declinaison(jj):
    """
    Calcul de la declinaison solaire (en radians) au jour julien jj.

    Parameters
    ----------
    jj : int
        Jour julien.

    Returns
    -------
    float
        Declinaison du soleil (radians) pour mohyse entre autre.
    """
    delta = 0.41 * np.sin((jj - 80) / 365 * 2 * np.pi)

    return delta


def etp_td_linacre(t_max, t_min):
    """
    Estimation du point de rosée de Linacre.

    Parameters
    ----------
    t_max : float
        Tmax journalière.
    t_min : float
        Tmin journalière.

    Returns
    -------
    float
        Point de rosée.
    """
    # td Point de rosée
    td = (
        0.38 + t_max - 0.018 * t_max**2 + 1.4 + t_min - 5
    )  # Proposition de Linacre pour estimer td, pas applicable dans les zones trés maritimes.

    return td


def etp_rayonnement_et(lat, jj):
    """
    Calcul du rayonnement extra-terrestre.

    Parameters
    ----------
    lat : float
        Latitude moyenne du bassin versant.
    jj : int
        Jour julien.

    Returns
    -------
    float
        Rayonnement extra-terrestre (MJ/m^2/j).

    Notes
    -----
    Selon http://www.argenco.ulg.ac.be/etudiants/Multiphysics/Xanthoulis#20-#20Calcul#20ETo#20-#20Penman.pdf
    """
    gsc = 0.0820  # MJ/m2/j constante solaire

    dr = 1 + 0.033 * np.cos(
        2 * np.pi / 365 * jj
    )  # distance relative inverse terre-soleil (rad)
    delta = 0.409 * np.sin(2 * np.pi * jj / 365 - 1.39)  # declinaison solaire (rad)
    ws = np.arccos(-np.tan(lat) * np.tan(delta))  # angle de coucher de soleil (rad)

    re = (
        24
        * 60
        / np.pi
        * gsc
        * dr
        * (
            ws * np.sin(lat) * np.sin(delta)
            + (np.cos(lat) * np.cos(delta) * np.sin(ws))
        )
    )

    return re


def etp_rayonnement_g(re, lat, jj, t_min=None, t_max=None):
    """
    Calcul du rayonnement global.

    Parameters
    ----------
    re : float
        Rayonnement extra-terrestre (MJ/m^2/j).
    lat : float
        Latitude moyenne du bassin versant (m).
    jj : int
        Jour julien.
    t_min : float
        Tmin journalière.
    t_max : float
        Tmax journalière.

    Returns
    -------
    float
        Rayonnement global (MJ/m^2/j).
    """
    dl = etp_duree_jour(jj, lat)
    d = (
        0.8 * dl
    )  # Hypothése, nous n'avons pas d'observation pour estimer la durée effective du jour.

    # Rayonnement global
    rg = re * (0.18 + 0.52 * d / dl)

    # Autre facon, bien si on ne connait pas d.
    if t_min:
        krs = 0.175
        rg = krs * (t_max - t_min) ** (1 / 2) * re

    # La différence entre la température maximum et minimum (Tmax-Tmin) de
    # l'air peut être utilisé comme un indicateur de la fraction de radiation
    # extraterrestre qui atteint la surface du sol.
    # Ra : Rayonnement extraterrestre [MJ m-2d-1],
    # Tmax: température maximum de l'air [oC],Tmin: température minimum de l'aair [oC],
    # Krs: Coefficient (0.16.. 0.19) [oC-0.5].pour des zones interieures où
    # les masses de terres ne sont pas influencées fortement par de grandes masses d'eau:
    # Krs= 0.16; pour des zones cétiéres situées sur ou adjacentes à une grande
    # masse de terre et où les masses d'air sont influencées par une masse d'eau proche: Krs= 0.19.

    return rg


def etp_m_courbe_pression(t_min, t_max):
    """
    Estimation de la pente de la courbe de pression de vapeur.

    Parameters
    ----------
    t_min : float
        Tmin journalière.
    t_max : float
        Tmax journalière.

    Returns
    -------
    float
        Pente de la courbe de pression de vapeur.
    """
    # température moyenne
    t_a = (t_min + t_max) / 2

    # ea pression de vapeur
    ea = etp_e(t_a)

    # m: pente
    m = 4098 * ea / (237.3 + t_a) ** 2  # Oudin (2004)

    return m


def etp_e(t):
    """
    Estimation du point de la pression de vapeur.

    Parameters
    ----------
    t : float
        Température.

    Returns
    -------
    float
        Pression de vapeur.
    """
    # e Pression de vapeur
    e = 0.6108 * np.exp(
        (17.27 * t) / (t + 237.3)
    )  # Lu et al. (2005)  #pression de vapeur é T

    return e


def etp_chaleur_lat_vaporisation(t_max, t_min):
    """
    Estimation de la chaleur latente de vaporisation.

    Parameters
    ----------
    t_max : float
        Tmax journalière.
    t_min : float
        Tmin journalière.

    Returns
    -------
    float
        Chaleur latente de vaporisation (MJ/kg).
    """
    # température moyenne
    t_a = (t_min + t_max) / 2

    # lamda
    lamda = (
        2.5 - 2.36 * 10**-3 * t_a
    )  # selon Dingman, p.274. En MJ/kg. A 20degC, ca revient au flux de chaleur latente de 2.45 MJ/kg fixé dans Oudin et al. (2005)

    return lamda


def etp_rayonnement_net(t_min, t_max, rg, rgo, albedo):
    """
    Calcul du rayonnement net.

    Parameters
    ----------
    t_min : float
        Tmin journalière.
    t_max : float
        Tmax journalière.
    rg : float
        Rayonnement global (MJ/m^2/j).
    rgo : float
        Rayonnement par temps clair (MJ/m^2/j).
    albedo : float
        Albedo de la surface.

    Returns
    -------
    float
        Rayonnement net (MJ/m^2/j).
    """
    # Rayonnement net de courte longueur d'ondes
    rns = rg * (1 - albedo)

    # Rayonnement net de longue longueur d'ondes
    sigma = 4.903 * 10 ** (-9)  # constante de S-B
    k = 273.16  # pour avoir des Kelvins

    ea = etp_e(t_min)  # Td = Tmin est une approximation valable (Kimball et al. 1997)

    rapport = rg / rgo
    if rapport >= 1:
        rapport = 1  # selon Xu et Singh (2002) - WRM

    rnl = (
        sigma
        * ((t_max + k) ** 4 + (t_min + k) ** 4)
        / 2
        * (0.34 - 0.14 * np.sqrt(ea))
        * (1.35 * rapport - 0.35)
    )

    # Rayonnement net
    rn = rns - rnl

    return rn


def etp_rayonnement_temps_clair(re, h):
    """
    Calcul du rayonnement par temps clair.

    Parameters
    ----------
    re : float
        Rayonnement extraterrestre (MJ/m^2/j).
    h : float
        Hauteur moyenne du bassin versant au dessus du niveau de la mer (m).

    Returns
    -------
    float
        Rayonnement pas temps clair considérant D=DL (MJ/m^2/j).
    """
    # Rayonnement solaire par temps clair.
    rgo = (0.75 + 2.10 * 10**-5 * h) * re  # Xu et Singh (2002). WRM

    return rgo
