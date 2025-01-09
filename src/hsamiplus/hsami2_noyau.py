"""The core of HSAMI+ that simulates hydrological processes for ONE TIME STEP."""

from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np

from .hsami_ecoulement_horizontal import hsami_ecoulement_horizontal
from .hsami_ecoulement_vertical import hsami_ecoulement_vertical
from .hsami_etp import hsami_etp
from .hsami_glace import hsami_glace
from .hsami_hydrogramme import hsami_hydrogramme
from .hsami_interception import hsami_interception
from .hsami_mhumide import hsami_mhumide
from .hsami_ruissellement_surface import hsami_ruissellement_surface


def hsami2_noyau(projet, etat):
    """
    Noyau d'HSAMI pour simule UN SEUL PAS DE TEMPS.

    Parameters
    ----------
    projet : dict
        Dictionnaire contenant données d'entrée de HSAMI+.
    etat : dict
        États du bassin versants et du réservoir.

    Returns
    -------
    s : dict
        Sorties de simulation.
    etats: dict
        États du bassin versants et du réservoir.
    deltas: dict
        Composants du bilan massique.

    Notes
    -----
    La fonction constitue le noyau d'HSAMI et simule UN SEUL PAS DE TEMPS.
    Elle reçoit donc en entrée les états du pas de temps précédent.

    projet : un dictionnaire contenant les clés suivantes :
        - 'superficie' : superficie(1) : Superficie du bassin versant, incluant le réservoir (km^2, scalaire)
            superficie(2) : Superficie moyenne du réservoir (= 0 si non modélisé) (km^2, scalaire).
        - 'param' : liste des float, Paramètres pour la simulation.
        - 'mémoire' : int, Mémoire utilisée pour les hydrogrammes unitaires (j, scalaire)
        - 'physio' : dict, les données physiographiques peuvent être vides.
        - 'modules' : dict, les modules pour la simulation peuvent être vides. Les valeurs par défaut sont définies si elles ne sont pas fournies.
        - 'meteo' : dict, données météorologiques pour la simulation..
        - 'dates' : Date de la journée de la simulation (format datevec, vecteur).
        - 'nb_pas_par_jour' : Nombre de pas de temps par jour (scalaire).
        - 'pas' : Pas à l'intérieur de la journée (scalaire).

    projet['param ']             Paramétres d'HSAMI (scalaires)
        param[0] : Efficacité évapo été (adim.)
        param[1] : Efficacité évapo hiver (adim.)
        param[2] : Taux de fonte jour (cm/degC/j)
        param[3] : Taux de fonte nuit (cm/degC/j)
        param[4] : Température fonte jour (degC)
        param[5] : Température fonte nuit (degC)
        param[6] : Température référence pluie (degC)
        param[7] : Effet redoux sur aire enneigée (adim.)
        param[8] : Effet gel (adim.)
        param[9] : Effet sol (adim.)
        param[10] : Seuil_min (cm)
        param[11] : Sol_min (cm)
        param[12] : Sol_max (cm)
        param[13] : Nappe_max (cm)
        param[14] : Portion ruissellement surface (fraction)
        param[15] : Portion ruissellement sol max (fraction)
        param[16] : Taux vidange sol (cm/j)
        param[17] : Taux vidange nappe (cm/j)
        param[18] : Taux vidange inter (cm/j)
        param[19] : Mode hydrogramme surface
        param[20] : Forme hydrogramme surface
        param[21] : Mode hydrogramme intermédiaire
        param[22] : Forme hydrogramme intermédiaire
        param[23] : Curve Number
        param[24] : Puissance de la cond. hydraulique saturée (Ks) de la couche 1 (Ks [=] cm/j)
        param[25] : Potentiel matriciel au front mouillant (cm)
        param[26] : Coeff. de récession de la nappe
        param[27] : Rendement spécifique de la nappe
        param[28] : Taux de fonte milieu 1 (coniféres)
        param[29] : Taux de fonte milieu 2 (feuillus)
        param[30] : Taux de fonte milieu 3 (autres)
        param[31] : Température de fonte milieu 1 (coniféres)
        param[32] : Température de fonte milieu 2 (feuillus)
        param[33] : Température de fonte milieu 3 (autres)
        param[34] : Puissance de Ks pour l'infiltration
        param[35] : Capacité de retenue de la neige
        param[36] : Indice de distribution de la taille des pores (b), couche 1
        param[37] : Indice de distribution de la taille des pores (b), couche 2
        param[38] : Puissance de K_s, couche 2
        param[39] : épaisseur couche 1 (cm)
        param[40] : épaisseur couche 2 (cm)
        param[41] : Point de flétrissement permanent, couche 1
        param[42] : Capacité au champ, couche 1
        param[43] : Capacité au champ, couche 2
        param[44] : Porosité couche 1
        param[45] : Porosité couche 2m
        param[46] : Coefficient de Stefan (k)
        param[47] : Coefficient pour calcul du volume max du MHE (hmax)
        param[48] : Coefficient pour détermination de la surface normale (30 dans HYDROTEL)(p_norm)
        param[49] : Puissance de la conductivité hydraulique à saturation à la base du MHE (mm/j)

    projet['meteo']['bassin']  Vecteur météo
        tmin [cm]
        tmax [cm]
        pluie [cm]
        neige [cm]
        soleil [fraction] (optionnel)
        een [cm] (optionnel, défaut = -1)

        projet['meteo']['reservoir']
        ** S'il n'existe aucune météo au réservoir, celle  du bassin est utilisée. **
        Vecteur météo :
        tmin [cm]
        tmax [cm]
        pluie [cm]
        neige [cm]
        soleil [fraction] (optionnel)

    projet['modules']            Structure spécifiant les modules é utiliser
        etp_bassin :
        'hsami' (défaut)
        'blaney_criddle'
        'hamon'
        'linacre'
        'kharrufa'
        'mohyse'
        'romanenko'
        'makkink'
        'turc'
        'mcguinness_bordne'
        'abtew'
        'hargreaves'
        'priestley-taylor'

        etp_reservoir :
        'hsami' (défaut)
        'blaney_criddle'
        'hamon'
        'linacre'
        'kharrufa'
        'mohyse'
        'romanenko'
        'makkink'
        'turc'
        'mcguinness_bordne'
        'abtew'
        'hargreaves'
        'priestley-taylor'

        een :
        'hsami' (défaut)
        'dj'
        'mdj'
        'alt'

        infiltration :
        'hsami' (défaut)
        'green_ampt'
        'scs_cn'

        sol :
        'hsami' (défaut)
        '3couches'

        qbase :
        'hsami' (défaut)
        'dingman'

        radiation :
        'hsami' (défaut)
        'mdj'

        mhumide:
        0  (défaut)
        1

        reservoir :
        0 (défaut)
        1

        glace_reservoir :
        0 (défaut)
        'stefan'
        'mylake'

    projet['physio']
    Structure contenant des variables physiographiques du bassin
    ** Toutes les variables physiographiques peuvent
    étre facultatives, dépamment des modules utilisés.
    latitude :         Latitude moyenne (degrés, scalaire)
    altitude :         Altitude moyenne (m, scalaire)
    albedo_sol :       Albédo du sol (fraction décimale, scalaire)
    i_orientation_bv : Indice d'orientation
    pente_bv :         Pente moyenne (degrés, scalaire)
    occupation :       Fractions d'occupation des milieux (fraction, vecteur de 1x2 ou 1x3)
    niveau :           Niveau du réservoir (m, scalaire)
    coeff :            Coefficients de l'équation de la courbe d'emmagasinement (vecteur 1 x 3)
    samax :            Surface maximale du MHE (km2, scalaire)
    occupation_bande : Pourcentage d'occupation par bande d'altitude
    altitude_bande :   Altitude de chaque bande

    etat : états du bassin versants et du réservoir
        - etat['eau_hydrogrammes']            : Eau en transit dans les HU (cm, matrice de mémoire x 2, ou mémoire x 3 si module['mhumide'] = 1)
        - etat['neige_au_sol ']               : équivalent en eau de la neige au bassin versant (cm, scalaire)
        - etat['fonte']                       : Eau libre dans la neige (cm, scalaire)
        - etat['nas_tot']                     : Neige au sol totale (cm, scalaire)
        - etat['fonte_tot']                   : Fonte totale (cm, scalaire)
        - etat['derniere_neige']              : Nombre de jours depuis la derniére neige (j, scalaire)
        - etat['gel']                         : Eau gelée dans le sol (cm, scalaire)
        - etat['sol']                         : Eau dans la zone non saturée (cm, vecteur de 1x2 si modules['sol = '3couches')
        - etat['nappe']                       : Eau dans la zone saturée (cm, scalaire)
        - etat['reserve']                     : Eau dans la réserve intermédiaire (cm, scalaire)
        - etat['mdj']['couvert_neige']        : Hauteur du couvert de neige dans les 2 ou 3 milieux (m, vecteur de la même
                                                taille que physio.occupation)
        - etat['mdj']['densite_neige']        : Densité du couvert de neige (fraction décimale, vecteur de la même taille que physio.occupation)
        - etat['mdj']['albedo_neige']         : Albédo de la neige (fraction, vecteur de la même taille que physio.occupation)
        - etat['mdj']['neige_au_sol']         : équivalent en eau de la neige (m, vecteur de la même taille que physio.occupation)
        - etat['mdj']['fonte']                : Eau libre dans la neige (m, vecteur de la même taille que physio.occupation)
        - etat['mdj']['gel']                  : Eau gelée dans le sol (cm, vecteur de la même taille que physio.occupation)
        - etat['mdj']['sol']                  : Eau dans la zone non saturée (cm, vecteur de la même taille que physio.occupation)
        - etat['mdj']['energie_neige']        : Bilan énergétique de la neige (J/m2, vecteur de la même taille que physio.occupation)
        - etat['mdj']['energie_glace']        : Bilan énergétique de la glace (J/m2, scalaire)
        - etat['alt']['couvert_neige']        : Hauteur du couvert de neige dans les 2 ou 3 milieux (m, vecteur de la même taille
                                                que physio.occupation_bande)
        - etat['alt']['densite_neige']        : Densité du couvert de neige (fraction décimale, vecteur de la même taille
                                                que physio.occupation_bande)
        - etat['alt']['albedo_neige']         : Albédo de la neige (fraction, vecteur de la même taille que physio.occupation_bande)
        - etat['alt']['neige_au_sol']         : équivalent en eau de la neige (m, vecteur de la même taille que physio.occupation_bande)
        - etat['alt']['fonte']                : Eau libre dans la neige (m, vecteur de la même taille que physio.occupation_bande)
        - etat['alt']['gel']                  : Eau gelée dans le sol (cm, vecteur de la même taille que physio.occupation_bande)
        - etat['alt']['sol']                  : Eau dans la zone non saturée (cm, vecteur de la même taille que physio.occupation_bande)
        - etat['alt']['energie_neige']        : Bilan énergétique de la neige (J/m2, vecteur de la même taille que physio.occupation_bande)
        - etat['alt']['energie_glace']        : Bilan énergétique de la glace (J/m2, scalaire)
        - etat['mh_vol']                      : Volume du milieu humide (m3, scalaire)
        - etat['ratio_MH']                    : Ratio du milieu humide dans le bassin versant (fraction décimale, scalaire)
        - etat['mh_surf']                     : Superficie du milieu humide (ha, scalaire)
        - etat['mhumide']                     : Lame d'eau du milieu humide (cm, scalaire)
        - etat['ratio_qbase']                 : Ratio du debit de base provenant du milieu humide (fraction décimale, scalaire)
        - etat['cumdegGel']                   : Cumulatif de degrés-jour de gel (degC, scalaire)
        - etat['obj_gel']                     : Objectif de degrés-jour de gel (degC, scalaire)
        - etat['dernier_gel']                 : Nombre de jour depuis le dernier gel (j, scalaire)
        - etat['reservoir_epaisseur_glace']   : épaisseur de la glace sur le réservoir (cm, scalaire)
        - etat['reservoir_energie_glace']     : Bilan énergétique de la glace sur le réservoir (J/m2, scalaire)
        - etat['reservoir_superficie']        : Superficie du réservoir (km2, scalaire)
        - etat['reservoir_superficie_glace']  : Superficie de berges avec de la glace déposée (km2, scalaire)
        - etat['reservoir_superficie_ref']    : Superficie de référence en début d'hiver (km2, scalaire)
        - etat['eeg']                         : équivalent en eau de la glace déposée en berges (cm, vecteur de 3000x1)
        - etat['ratio_bassin']                : Fraction de la superficie occupée par la partie terrestre du bassin versant
                                                (fraction décimale, scalaire)
        - etat['ratio_reservoir']             : Fraction de la superficie occupée par le réservoir (fraction décimale, scalaire)
        - etat['ratio_fixe']                  : Fraction de la superficie occupée par la partie terrestre pour le calcul des processus
                                                souterrains (fraction décimale, scalaire)

    s
        s['Qtotal']
        s['Qbase']
        s['Qinter']
        s['Qsurf']
        s['Qreservoir']
        s['Qglace']
        s['Qmh']
        s['ETP']
        s['ETRtotal']
        s['ETRsublim']
        s['ETRPsurN']
        s['ETRintercept']
        s['ETRtranspir']
        s['ETRreservoir']
        s['ETRmhumide']

    delta   : fermeture du bilan hydrologique pour la fonction principale et les sous-fonctions
        - delta['total']         : Fermeture de la fonction principale (cm, scalaire)
        - delta['glace']         : Fermeture de la fonction hsami_glace (cm, scalaire)
        - delta['interception']  : Fermeture de la fonction hsami_interception (cm, scalaire)
        - delta['ruissellement'] : Fermeture de la fonction hsami_ruissellement (cm, scalaire)
        - delta['vertical']      : Fermeture de la fonction hsami_ecoulement_vertical (cm, scalaire)
        - delta['horizontal']    : Fermeture de la fonction hsami_ecoulement_horizontal (cm, scalaire)
        - delta['mhumide']       : Fermeture de la fonction hsami_mhumide (cm, scalaire)
    """
    # --------------
    # Pré-traitement
    # --------------
    nb_pas = projet["nb_pas_par_jour"]

    # Conversion de la latitude en radians
    if "physio" in projet:
        physio = projet["physio"]
        if "latitude" in physio:
            physio["latitude"] = physio["latitude"] * np.pi / 180

    # Vérification que tmin < tmax
    meteo = projet["meteo"]
    if meteo["bassin"][0] > meteo["bassin"][1]:
        meteo["bassin"][0:2] = np.flip(meteo["bassin"][0:2])

    # Vérification que tmin < tmax pour le réservoir
    if "reservoir" in meteo:
        if meteo["reservoir"][0] > meteo["reservoir"][1]:
            meteo["reservoir"][0:2] = np.flip(meteo["reservoir"][0:2])

    superficie = projet["superficie"]
    param = projet["param"]
    modules = projet["modules"]

    if modules.get("een"):
        if modules["een"] == "mdj":
            # Verifier si la somme des occupations = 1.
            if np.sum(physio["occupation"][:]) != 1:
                warnings.warn("La somme des occupations n" "est pas égale à 1")

        if modules["een"] == "alt":
            # Verifier si la somme des occupations = 1.
            if np.sum(physio["occupation_bande"][:]) != 1:
                warnings.warn("La somme des occupations n" "est pas égale à 1")

    # -------------------------------------------
    # Sauvegarde des états initiaux pour le bilan
    # -------------------------------------------
    bilan = {}
    if etat["eeg"].size == 0:
        etat["eeg"] = 0

    etats_ini = (
        etat["ratio_bassin"] * etat["neige_au_sol"]
        + etat["ratio_fixe"]
        * (etat["gel"] + etat["nappe"] + np.nansum(etat["sol"]) + etat["mhumide"])
        + np.nansum(etat["eeg"]) / superficie[0]
    )

    reserv_ini = etat["ratio_fixe"] * etat["reserve"]
    eaux_hu_ini = etat["ratio_fixe"] * (
        np.nansum(etat["eau_hydrogrammes"][:, 0] + etat["eau_hydrogrammes"][:, 2])
        + np.nansum(etat["eau_hydrogrammes"][0:9, 1])
    )

    (
        etat,
        eau_surface,
        dem_eau,
        etps,
        etr,
        apport_vert,
        glace_vers_res,
        bassin_vers_res,
        bilan,
    ) = etp_glace_interception(
        projet, param, modules, physio, superficie, meteo, nb_pas, etat, bilan
    )

    etat, q, etp_tot, etr_tot, etr, bilan = ruissellement_ecoulement(
        projet,
        param,
        modules,
        physio,
        superficie,
        nb_pas,
        etat,
        etps,
        eau_surface,
        dem_eau,
        etr,
        apport_vert,
        glace_vers_res,
        bassin_vers_res,
        bilan,
    )

    s, etat, delta = bilan_sorties(
        modules,
        meteo,
        superficie,
        nb_pas,
        etat,
        q,
        etp_tot,
        etr_tot,
        etr,
        reserv_ini,
        etats_ini,
        eaux_hu_ini,
        bilan,
    )

    return s, etat, delta


def etp_glace_interception(
    projet, param, modules, physio, superficie, meteo, nb_pas, etat, bilan
):
    """
    HSAMI_ETP, HSAMI_GLACE, HSAMI_INTERCEPTION.

    Parameters
    ----------
    projet : dict
        Projet.
    param : list
        Paramètres pour la simulation.
    modules : dict
        Les modules pour la simulation.
    physio : dict
        Les données physiographiques.
    superficie : list
        La superficie du bassin versan et  la uuperficie moyenne du réservoir.
    meteo : dict
        Données météorologiques pour la simulation.
    nb_pas : int
        Nombre de pas de temps.
    etat : dict
        États du bassin versants et du réservoir.
    bilan : dict
        Bilan hydrologique.

    Returns
    -------
    etat : dict
        États du bassin versants et du réservoir.
    eau_surface : float
        Eau disponible à la surface pour évaporation, ruissellement et infiltration.
    demande_eau : float
        Demande en eau restante.
    etps : list,
        Évapotranspiration au bassin et réservoir.
    etr : list
        Évapotranspiration et évaporation.
    apport_vertical : list
        Lames d'eau à moduler par les hydrogrammes unitaires.
    glace_vers_reservoir : float
        Lame d'eau transitant de la glace de rive vers le réservoir pour le pas de temps (cm).
    bassin_vers_reservoir : float
        Lame d'eau transitant du réservoir vers la glace de rive pour le pas de temps (cm).
    bilan : dict,
        Bilan hydrologique.

    Notes
    -----
    HSAMI was developed by J.L. Bisson, and F. Roberge in MATLAB, 1983. It was then
    modified and improved by Catherine Guay, Marie Minville, Isabelle Chartier
    and Jonathan Roy, 2013-2017 to become HSAMI+. Translated into Python by
    Didier Haguma, 2024.
    """
    # projet, param, modules, physio, superficie, etat
    # ------------
    # 1. HSAMI_ETP
    # ------------
    date = datetime(*projet["date"])
    pas = projet["pas"]
    jj = int(date.strftime("%j"))

    # ETP au bassin
    etps = [0, 0]  # Define the variable "etps"
    etps[0] = hsami_etp(
        pas,
        nb_pas,
        jj,
        meteo["bassin"][0],
        meteo["bassin"][1],
        modules["etp_bassin"],
        physio,
    )

    # ETP au réservoir
    etps[1] = hsami_etp(
        pas,
        nb_pas,
        jj,
        meteo["reservoir"][0],
        meteo["reservoir"][1],
        modules["etp_reservoir"],
        physio,
    )

    # --------------
    # 2. HSAMI_GLACE
    # --------------
    # entrees
    bilan["glace"] = {}
    bilan["glace"]["etat"] = [0, 0]
    bilan["glace"]["entrees"] = 0
    bilan["glace"]["etat"][0] = np.sum(etat["eeg"])

    if modules["glace_reservoir"] == "stefan" or modules["glace_reservoir"] == "mylake":
        if "niveau" in physio:
            if physio["niveau"] is not None:
                glace_vers_reservoir, bassin_vers_reservoir, etat = hsami_glace(
                    modules, superficie, etat, meteo, physio, param
                )
            else:
                glace_vers_reservoir, bassin_vers_reservoir, etat = hsami_glace(
                    modules, superficie, etat
                )
        else:
            glace_vers_reservoir, bassin_vers_reservoir, etat = hsami_glace(
                modules, superficie, etat
            )

    elif modules["glace_reservoir"] == 0:
        glace_vers_reservoir, bassin_vers_reservoir, etat = hsami_glace(
            modules, superficie, etat
        )
    else:
        raise ValueError(
            "L"
            "option spécifiée pour la modélisation de la glace de réservoir est invalide"
        )

    # sorties
    bilan["glace"]["sorties"] = glace_vers_reservoir
    bilan["glace"]["etat"][1] = np.sum(etat["eeg"])

    # ---------------------
    # 3. HSAMI_INTERCEPTION
    # ---------------------
    # entrées
    bilan["interception"] = {}
    bilan["interception"]["etat"] = [0, 0]
    bilan["interception"]["entrees"] = np.sum(meteo["bassin"][2:4]) + np.sum(
        meteo["reservoir"][2:4]
    )
    bilan["interception"]["etat"][0] = (
        etat["neige_au_sol"]
        + etat["gel"]
        + np.nansum(etat["sol"])
        + np.nansum(etat["eeg"])
    )

    eau_surface, demande_eau, etat, etr, apport_vertical = hsami_interception(
        nb_pas, jj, param, meteo, etps, etat, modules, physio
    )

    # sorties
    bilan["interception"]["sorties"] = (
        eau_surface + np.nansum(etr) + np.nansum(apport_vertical[[3, 4]])
    )
    bilan["interception"]["etat"][1] = (
        etat["neige_au_sol"]
        + etat["gel"]
        + np.nansum(etat["sol"])
        + np.nansum(etat["eeg"])
    )

    return (
        etat,
        eau_surface,
        demande_eau,
        etps,
        etr,
        apport_vertical,
        glace_vers_reservoir,
        bassin_vers_reservoir,
        bilan,
    )


def ruissellement_ecoulement(
    projet,
    param,
    modules,
    physio,
    superficie,
    nb_pas,
    etat,
    etps,
    eau_surface,
    demande_eau,
    etr,
    apport_vertical,
    glace_vers_reservoir,
    bassin_vers_reservoir,
    bilan,
):
    """
    HSAMI_RUISSELLEMENT_SURFACE, HSAMI_ECOULEMENT_VERTICAL, HSAMI_ECOULEMENT_HORIZONTAL.

    Parameters
    ----------
    projet : dict
        Projet.
    param : list
        Paramètres pour la simulation.
    modules : dict
        Les modules pour la simulation.
    physio : dict
        Les données physiographiques.
    superficie : list
        La superficie du bassin versan et  la uuperficie moyenne du réservoir.
    nb_pas : int
        Nombre de pas de temps.
    etat : dict
        États du bassin versants et du réservoir.
    etps : liste
        Évapotranspiration au bassin et réservoir.
    eau_surface : float
        Eau disponible à la surface pour évaporation, ruissellement et infiltration.
    demande_eau : float
        Demande en eau restante.
    etr : list
        Évapotranspiration et évaporation.
    apport_vertical : list
        Lames d'eau à moduler par les hydrogrammes unitaires.
    glace_vers_reservoir : float
        Lame d'eau transitant de la glace de rive vers le réservoir pour le pas de temps (cm).
    bassin_vers_reservoir : float
        Lame d'eau transitant du réservoir vers la glace de rive pour le pas de temps (cm).
    bilan : dict
        Bilan hydrologique.

    Returns
    -------
    etat : dict
        États du bassin versants et du réservoir.
    q : list
        Débits provenant du bassin.
    etp_tot : float
        Évapotranspiration totale.
    etr_tot : float
        Évapotranspiration réelle totale.
    bilan : dict
        Bilan hydrologique.
    """
    # ------------------------------
    # 4. HSAMI_RUISSELLEMENT_SURFACE
    # ------------------------------
    # entrées
    bilan["ruissellement"] = {}
    bilan["ruissellement"]["etat"] = [0, 0]
    bilan["ruissellement"]["entrees"] = eau_surface
    bilan["ruissellement"]["etat"][0] = 0

    ruissellement_surface, infiltration = hsami_ruissellement_surface(
        nb_pas, param, etat, eau_surface, modules
    )

    # sorties
    bilan["ruissellement"]["sorties"] = ruissellement_surface + infiltration
    bilan["ruissellement"]["etat"][1] = 0

    # ----------------------------
    # 5. HSAMI_ECOULEMENT_VERTICAL
    # ----------------------------
    # Ajustement de l'offre et de la demande pour le passage d'une superficie
    # variable à une superficie fixe
    infiltration = infiltration * etat["ratio_bassin"] / etat["ratio_fixe"]
    demande_eau = demande_eau * etat["ratio_bassin"] / etat["ratio_fixe"]
    ruissellement = ruissellement_surface * etat["ratio_bassin"] / etat["ratio_fixe"]

    # entrées
    bilan["vertical"] = {}
    bilan["vertical"]["etat"] = [0, 0]
    bilan["vertical"]["entrees"] = infiltration + ruissellement
    bilan["vertical"]["etat"][0] = (
        etat["neige_au_sol"] + etat["gel"] + etat["nappe"] + np.nansum(etat["sol"])
    )

    apport_vertical, etat, etr = hsami_ecoulement_vertical(
        nb_pas,
        param,
        etat,
        infiltration,
        demande_eau,
        modules,
        ruissellement,
        apport_vertical,
        etr,
    )

    # sorties
    bilan["vertical"]["sorties"] = np.nansum(apport_vertical[0:3]) + etr[2] + etr[3]
    bilan["vertical"]["etat"][1] = (
        etat["neige_au_sol"] + etat["gel"] + etat["nappe"] + np.nansum(etat["sol"])
    )

    if modules["mhumide"] == 1:
        # entrées
        bilan["mhumide"] = {}
        bilan["mhumide"]["etat"] = [0, 0]
        bilan["mhumide"]["entrees"] = np.nansum(apport_vertical) + np.nansum(etr)
        bilan["mhumide"]["etat"][0] = etat["mhumide"]

        apport_vertical, etat, etr = hsami_mhumide(
            apport_vertical, param, etat, demande_eau, etr, physio, superficie
        )

        # sorties
        bilan["mhumide"]["sorties"] = np.nansum(apport_vertical) + np.nansum(etr)
        bilan["mhumide"]["etat"][1] = etat["mhumide"]

        # Ajustement de l'évapo pour le passage d'une superficie fixe à une superficie variable
        etr[5] = etr[5] * etat["ratio_fixe"] / etat["ratio_bassin"]

    # Ajustement du pompage et de l'évapo pour le passage d'une superficie fixe
    # à une superficie variable
    etr[2] = etr[2] * etat["ratio_fixe"] / etat["ratio_bassin"]
    etr[3] = etr[3] * etat["ratio_fixe"] / etat["ratio_bassin"]

    # ------------------------------
    # 6. HSAMI_ECOULEMENT_HORIZONTAL
    # ------------------------------
    # Note : apport_vertical[3] et apport_vertical[4] ne sont pas laminés
    # Fonction d'imposer des hydrogrammes.

    if "hu_surface" in projet:
        if len(projet["hu_surface"]) != projet["memoire"]:
            print(
                "L"
                "hydrogramme unitaire de surface imposé doit avoir la même durée que la variable mémoire"
            )

        hydrogrammes = np.zeros((len(projet["hu_surface"]), 2))
        hydrogrammes[:, 0] = projet[
            "hu_surface"
        ]  # Note : Les paramétres 19 et 20 ne seront pas utilisés
    else:
        hydrogrammes = hsami_hydrogramme(
            param[19], param[20], nb_pas, projet["memoire"] / nb_pas
        )  # hu surface

    if "hu_inter" in projet:
        if len(projet["hu_inter"]) != projet["memoire"]:
            print(
                "L"
                "hydrogramme unitaire intermédiaire imposé doit avoir la même durée que la variable mémoire"
            )

        hydrogrammes[:, 1] = projet[
            "hu_inter"
        ]  # Note : Les paramétres 21 et 22 ne seront pas utilisés
    else:
        hydrogrammes = np.vstack(
            (
                hydrogrammes,
                hsami_hydrogramme(
                    param[21], param[22], nb_pas, projet["memoire"] / nb_pas
                ),
            )
        ).T  # hu inter

    # entrées
    bilan["horizontal"] = {}
    bilan["horizontal"]["etat"] = [0, 0]
    bilan["horizontal"]["entrees"] = np.sum(apport_vertical)
    bilan["horizontal"]["etat"][0] = (
        np.sum(etat["eau_hydrogrammes"][:, 0] + etat["eau_hydrogrammes"][:, 2])
        + np.sum(etat["eau_hydrogrammes"][0:9, 1])
        + etat["reserve"]
    )

    apport_horizontal, etat["reserve"], etat["eau_hydrogrammes"] = (
        hsami_ecoulement_horizontal(
            nb_pas,
            param[18],
            etat["reserve"],
            etat["eau_hydrogrammes"],
            hydrogrammes,
            apport_vertical,
            modules,
        )
    )

    # sorties
    bilan["horizontal"]["sorties"] = np.sum(apport_horizontal)
    bilan["horizontal"]["etat"][1] = np.sum(np.sum(etat["eau_hydrogrammes"]))

    # ------------------
    # Calcul des sorties
    # ------------------
    # Facteurs de conversion pour les debits
    facteur_fixe = superficie[0] * etat["ratio_fixe"] / 8.64  # Pour les cm/j en m3/s
    facteur_reservoir = (
        superficie[0] * etat["ratio_reservoir"] / 8.64
    )  # Pour les cm/j en m3/s

    # debits provenant du bassin (m3/s)
    # ---------------------------------
    q = np.zeros(np.size(apport_horizontal))
    for i_q in [0, 1, 2, 5]:
        q[i_q] = apport_horizontal[i_q] * facteur_fixe

    # debit provenant du réservoir (m3/s)
    # ----------------------------------
    # On retire l'ETR directe au réservoir dans la fonction
    # hsami_interception
    q[3] = (
        apport_horizontal[3] * facteur_reservoir
        + bassin_vers_reservoir * superficie[0] / 8.64
    )

    # debit provenant de la glace (m3/s)
    # ---------------------------------
    # apports_horizontaux[:,5] et glace_vers_reservoir sont déjà à l'échelle du
    # bassin versant (divisés par superficie(1))
    q[4] = (apport_horizontal[4] + glace_vers_reservoir) / 8.64

    # ETR et ETP total (cm/j)
    # -----------------------
    # L'ETR et l'ETP sont pondérées par le ratio de superficie occupée par le
    # réservoir.

    if modules["mhumide"] == 1:
        etr_tot = (
            np.sum(etr[[0, 1, 2, 3, 5]]) * etat["ratio_bassin"]
            + etr[4] * etat["ratio_reservoir"]
        )
    if modules["mhumide"] == 0:
        etr_tot = (
            np.sum(etr[0:4]) * etat["ratio_bassin"] + etr[4] * etat["ratio_reservoir"]
        )

    etp_tot = etps[0] * etat["ratio_bassin"] + etps[1] * etat["ratio_reservoir"]

    return etat, q, etp_tot, etr_tot, etr, bilan


def bilan_sorties(
    modules,
    meteo,
    superficie,
    nb_pas,
    etat,
    q,
    etp_tot,
    etr_tot,
    etr,
    reserv_ini,
    etats_ini,
    eaux_hu_ini,
    bilan,
):
    """
    CALCUL DU BILAN TOTAL ET CALCUL DU BILAN PAR SOUS-FONCTION.

    Parameters
    ----------
    modules : dict
        Les modules pour la simulation.
    meteo : dict
        Données météorologiques pour la simulation.
    superficie : list
        La superficie du bassin versan et  la uuperficie moyenne du réservoir.
    nb_pas : int
        Nombre de pas de temps.
    etat : dict
        États du bassin versants et du réservoir.
    q : list
        Débits provenant du bassin.
    etp_tot : float
        Évapotranspiration totale.
    etr_tot : float
        Évapotranspiration réelle totale.
    etr : list
        Évapotranspiration et évaporation.
    reserv_ini : float
        Etat initial de la réserve.
    etats_ini : float
        Etat initial du sol .
    eaux_hu_ini : float
        Etat initial du HU.
    bilan : dict
        Bilan hydrologique.

    Returns
    -------
    s : dict
        Sorties de simulation.
    etat : dict
        États du bassin versants et du réservoir.
    delta : dict
        Fermeture du bilan hydrologique.
    """
    # ----------------------------------------------------------------------
    # Création de la structure de sortie (variables pondérées à l'échelle du
    # bassin)
    # ----------------------------------------------------------------------
    s = {}
    s["Qtotal"] = np.sum(q)
    s["Qbase"] = q[0] * (1 - etat["ratio_qbase"])
    s["Qinter"] = q[1]
    s["Qsurf"] = q[2]
    s["Qreservoir"] = q[3]
    s["Qglace"] = q[4]

    if modules["mhumide"] == 1:
        s["Qmh"] = q[0] * etat["ratio_qbase"] + q[5]

    if modules["mhumide"] == 0:
        s["Qmh"] = 0.0 if isinstance(q[0], float) else [0.0] * len(q[0]).tolist()
        np.zeros(np.size(q[0]))

    s["ETP"] = etp_tot
    s["ETRtotal"] = etr_tot
    s["ETRsublim"] = etr[0]
    s["ETRPsurN"] = etr[1]
    s["ETRintercept"] = etr[2]
    s["ETRtranspir"] = etr[3]

    if modules["reservoir"] == 1:
        s["ETRreservoir"] = etr[4]

    if modules["reservoir"] == 0:
        s["ETRreservoir"] = 0

    if modules["mhumide"] == 1:
        s["ETRmhumide"] = etr[5]

    if modules["mhumide"] == 0:
        s["ETRmhumide"] = 0

    # ---------------------
    # CALCUL DU BILAN TOTAL
    # ---------------------
    entrees_bilan = etat["ratio_bassin"] * np.sum(meteo["bassin"][2:4]) + etat[
        "ratio_reservoir"
    ] * np.sum(meteo["reservoir"][2:4])

    etats_bilan = (
        etat["ratio_bassin"] * etat["neige_au_sol"]
        + etat["ratio_fixe"]
        * (np.nansum(etat["sol"]) + etat["gel"] + etat["nappe"] + etat["mhumide"])
        + np.nansum(etat["eeg"]) / superficie[0]
    )

    eaux_hu = etat["ratio_fixe"] * np.sum(np.sum(etat["eau_hydrogrammes"]))

    debit = s["Qtotal"] * 8.64 / superficie[0]

    delta = {}
    delta["total"] = (
        entrees_bilan
        + reserv_ini
        + etats_ini
        + eaux_hu_ini
        - etats_bilan
        - eaux_hu
        - debit
        - s["ETRtotal"]
    )

    # ---------------------------------
    # CALCUL DU BILAN PAR SOUS-FONCTION
    # ---------------------------------
    # Glace
    delta["glace"] = (
        bilan["glace"]["entrees"]
        - bilan["glace"]["sorties"]
        + bilan["glace"]["etat"][0]
        - bilan["glace"]["etat"][1]
    )

    # Interception
    delta["interception"] = (
        bilan["interception"]["entrees"]
        - bilan["interception"]["sorties"]
        + bilan["interception"]["etat"][0]
        - bilan["interception"]["etat"][1]
    )

    # Ruissellement
    delta["ruissellement"] = (
        bilan["ruissellement"]["entrees"]
        - bilan["ruissellement"]["sorties"]
        + bilan["ruissellement"]["etat"][0]
        - bilan["ruissellement"]["etat"][1]
    )

    # Ecoulement vertical
    delta["vertical"] = (
        bilan["vertical"]["entrees"]
        - bilan["vertical"]["sorties"]
        + bilan["vertical"]["etat"][0]
        - bilan["vertical"]["etat"][1]
    )

    # Milieu humide
    if modules["mhumide"] == 1:
        delta["mhumide"] = (
            bilan["mhumide"]["entrees"]
            - bilan["mhumide"]["sorties"]
            + bilan["mhumide"]["etat"][0]
            - bilan["mhumide"]["etat"][1]
        )
    else:
        delta["mhumide"] = 0.0 if nb_pas == 1 else [0.0] * nb_pas.tolist()

    # Ecoulement horizontal
    delta["horizontal"] = (
        bilan["horizontal"]["entrees"]
        - bilan["horizontal"]["sorties"]
        + bilan["horizontal"]["etat"][0]
        - bilan["horizontal"]["etat"][1]
    )

    f = list(delta.keys())
    for i_f in range(len(f)):
        delta[f[i_f]] = np.round(delta[f[i_f]], 10)

    return s, etat, delta
