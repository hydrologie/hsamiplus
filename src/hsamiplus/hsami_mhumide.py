"""The function simulates flow in wwetlands in HSAMI+ model."""

from __future__ import annotations

import numpy as np


def hsami_mhumide(apport, param, etat, demande, etr, physio, superficie):
    """
    Module de milieux humides.

    Parameters
    ----------
    apport : list
        Lames d'eau verticales (cm, voir hsami_interception).
    param :  list
        Paramètres pour la simulation.
    etat : dict
        États du bassin versants et du réservoir.
    demande : float
        Demande évaporative de l'atmosphére (cm).
    etr : list
        Composantes de l'évapotranspiration (cm, voir hsami_interception).
    physio : dict
        Les données physiographiques peuvent être vides.
    superficie : list
        La superficie du bassin versan et  la uuperficie moyenne du réservoir.

    Returns
    -------
    apport : list
        Lames d'eau verticales (cm, voir hsami_interception).
    etat : dict
        États du bassin versants et du réservoir.
    etr : list
        Composantes de l'évapotranspiration (cm, voir hsami_interception).
    """
    # -----------------------------
    # Identification des paramétres
    # -----------------------------
    hmax = param[47]  # Coefficient pour calcul du volume max du MHE (hmax)
    p_norm = param[
        48
    ]  # Coefficient pour détermination de la surface normale (30# dans HYDROTEL)(p_norm)
    ksat = (
        10 ** param[49]
    )  # Puissance de la conductivité hydraulique é saturation é la base du MHE (cm/j)

    # -----------------------------------
    # Identification des variables d'état
    # -----------------------------------
    v_init = etat["mh_vol"]
    sa = etat["mh_surf"]  # Superficie du MHE au début du pas de temps (hectares)

    sup_bv = (
        superficie[0] * 100
    )  # Surface totale du BV (en hectares)- NE VARIE PAS PDT LA SIMULATION
    sa_max = (
        physio["samax"]
    ) * 100  # Surface max du MHE (en hectares)- NE VARIE PAS PDT LA SIMULATION
    sa_norm = (
        p_norm * sa_max
    )  # Surface normale du MHE (30# de Smax dans HYDROTEL) (en hectares) - NE VARIE PAS PDT LA SIMULATION

    # Calcul de v_max et v_norm
    v_max = hmax * (sa_max * 10000)  # Volume d'eau max dans le MHE (m^3)
    v_norm = p_norm * v_max  # Volume d'eau normal dans le MHE (m^3)
    vmin = 0.5 * v_norm  # Volume d'eau minimal dans le MHE (m^3)

    # Calcul des coefficients alpha et beta
    alpha = (np.log10(sa_max) - np.log10(sa_norm)) / (
        np.log10(v_max) - np.log10(v_norm)
    )  # Ex.: alpha = 1.000
    beta = sa_max / (v_max**alpha)  # Ex.: 1.0000e-04

    # ===================
    # Ecoulement vertical
    # ===================
    qb = apport[0]  # écoulement de base vers le MH en cm
    qi = apport[1]  # écoulement latéral vers le MH en cm
    qs = apport[2]  # écoulement de surf. vers le MH en cm

    # --------------------------------------------------
    # Calcul du volume d'eau qui entre dans le MH - Vin
    # --------------------------------------------------
    vb = qb * sa * 100  # en m^3
    vi = qi * sa * 100  # en m^3
    vs = qs * sa * 100  # en m^3

    # ------------------------------------------
    # Calcul du volume de ruissellement - Vsurf
    # ------------------------------------------
    # Le volume et débit de ruissellement sont calculés en prenant é la base un
    # vsurf de 0 et en recalculant le nouveau volume du MH. En fonction de la
    # valeur du MH et sa comparaison avec les seuil v_norm et v_max, on établit la
    # valeur de vsurf et ainsi le débit et volume de ruissellement.

    v_actuel = v_init + vb + vi + vs  # Ex.: v_actuel = 2.4645e+07

    if v_actuel <= v_norm:
        vsurf = 0

    elif v_actuel <= v_max:
        vsurf = (v_actuel - v_norm) / 10  # Ex.: vsurf = 3.4845e+04

    elif v_actuel > v_max:
        vsurf = (v_actuel - v_max) + (v_max - v_norm) / 10

    v_actuel = v_actuel - vsurf

    # --------------------------------------
    # Calcul du volume d'eau évaporé - vevap
    # ---------------------------------------
    # La demande en evaporation est comblée é cette étape. étant donnée que
    # c'est un MH assimilé é un lac non connecté, la demande devrait toujours
    # étre comblée.
    # on n'offre en evap que le v_initial - v_normal + Vsurface)

    offre_evap = (v_actuel - vmin) / (sa * 100)

    if offre_evap > demande:
        vevap = demande * sa * 100  # Ex.: vevap = 2.2023e+04
    else:
        vevap = offre_evap * sa * 100

    v_actuel = v_actuel - vevap

    # -------------------------------------------------
    # Calcul du volume sortant à la base du MH - vseep
    # -------------------------------------------------
    # é cette étape, on calcule le débit et volume de base
    # offre_seep = ce qu'il reste dans le MHE aprés l'évap.

    demande_seep = ksat * sa * 100

    offre_seep = v_actuel - vmin

    if offre_seep > demande_seep:
        vseep = demande_seep  # Ex.: vseep = 2.4633e+03
    else:
        vseep = offre_seep

    v_actuel = v_actuel - vseep

    # ----------------------------------------
    # Calcul de la surface et du volume du MHE
    # ----------------------------------------
    # é partir du nouveau volume, la nouvelle surface du MH peut-étre déterminée
    # Cette surface sera donc réutilisée au prochain pas de temps

    etat["mh_surf"] = beta * (v_actuel**alpha)
    etat["mh_vol"] = v_actuel

    # -------------------------------------------------------
    # Calcul des Returnss pondérées au bassin versant et au MH
    # -------------------------------------------------------
    # Returnss du MH
    qbase_mh = np.round(
        vseep * etat["ratio_MH"] / (sa * 100), 10
    )  # Ex.: qbase_mh = 9.3306e-05
    qsurf_mh = vsurf * etat["ratio_MH"] / (sa * 100)  # Ex.: qsurf_mh = 0.0013
    etr_mh = np.round(
        vevap * etat["ratio_MH"] / (sa * 100), 10
    )  # Ex.: etr_mh = 8.3422e-04

    # Returnss du BV pondérées

    qbase_bv = apport[0] * (1 - etat["ratio_MH"])
    qintr_bv = apport[1] * (1 - etat["ratio_MH"])
    qsurf_bv = apport[2] * (1 - etat["ratio_MH"])

    # Returnss totales

    apport = [
        qbase_mh + qbase_bv,
        qintr_bv,
        qsurf_bv,
        apport[3],
        apport[4],
        qsurf_mh,
    ]  # Ex.: apport = [0.0507, 0, 0, -0.0894, 0, 0.0013]
    etr = np.append(etr, etr_mh)
    etat["ratio_qbase"] = qbase_mh / (
        qbase_bv + qbase_mh
    )  # Ex.: etat.ratio_qbase = 0.0018

    # Recalcul des ratios
    etat["ratio_MH"] = etat["mh_surf"] / sup_bv  # Ex.: 0.0093
    etat["mhumide"] = (
        etat["mh_vol"] * etat["ratio_MH"] / (etat["mh_surf"] * 100)
    )  # Ex.: 0.9313

    return apport, etat, etr
