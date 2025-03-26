"""The function simulates the lateral flow in HSAMI+ model."""

from __future__ import annotations

import numpy as np


def hsami_ecoulement_horizontal(
    nb_pas,
    vidange_reserve_inter,
    reserve_inter,
    eau_hydrogrammes,
    hydrogrammes,
    apport_vertical,
    modules,
):
    """
    Module d'écoulement horizontal.

    Parameters
    ----------
    nb_pas : int
        Nombre de pas de temps.
    vidange_reserve_inter : float
        Taux vidange inter.
    reserve_inter : float
        Eau dans la réserve intermédiaire.
    eau_hydrogrammes : list
        Eau en transit dans les HU.
    hydrogrammes : list
        Hydrogrammes unitaires de surface et intermediaire.
    apport_vertical : list
        Lames d'eau verticales (voir hsami_interception).
    modules : dict
        Les modules pour la simulation.

    Returns
    -------
    apport : float
        Apports verticaux laminés.
    reserve_inter : float
        Eau dans la réserve intermédiaire.
    eau_hydrogrammes : list
        Eau en transit dans les HU.
    """
    # On pondere le taux de vidange en fonction du pas de temps
    vidange_reserve_inter = 1 - (1 - vidange_reserve_inter) / nb_pas

    # Distribution du ruissellement de surface en fonction de l'hydrogramme de surface
    eau_hydrogrammes[:, 0] = (
        eau_hydrogrammes[:, 0] + hydrogrammes[:, 0] * apport_vertical[2]
    )

    if modules["mhumide"] == 1:
        eau_hydrogrammes[:, 2] = (
            eau_hydrogrammes[:, 2] + hydrogrammes[:, 0] * apport_vertical[5]
        )

    # Calcul de l'apport latéral
    apport = [
        apport_vertical[0],
        reserve_inter,
        eau_hydrogrammes[0, 0],
        apport_vertical[3],
        apport_vertical[4],
        eau_hydrogrammes[0, 2],
    ]
    # Ex.: apport = [0.0510, 0.0147, 0.0012, -0.0894, 0, 0]

    # Alimentation de la réserve intermédiaire par le ruissellement intermédiaire
    eau_hydrogrammes[0, 1] = apport_vertical[1]
    eau_inter = np.sum(eau_hydrogrammes[:, 1] * hydrogrammes[:, 1])
    reserve_inter = reserve_inter * vidange_reserve_inter + eau_inter * (
        1 - vidange_reserve_inter
    )
    # Ex.: eau_inter = 0.0060

    # ---------------------------------------
    # Décalage de l'eau dans les hydrogrammes
    # ---------------------------------------

    # Hydrogramme de surface: décalage vers la gauche
    eau_hydrogrammes[:-1, 0] = eau_hydrogrammes[1:, 0]
    eau_hydrogrammes[-1, 0] = 0

    eau_hydrogrammes[:-1, 2] = eau_hydrogrammes[1:, 2]
    eau_hydrogrammes[-1, 2] = 0

    # Hydrogramme intermédiaire: décalage vers la droite
    eau_hydrogrammes[1:, 1] = eau_hydrogrammes[:-1, 1]
    eau_hydrogrammes[0, 1] = 0

    return apport, reserve_inter, eau_hydrogrammes
