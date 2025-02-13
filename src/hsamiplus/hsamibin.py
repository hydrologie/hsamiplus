"""Fonction qui lit le fichier du projet, exécute et sauvegarde les sorties d'HSAMI."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from .hsami2 import hsami2


def hsamibin(path, filename):
    """
    Lecture de fichier du projet.

    Parameters
    ----------
    path : str
        Emplacement du fichier de projet, ex ./data.
    filename : str
        Nom du fichier projet, ex projet.json.

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
    Fonction qui lit un projet HSAMI+ en format JSON, exécute HSAMI+, et
    sauvegarde les sorties d'HSAMI+ en format JSON dans le méme répertoire
    que le projet. La fonction peut étre compilée avec le makefile disponible
    dans le répertoire.

    Le modèle HSAMI a été originalement développé par J.L. Bisson, et F. Roberge
    en Matlab, 1983. HSAMI a été modifié et bonifié par Catherine Guay, Marie Minville,
    Isabelle Chartier et Jonathan Roy, 2013-2017 pour devenir HSAMI+. Le code a été
    traduit en Python par Didier Haguma, 2024.
    """
    # Load json files and convert to Python format

    with Path.open(Path(path) / filename) as file:
        projet = json.load(file)

    # Execute hsami2
    date = datetime.date.today()

    s, etats, deltas = hsami2(projet)

    # Write output file
    output = {"S": s, "etats": etats, "deltas": deltas}
    output_json = json.dumps(output, indent=4)

    output_file = "output_" + date.strftime("%d_%m_%Y") + ".json"

    with Path.open(Path(path) / output_file, "w") as file:
        file.write(output_json)

    return s, etats, deltas


if __name__ == "__main__":  # pragma: no cover
    """
    path : str, path to the data directory
    filename : str, projet file name
    """

    # import sys
    import time

    start_time = time.time()

    path = "../../data"

    filename = "projet.json"

    s, etats, deltas = hsamibin(path, filename)

    print(f"Fin, après {time.time() - start_time:.2f} secondes !!! ")
