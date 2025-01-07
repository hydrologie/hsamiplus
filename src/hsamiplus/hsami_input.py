"""Create hsami input 'projet' file."""

import json
import os
import re
from datetime import date
from pathlib import Path

import pandas as pd


def make_project(data_dir, basin_file, param_file, projet_file):
    """
    Traitement des données d'entrée de HSAMI+.

    Parameters
    ----------
    data_dir : str
        Répertoire contenant les fichiers de données.
    basin_file : str
        Chemin d'accès au fichier de données du bassin.
    param_file : str
        Chemin d'accès au fichier de paramètres.
    projet_file : str
        Chemin d'accès au fichier de projet de sortie.

    Notes
    -----
    Fonction principale pour traiter les données de bassin, les paramètres et les données météorologiques, et
    générer un fichier de projet pour HSAMI.

    La fonction effectue les étapes suivantes:
    1. Lit et traite les données du bassin à partir du fichier_bassin.
    2. Configure divers modules et leurs configurations.
    3. Lit et traite les paramètres du param_file.
    4. Configure les données physiologiques du bassin.
    5. Lit et traite les données météorologiques des fichiers spécifiés.
    6. Compile toutes les données dans un dictionnaire de projet.
    7. Écrit le dictionnaire du projet dans le fichier projet_file spécifié au format JSON.
    8. Imprime un message de confirmation une fois la création réussie du fichier de projet.
    """
    # -----------------------------------------------------------------------------------------
    # Données du bassin versant
    # -----------------------------------------------------------------------------------------
    with Path.open(basin_file) as f:
        lines = f.readlines()

    donnees_bv = {}

    for line in lines:
        line_split = line.split(":")
        key = line_split[0].strip()
        values = re.split(r"[,;\s+]", line_split[1:][0])
        values = [v for v in values if v != ""]

        if (".csv" in line) or (".txt" in line):
            donnees_bv[key] = values[0]
        else:
            if key == "ID":
                donnees_bv[key] = values[0]
            elif len(values) == 1:
                donnees_bv[key] = float(values[0])
            else:
                donnees_bv[key] = [float(v) for v in values]

    #  Modules
    # -----------------------------------------------------------------------------------------
    een = "dj"  # hsami, mdj, dj, alt
    etp_bassin = (
        "priestley_taylor"  # hsami, blaney_criddle, hamon, linacre, kharuffa, mohyse,
    )
    # romanenko, makkink, mcguinness_bordne,
    # abtew, hargreaves, priestley_taylor
    etp_reservoir = "priestley_taylor"  # hsami,blaney_criddle, hamon, linacre, ...
    glace_reservoir = "stefan"  # 0, stefan, mylake
    infiltration = "green_ampt"  # hsami, green_ampt, scs_cn
    mhumide = 1  # 0, 1
    qbase = "dingman"  # hsami,dingman
    radiation = " mdj"  # hsami, mdj
    reservoir = 1  # 0, 1
    sol = "3couches"  # hsami, 3couches

    modules = {
        "een": een,
        "etp_bassin": etp_bassin,
        "etp_reservoir": etp_reservoir,
        "glace_reservoir": glace_reservoir,
        "infiltration": infiltration,
        "mhumide": mhumide,
        "qbase": qbase,
        "radiation": radiation,
        "reservoir": reservoir,
        "sol": sol,
    }

    # Paramétres
    # -----------------------------------------------------------------------------------------

    params, df_parm = paramshsami(param_file)

    # Physio
    # -----------------------------------------------------------------------------------------

    physio = {
        "latitude": donnees_bv["latitude_bv"],
        "altitude": donnees_bv["altitude_bv"],
        "albedo_sol": donnees_bv["albedo_sol"],
        "i_orientation_bv": int(
            donnees_bv["indice_orientation_bv"]
        ),  # Indice d’orientation du bassin versant.
        "pente_bv": donnees_bv["pente_bv"],  # in degrees 3.0,
        "occupation": donnees_bv[
            "occupation_bv"
        ],  # Fraction d’occupation des milieux forestiers (1x3 ou 1x2,
        # du plus dense au plus ouvert). e.g. [0.129, 0.489, 0.382]
        "coeff": donnees_bv[
            "coeff_reservoir"
        ],  # Coefficients de la courbe polynomiale de degrés 2
        # d’emmagasinement du réservoir (1x3).
        "samax": donnees_bv[
            "surface_maximale_mhe"
        ],  # Superficie maximale du milieu humide équivalent
        # 'occupation_bande':  donnees_bv['occupation_bande'],           # Fraction d’occupation par bande d’altitude (1x5).
        # e.g [0.001, 0.026, 0.29, 0.498, 0.185]
        # 'altitude_bande':    donnees_bv['altitude_bande'],             # Altitudes des bandes (1x5).
        # e.g.[709.8, 599.4, 489.0, 378.6, 268.2],
    }

    if "niveau" in donnees_bv:
        if len(donnees_bv["niveau_reservoir"]) > 0:
            physio["niveau"] = donnees_bv["niveau_reservoir"]  # Niveau du réservoir (m)

    # Meteo
    # -----------------------------------------------------------------------------------------

    df_meteo = pd.read_csv(
        Path(data_dir) / donnees_bv["fichier_meteo_bv"],
        header=0,
        index_col=0,
        sep=",",
        parse_dates=True,
    )

    if donnees_bv["fichier_meteo_bv"] == donnees_bv["fichier_meteo_reservoir"]:
        dates = [
            [dt.year, dt.month, dt.day, dt.minute, dt.second] for dt in df_meteo.index
        ]
        meteo_bsn = df_meteo.to_numpy().tolist()
        meteo_res = meteo_bsn
    else:
        df_meteo_res = pd.read_csv(
            Path(data_dir) / donnees_bv["fichier_meteo_reservoir"],
            header=0,
            sep=",",
        )
        meteo_res = df_meteo_res.to_numpy().tolist()

    meteo = {"bassin": meteo_bsn, "reservoir": meteo_res}
    # meteo = {'bassin': meteo_bsn}

    # Projet
    # -----------------------------------------------------------------------------------------

    projet = {
        "id": donnees_bv["ID"],
        "nb_pas_par_jour": donnees_bv["nb_pas_par_jour"],
        "memoire": donnees_bv["memoire"],
        "superficie": [donnees_bv["superficie_bv"], donnees_bv["superficie_reservoir"]],
        "modules": modules,
        "param": params,
        "physio": physio,
        "meteo": meteo,
        "dates": dates,
        # 'hu_inter' :        donnees_bv['hydrogramme_inter'],
        # 'hu_surface' :      donnees_bv['hydrogramme_surface']
    }

    writejson(projet_file, projet)

    print(f"Le fichier de projet HSAMI {projet_file} a été créé !")


def paramshsami(param_file):
    """
    Définir les valeurs de paramètres par défaut.

    Parameters
    ----------
    param_file : str
        Chemin d'accès au fichier de paramètres à lire.

    Returns
    -------
    params : list
        Liste de valeurs de paramètres par défaut.
    df_param : pandas.DataFrame
        Un DataFrame contenant les données du paramètre avec des colonnes ['Nom', 'min', 'default', 'max'].

    Notes
    -----
    La fonction lit un fichier de paramètres et renvoie une liste de valeurs de paramètres
    par défaut ainsi que le DataFrame complet.
    """
    df_param = pd.read_csv(param_file, header=0, delim_whitespace=True)

    df_param.columns = ["Nom", "min", "default", "max"]

    params = df_param["default"].tolist()

    return params, df_param


def writejson(filename, dict_var):
    """
    Convert a dictionary to a JSON formatted string and writes it to a file.

    Parameters
    ----------
    filename : str
        The name of the file to write the JSON data to.
    dict_var : dict
        The dictionary to convert to JSON format.
    """
    # change format from dict to json
    js = json.dumps(dict_var, indent=4)

    # Open new json file if not exist it will create
    fp = Path.open(filename, "w")

    # write to json file
    fp.write(js)

    # close the connection
    fp.close()


if __name__ == "__main__":
    day = date.today()

    data_dir = "."
    basin_file = "bassin_versant_info.txt"
    param_file = "parametres.txt"
    projet_file = "projet_1_" + day.strftime("%Y%m%d") + ".json"

    basin_file = Path(data_dir) / basin_file
    param_file = Path(data_dir) / param_file
    projet_file = Path(data_dir) / projet_file

    make_project(data_dir, basin_file, param_file, projet_file)
