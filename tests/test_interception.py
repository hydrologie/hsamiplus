import unittest

import numpy as np

from hsamiplus.hsami_interception import (
    albedo_een,
    calcul_densite_neige,
    calcul_erf,
    calcul_indice_radiation,
    conductivite_neige,
    degel_sol,
    dj_hsami,
    gel_neige,
    gel_sol,
    hsami_interception,
    mdj_alt,
    percolation_eau_fonte,
    pluie_neige,
)


class TestHsamiInterception(unittest.TestCase):
    def setUp(self):
        self.nb_pas = 1
        self.pas_de_temps = 24 / self.nb_pas
        self.pdts = self.pas_de_temps * 60 * 60
        self.jj = 245
        lat = 47.1943
        alt = 390.9
        albedo = 0.7
        albedo_neige = 0.5
        self.param = [0] * 50  # Assuming 50 parameters for simplicity
        self.param[0] = 0.5  # efficacite_evapo_ete
        self.param[1] = 0.3  # efficacite_evapo_hiver
        self.param[2] = 0.1  # en cm/degre C/jour taux_fonte_jour
        self.param[3] = 0.05  # en cm/degre C/jour taux_fonte_nuit
        self.param[4] = -4.0  # C temp_fonte_jour
        self.param[5] = -4.0  # C temp_fonte_nuit
        self.param[6] = -2.0  # C temp_ref_pluie
        self.param[7] = 1.1  # adimensionnel effet_redoux_sur_aire_enneigee
        self.param[11] = 0.05  # sol_min

        self.meteo = {
            "bassin": [3.3, 15.5, 12.3, 0.0, 0.5, -1],
            "reservoir": [3.3, 15.5, 12.0, 0.0, 0.5, -1],
        }
        self.physio = {
            "latitude": lat,
            "altitude": alt,
            "albedo_sol": albedo,
            "i_orientation_bv": 3,
            "pente_bv": 1.80,
            "occupation": [0.0830, 0.5030, 0.4140],
            "altitude_bande": [581.0, 530.0, 479.0, 429.0, 379.0],
            "occupation_bande": [0.0030, 0.0150, 0.0430, 0.1940, 0.745],
        }
        self.etp = [0.5, 0.3]
        self.etat = {
            "sol": [5.8012, np.nan],
            "neige_au_sol": 0,
            "neige_au_sol_totale": 0,
            "fonte": 0,
            "fonte_totale": 0,
            "nas_tot": 0,
            "fonte_tot": 0,
            "derniere_neige": 0,
            "eeg": np.zeros(5000),
            "gel": 0,
        }
        n_occupation = len(self.physio["occupation"])
        n_occupation_bande = len(self.physio["occupation_bande"])
        self.etat["mdj"] = {
            "sol": n_occupation * [0.0],
            "neige_au_sol": n_occupation * [self.etat["neige_au_sol"]],
            "couvert_neige": n_occupation * [0.0],
            "densite_neige": n_occupation * [0.5],
            "fonte": n_occupation * [self.etat["fonte"]],
            "gel": n_occupation * [self.etat["gel"]],
            "albedo_neige": n_occupation * [albedo_neige],
            "energie_neige": n_occupation * [0.0],
            "energie_glace": n_occupation * [0.0],
        }
        self.etat["alt"] = {
            "sol": n_occupation_bande * [0.0],
            "neige_au_sol": n_occupation_bande * [self.etat["neige_au_sol"]],
            "couvert_neige": n_occupation_bande * [0.0],
            "densite_neige": n_occupation_bande * [0.5],
            "fonte": n_occupation_bande * [self.etat["fonte"]],
            "gel": n_occupation_bande * [self.etat["gel"]],
            "albedo_neige": n_occupation_bande * [albedo_neige],
            "energie_neige": n_occupation_bande * [0.0],
            "energie_glace": n_occupation_bande * [0.0],
        }

        self.modules = {
            "sol": "hsami",
            "een": "hsami",  # 'mj' 'mdj' 'hsami' 'alt'
            "radiation": "hsami",  # 'hsami' 'mdj'
        }

        self.t_min = self.meteo["bassin"][0]
        self.t_max = self.meteo["bassin"][1]
        self.pluie = self.meteo["bassin"][2]
        self.neige = self.meteo["bassin"][3]

        self.duree = 1 / self.nb_pas
        self.dt_max = self.t_max - self.param[5]

    def test_hsami_interception(self):
        # modules["sol"] = hsami
        eau_surface, demande_eau, etat, etr, apport_vertical = hsami_interception(
            self.nb_pas,
            self.jj,
            self.param,
            self.meteo,
            self.etp,
            self.etat,
            self.modules,
            self.physio,
        )
        self.assertIsInstance(eau_surface, float)
        self.assertIsInstance(demande_eau, float)
        self.assertIsInstance(etat, dict)
        self.assertIsInstance(etr, np.ndarray)
        self.assertEqual(etr.shape, (5,))
        self.assertIsInstance(apport_vertical, np.ndarray)
        self.assertEqual(apport_vertical.shape, (5,))

        # modules["sol"] = 3couches
        self.modules["sol"] = "3couches"
        eau_surface, demande_eau, etat, etr, apport_vertical = hsami_interception(
            self.nb_pas,
            self.jj,
            self.param,
            self.meteo,
            self.etp,
            self.etat,
            self.modules,
            self.physio,
        )
        self.assertIsInstance(eau_surface, float)
        self.assertIsInstance(demande_eau, float)
        self.assertIsInstance(etat, dict)
        self.assertIsInstance(etr, np.ndarray)
        self.assertEqual(etr.shape, (5,))
        self.assertIsInstance(apport_vertical, np.ndarray)
        self.assertEqual(apport_vertical.shape, (5,))

        # len(meteo["bassin"]) < 5
        self.modules["sol"] = "hsami"
        self.meteo = {
            "bassin": [3.3, 15.5, 12.3, 0.0],
            "reservoir": [3.3, 15.5, 12.0, 0.0],
        }
        eau_surface, demande_eau, etat, etr, apport_vertical = hsami_interception(
            self.nb_pas,
            self.jj,
            self.param,
            self.meteo,
            self.etp,
            self.etat,
            self.modules,
            self.physio,
        )
        self.assertIsInstance(eau_surface, float)
        self.assertIsInstance(demande_eau, float)
        self.assertIsInstance(etat, dict)
        self.assertIsInstance(etr, np.ndarray)
        self.assertEqual(etr.shape, (5,))
        self.assertIsInstance(apport_vertical, np.ndarray)
        self.assertEqual(apport_vertical.shape, (5,))

        # modules["sol"] = "3couches"
        self.modules["sol"] == "3couches"
        eau_surface, demande_eau, etat, etr, apport_vertical = hsami_interception(
            self.nb_pas,
            self.jj,
            self.param,
            self.meteo,
            self.etp,
            self.etat,
            self.modules,
            self.physio,
        )
        self.assertIsInstance(eau_surface, float)
        self.assertIsInstance(demande_eau, float)
        self.assertIsInstance(etat, dict)
        self.assertIsInstance(etr, np.ndarray)
        self.assertEqual(etr.shape, (5,))
        self.assertIsInstance(apport_vertical, np.ndarray)
        self.assertEqual(apport_vertical.shape, (5,))

        # Données de EEN
        self.modules["sol"] == "hsami"
        self.meteo = {
            "bassin": [-3.3, 1.5, 2.3, 0.0, 0.5, 19.3],
            "reservoir": [-3.3, 1.5, 2.0, 0.0, 0.5, 19.3],
        }
        eau_surface, demande_eau, etat, etr, apport_vertical = hsami_interception(
            self.nb_pas,
            self.jj,
            self.param,
            self.meteo,
            self.etp,
            self.etat,
            self.modules,
            self.physio,
        )
        self.assertIsInstance(eau_surface, float)
        self.assertIsInstance(demande_eau, float)
        self.assertIsInstance(etat, dict)
        self.assertIsInstance(etr, np.ndarray)
        self.assertEqual(etr.shape, (5,))
        self.assertIsInstance(apport_vertical, np.ndarray)
        self.assertEqual(apport_vertical.shape, (5,))

    def test_hsami_dj_hsami(self):
        # Module hsami
        # ------------
        eau_surface, demande_eau, etat, etr, apport_vertical = dj_hsami(
            self.modules,
            self.meteo,
            self.etat,
            np.zeros(5),
            np.zeros(5),
            self.duree,
            self.param[1],
            self.param[2],
            self.param[3],
            self.param[4],
            self.param[5],
            self.param[6],
            self.param[7],
            self.param[11],
            self.etat["sol"],
            self.t_min,
            self.t_max,
            self.pluie,
            self.neige,
            0.0,  # soleil
            0.0,  # demande_eau
            0.0,  # demande_reservoir
            self.etat["neige_au_sol"],
            self.etat["fonte"],
            self.etat["neige_au_sol_totale"],
            self.etat["fonte_totale"],
            self.etat["derniere_neige"],
            self.etat["eeg"],
            self.etat["gel"],
        )
        self.assertIsInstance(eau_surface, float)
        self.assertIsInstance(demande_eau, float)
        self.assertIsInstance(etat, dict)
        self.assertIsInstance(etr, np.ndarray)
        self.assertEqual(etr.shape, (5,))
        self.assertIsInstance(apport_vertical, np.ndarray)
        self.assertEqual(apport_vertical.shape, (5,))

        # neige_fondue > 0
        eau_surface, demande_eau, etat, etr, apport_vertical = dj_hsami(
            self.modules,
            self.meteo,
            self.etat,
            np.zeros(5),
            np.zeros(5),
            self.duree,
            self.param[1],
            self.param[2],
            self.param[3],
            self.param[4],
            self.param[5],
            self.param[6],
            self.param[7],
            self.param[11],
            1.9427,  # etat["sol"]
            -17.2,  # t_min
            13.3,  # t_max
            1.6,  # pluie
            0,  # neige
            0.5,  # soleil
            0.13338,  # demande_eau
            0.1338,  # demande_reservoir
            3.873,  # etat["neige_au_sol"]
            0,  # etat["fonte"],
            5.794,  # etat["neige_au_sol_totale"]
            0,  # etat["fonte_totale"],
            1.0,  # etat["derniere_neige"],
            self.etat["eeg"],
            0.0,  # etat["gel"],
        )
        self.assertIsInstance(eau_surface, float)
        self.assertIsInstance(demande_eau, float)
        self.assertIsInstance(etat, dict)
        self.assertIsInstance(etr, np.ndarray)
        self.assertEqual(etr.shape, (5,))
        self.assertIsInstance(apport_vertical, np.ndarray)
        self.assertEqual(apport_vertical.shape, (5,))

        # neige_au_sol + pluie_moins_evaporation < 0
        eau_surface, demande_eau, etat, etr, apport_vertical = dj_hsami(
            self.modules,
            self.meteo,
            self.etat,
            np.zeros(5),
            np.zeros(5),
            self.duree,
            0.5,  # self.param[1], Efficacité évapo hiver
            self.param[2],
            self.param[3],
            self.param[4],
            self.param[5],
            self.param[6],
            self.param[7],
            self.param[11],
            self.etat["sol"],
            self.t_min,
            self.t_max,
            0.1,  # self.pluie,
            self.neige,
            0.0,  # soleil
            3.7,  # demande_eau
            0.0,  # demande_reservoir
            self.etat["neige_au_sol"],
            self.etat["fonte"],
            3.69,  # etat["neige_au_sol_totale"]
            self.etat["fonte_totale"],
            self.etat["derniere_neige"],
            self.etat["eeg"],
            self.etat["gel"],
        )
        self.assertIsInstance(eau_surface, float)
        self.assertIsInstance(demande_eau, float)
        self.assertIsInstance(etat, dict)
        self.assertIsInstance(etr, np.ndarray)
        self.assertEqual(etr.shape, (5,))
        self.assertIsInstance(apport_vertical, np.ndarray)
        self.assertEqual(apport_vertical.shape, (5,))

        # Module dj
        # ---------
        self.meteo = {
            "bassin": [3.3, 15.5, 12.3, 0.0, 0.5, -1],
            "reservoir": [3.3, 15.5, 12.0, 0.0, 0.5, -1],
        }
        self.modules["een"] = "dj"
        eau_surface, demande_eau, etat, etr, apport_vertical = dj_hsami(
            self.modules,
            self.meteo,
            self.etat,
            np.zeros(5),
            np.zeros(5),
            self.duree,
            self.param[1],
            self.param[2],
            self.param[3],
            self.param[4],
            self.param[5],
            self.param[6],
            self.param[7],
            self.param[11],
            self.etat["sol"],
            self.meteo["bassin"][0],  # t_min,
            self.meteo["bassin"][1],  # t_max,
            self.meteo["bassin"][2],  # pluie,
            self.meteo["bassin"][3],  # neige,
            0.5,  # soleil
            0.0,  # demande_eau
            0.0,  # demande_reservoir
            self.etat["neige_au_sol"],
            self.etat["fonte"],
            2.72,  # etat["neige_au_sol_totale"],
            self.etat["fonte_totale"],
            self.etat["derniere_neige"],
            self.etat["eeg"],
            self.etat["gel"],
        )
        self.assertIsInstance(eau_surface, float)
        self.assertIsInstance(demande_eau, float)
        self.assertIsInstance(etat, dict)
        self.assertIsInstance(etr, np.ndarray)
        self.assertEqual(etr.shape, (5,))
        self.assertIsInstance(apport_vertical, np.ndarray)
        self.assertEqual(apport_vertical.shape, (5,))

        # potentiel_fonte < 0 : line 464
        self.meteo = {
            "bassin": [-9.3, -3.5, 2.3, 0.0, 0.5, -1],
            "reservoir": [-9.3, -3.5, 2.3, 0.0, 0.5, -1],
        }
        eau_surface, demande_eau, etat, etr, apport_vertical = dj_hsami(
            self.modules,
            self.meteo,
            self.etat,
            np.zeros(5),
            np.zeros(5),
            self.duree,
            self.param[1],
            self.param[2],
            self.param[3],
            self.param[4],
            self.param[5],
            self.param[6],
            self.param[7],
            self.param[11],
            self.etat["sol"],
            self.meteo["bassin"][0],  # t_min,
            self.meteo["bassin"][1],  # t_max,
            self.meteo["bassin"][2],  # pluie,
            self.meteo["bassin"][3],  # neige,
            0.5,  # soleil
            0.0,  # demande_eau
            0.0,  # demande_reservoir
            2.5,  # self.etat["neige_au_sol"],
            self.etat["fonte"],
            2.72,  # etat["neige_au_sol_totale"],
            self.etat["fonte_totale"],
            self.etat["derniere_neige"],
            self.etat["eeg"],
            self.etat["gel"],
        )
        self.assertIsInstance(eau_surface, float)
        self.assertIsInstance(demande_eau, float)
        self.assertIsInstance(etat, dict)
        self.assertIsInstance(etr, np.ndarray)
        self.assertEqual(etr.shape, (5,))
        self.assertIsInstance(apport_vertical, np.ndarray)
        self.assertEqual(apport_vertical.shape, (5,))

    def test_hsami_mdj_alt(self):
        # Module een: mdj
        self.modules["een"] = "mdj"
        # Module radiation : hsami
        # neige_au_sol == 0 : lines 1444 - 1568

        eau_surface, demande_eau, etat, etr, apport_vertical = mdj_alt(
            self.param,
            self.modules,
            self.meteo,
            self.physio,
            self.etat,
            np.zeros(5),
            np.zeros(5),
            self.duree,
            self.pdts,
            self.jj,
            self.pas_de_temps,
            self.param[1],
            self.param[4],
            self.param[11],
            self.etat["sol"],
            self.t_min,
            self.t_max,
            self.pluie,
            self.neige,
            0.0,  # soleil
            0.0,  # demande_eau
            0.0,  # demande_reservoir
            self.etat["neige_au_sol"],
            self.etat["fonte"],
            self.etat["derniere_neige"],
            self.etat["eeg"],
            self.etat["gel"],
        )
        self.assertIsInstance(eau_surface, float)
        self.assertIsInstance(demande_eau, float)
        self.assertIsInstance(etat, dict)
        self.assertIsInstance(etr, np.ndarray)
        self.assertEqual(etr.shape, (5,))
        self.assertIsInstance(apport_vertical, np.ndarray)
        self.assertEqual(apport_vertical.shape, (5,))

        # Module een : alt
        self.modules["een"] = "alt"
        # Module radiation : hsami

        eau_surface, demande_eau, etat, etr, apport_vertical = mdj_alt(
            self.param,
            self.modules,
            self.meteo,
            self.physio,
            self.etat,
            np.zeros(5),
            np.zeros(5),
            self.duree,
            self.pdts,
            self.jj,
            self.pas_de_temps,
            self.param[1],
            self.param[4],
            self.param[11],
            self.etat["sol"],
            self.t_min,
            self.t_max,
            self.pluie,
            self.neige,
            0.0,  # soleil
            0.0,  # demande_eau
            0.0,  # demande_reservoir
            self.etat["neige_au_sol"],
            self.etat["fonte"],
            self.etat["derniere_neige"],
            self.etat["eeg"],
            self.etat["gel"],
        )
        self.assertIsInstance(eau_surface, float)
        self.assertIsInstance(demande_eau, float)
        self.assertIsInstance(etat, dict)
        self.assertIsInstance(etr, np.ndarray)
        self.assertEqual(etr.shape, (5,))
        self.assertIsInstance(apport_vertical, np.ndarray)
        self.assertEqual(apport_vertical.shape, (5,))

        # Module een: mdj
        self.modules["een"] = "mdj"
        # Module radiation : mdj
        self.modules["radiation"] = "mdj"

        eau_surface, demande_eau, etat, etr, apport_vertical = mdj_alt(
            self.param,
            self.modules,
            self.meteo,
            self.physio,
            self.etat,
            np.zeros(5),
            np.zeros(5),
            self.duree,
            self.pdts,
            self.jj,
            self.pas_de_temps,
            self.param[1],
            self.param[4],
            self.param[11],
            self.etat["sol"],
            self.t_min,
            self.t_max,
            self.pluie,
            self.neige,
            0.0,  # soleil
            0.0,  # demande_eau
            0.0,  # demande_reservoir
            self.etat["neige_au_sol"],
            self.etat["fonte"],
            self.etat["derniere_neige"],
            self.etat["eeg"],
            self.etat["gel"],
        )
        self.assertIsInstance(eau_surface, float)
        self.assertIsInstance(demande_eau, float)
        self.assertIsInstance(etat, dict)
        self.assertIsInstance(etr, np.ndarray)
        self.assertEqual(etr.shape, (5,))
        self.assertIsInstance(apport_vertical, np.ndarray)
        self.assertEqual(apport_vertical.shape, (5,))

        # Module een : alt
        self.modules["een"] = "alt"
        # Module radiation : mdj
        self.modules["radiation"] = "mdj"

        eau_surface, demande_eau, etat, etr, apport_vertical = mdj_alt(
            self.param,
            self.modules,
            self.meteo,
            self.physio,
            self.etat,
            np.zeros(5),
            np.zeros(5),
            self.duree,
            self.pdts,
            self.jj,
            self.pas_de_temps,
            self.param[1],
            self.param[4],
            self.param[11],
            self.etat["sol"],
            self.t_min,
            self.t_max,
            self.pluie,
            self.neige,
            0.0,  # soleil
            0.0,  # demande_eau
            0.0,  # demande_reservoir
            self.etat["neige_au_sol"],
            self.etat["fonte"],
            self.etat["derniere_neige"],
            self.etat["eeg"],
            self.etat["gel"],
        )
        self.assertIsInstance(eau_surface, float)
        self.assertIsInstance(demande_eau, float)
        self.assertIsInstance(etat, dict)
        self.assertIsInstance(etr, np.ndarray)
        self.assertEqual(etr.shape, (5,))
        self.assertIsInstance(apport_vertical, np.ndarray)
        self.assertEqual(apport_vertical.shape, (5,))

        # neige_au_sol > 0 or neige > 0 : lines 983 - 1443
        self.modules["een"] = "mdj"
        etat["neige_au_sol"] = 2.8
        n_occupation = len(self.physio["occupation"])
        self.etat["mdj"]["neige_au_sol"] = n_occupation * [self.etat["neige_au_sol"]]
        self.etat["mdj"]["couvert_neige"] = n_occupation * [0.19]

        eau_surface, demande_eau, etat, etr, apport_vertical = mdj_alt(
            self.param,
            self.modules,
            self.meteo,
            self.physio,
            self.etat,
            np.zeros(5),
            np.zeros(5),
            self.duree,
            self.pdts,
            self.jj,
            self.pas_de_temps,
            self.param[1],
            self.param[4],
            self.param[11],
            self.etat["sol"],
            self.t_min,
            self.t_max,
            self.pluie,
            self.neige,
            0.0,  # soleil
            0.0,  # demande_eau
            0.0,  # demande_reservoir
            etat["neige_au_sol"],
            self.etat["fonte"],
            self.etat["derniere_neige"],
            self.etat["eeg"],
            self.etat["gel"],
        )
        self.assertIsInstance(eau_surface, float)
        self.assertIsInstance(demande_eau, float)
        self.assertIsInstance(etat, dict)
        self.assertIsInstance(etr, np.ndarray)
        self.assertEqual(etr.shape, (5,))
        self.assertIsInstance(apport_vertical, np.ndarray)
        self.assertEqual(apport_vertical.shape, (5,))

    def test_gel_sol(self):
        result = gel_sol(
            self.duree,
            self.dt_max,
            self.param[11],
            self.etat["sol"][0],
            self.etat["gel"],
            self.etat["neige_au_sol"],
        )

        self.assertIsNotNone(result)

    def test_degel_sol(self):
        # Sol partiellement dégelé
        result = degel_sol(
            self.duree,
            self.dt_max,
            self.etat["sol"][0],
            self.etat["gel"],
            self.etat["neige_au_sol"],
        )
        self.assertIsNotNone(result)

        # Sol complétement dégelé
        result = degel_sol(
            self.duree,
            6.4,
            0.53,
            0.31,
            0.1,
        )
        self.assertIsNotNone(result)

    def test_gel_neige(self):
        if self.etat["neige_au_sol"] > 0.0254:
            result = gel_neige(
                self.duree,
                self.dt_max,
                self.etat["neige_au_sol"],
                self.etat["fonte"],
                self.etat["fonte_totale"],
            )
        else:
            result = 0
        self.assertIsNotNone(result)

    def test_percolation_eau_fonte(self):
        result = percolation_eau_fonte(
            self.etat["neige_au_sol"],
            self.etat["neige_au_sol_totale"],
            self.etat["fonte"],
            self.etat["fonte_totale"],
        )
        self.assertIsNotNone(result)

    def test_conductivite_neige(self):
        result = conductivite_neige(300)
        self.assertIsNotNone(result)

    def test_erf(self):
        result = calcul_erf(1)
        self.assertIsNotNone(result)

    def test_indice_radiation(self):
        result = calcul_indice_radiation(
            self.jj,
            self.physio["latitude"],
            self.physio["i_orientation_bv"],
            24 / self.nb_pas,
            self.physio["pente_bv"],
        )
        self.assertIsNotNone(result)

    def test_albedo_een(self):
        tmoy = (self.t_max + self.t_min) / 2
        result = albedo_een(
            self.physio["albedo_sol"],
            calcul_densite_neige(tmoy) / 1000,
            self.etat["neige_au_sol"],
            self.neige,
            24 / self.nb_pas,
            self.pluie,
            tmoy,
        )
        self.assertIsNotNone(result)

        # st_neige > 0 : line 2092
        self.meteo = {
            "bassin": [-3.3, 1.5, 1.3, 3.0, 0.5, -1],
            "reservoir": [-3.3, 1.5, 1.0, 3.0, 0.5, -1],
        }
        self.physio["albedo_sol"] = 0.45
        self.etat["neige_au_sol"] = 12.8

        tmoy = (self.t_max + self.t_min) / 2
        result = albedo_een(
            self.physio["albedo_sol"],
            calcul_densite_neige(tmoy) / 1000,
            self.etat["neige_au_sol"],
            self.neige,
            24 / self.nb_pas,
            self.pluie,
            tmoy,
        )
        self.assertIsNotNone(result)

    def test_calcul_densite_neige(self):
        # temperature < -17 0 : line 3130
        self.t_max = -18.5
        self.t_min = -25.4
        result = calcul_densite_neige((self.t_max + self.t_min) / 2)

        self.assertIsNotNone(result)

        # temperature > 0 : line 3132
        self.t_max = 18.5
        self.t_min = 25.4
        result = calcul_densite_neige((self.t_max + self.t_min) / 2)
        self.assertIsNotNone(result)

        # temperature > 0  and temperature < -17: line 3134
        self.t_max = 8.5
        self.t_min = 5.4
        result = calcul_densite_neige((self.t_max + self.t_min) / 2)
        self.assertIsNotNone(result)

    def test_pluie_neige(self):
        # isinstance(prec, float) : line 2176
        self.meteo = {
            "bassin": [3.3, 15.5, 12.3, 0.0, 0.5, -1],
            "reservoir": [3.3, 15.5, 12.0, 0.0, 0.5, -1],
        }
        result = pluie_neige(self.t_max, self.t_min, self.pluie / 100 + self.neige / 100)
        self.assertIsNotNone(result)

        # isinstance(prec, list) | isinstance(prec, np.ndarray): line 2185
        self.t_max = [3.8, 1.8, 2.2, 6.1, 0.0, -2.7, 3.8, 2.7, 0.5, 6.1, 2.2]
        self.t_min = [
            -7.2,
            -4.9,
            -1.1,
            -0.5,
            -0.5,
            -17.2,
            -14.4,
            -11.0,
            -14.9,
            -12.7,
            -7.7,
        ]
        self.pluie = [0.1, 0.2, 0.5, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]
        self.neige = [0.4, 0.2, 0.2, 0.0, 0.0, 0.0, 0.1, 0.3, 0.0, 0.0, 0.5]

        result = pluie_neige(
            np.array(self.t_max),
            np.array(self.t_min),
            np.array(self.pluie) / 100.0 + np.array(self.neige) / 100.0,
        )
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
