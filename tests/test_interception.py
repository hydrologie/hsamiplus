import sys
import unittest

import numpy as np

sys.path.append("../src/hsamiplus")

from hsami_interception import (
    albedo_een,
    calcul_densite_neige,
    conductivite_neige,
    degel_sol,
    erf,
    gel_neige,
    gel_sol,
    hsami_interception,
    indice_radiation,
    percolation_eau_fonte,
    pluie_neige,
)


class TestHsamiInterception(unittest.TestCase):
    def setUp(self):
        self.nb_pas = 1
        self.jj = 245
        lat = 47.1943
        alt = 390.9
        albedo = 0.7
        albedo_neige = 0.5
        self.param = [0] * 50  # Assuming 50 parameters for simplicity
        self.param[0] = 0.5  # efficacite_evapo_ete
        self.param[1] = 0.0  # efficacite_evapo_hiver
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
        result = degel_sol(
            self.duree,
            self.dt_max,
            self.etat["sol"][0],
            self.etat["gel"],
            self.etat["neige_au_sol"],
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
        result = erf(1)
        self.assertIsNotNone(result)

    def test_indice_radiation(self):
        result = indice_radiation(
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

    def test_calcul_densite_neige(self):
        result = calcul_densite_neige((self.t_max + self.t_min) / 2)
        self.assertIsNotNone(result)

    def test_pluie_neige(self):
        result = pluie_neige(
            self.t_max, self.t_min, self.pluie / 100 + self.neige / 100
        )
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
