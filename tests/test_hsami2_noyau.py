import unittest
from unittest.mock import patch

import numpy as np

from hsamiplus.hsami2_noyau import etp_glace_interception, hsami2_noyau


class TestHsami2Noyau(unittest.TestCase):
    def setUp(self):
        neige_au_sol = 0.0
        fonte = 0.0
        gel = 0.0

        self.projet = {
            "nb_pas_par_jour": 1,
            "pas": 1,
            "date": [1950, 5, 8, 0, 0, 0],
            "superficie": [2640, 438],
            "memoire": 10,
            "param": [
                0.5,
                0.0,
                0.10,
                0.05,
                -4,
                -4,
                -2,
                1.10,
                1,
                5,
                1,
                0.05,
                10,
                8,
                0.25,
                0.20,
                0.01,
                0.0,
                0.40,
                0.50,
                0.70,
                1,
                0.30,
                25,
                -3,
                1,
                0.6,
                0.01,
                0.1,
                0.01,
                0.01,
                -4,
                -2,
                -2,
                0,
                0.02,
                4,
                4,
                -3,
                5,
                10,
                0.05,
                0.1,
                0.1,
                0.1,
                0.1,
                0.7,
                1,
                0.1,
                -2,
            ],
            "meteo": {
                "bassin": [-4.4, 12.2, 0.1, 0.0, 0.5, -1.0],
                "reservoir": [-4.4, 12.2, 0.1, 0.0, 0.5, -1.0],
            },
            "modules": {
                "etp_bassin": "hsami",
                "etp_reservoir": "hsami",
                "een": "hsami",
                "infiltration": "hsami",
                "sol": "hsami",
                "qbase": "hsami",
                "radiation": "hsami",
                "mhumide": 0,
                "reservoir": 0,
                "glace_reservoir": "stefan",
            },
            "physio": {
                "latitude": 47.1943,
                "altitude": 390.90,
                "albedo_sol": 0.7,
                "i_orientation_bv": 1,
                "pente_bv": 1.8,
                "occupation": [0.083, 0.503, 0.4140],
                "niveau": 359.17,
                "coeff": [-0.0119, 52.095, -16814],
                "samax": 242.970,
                "occupation_bande": [0.003, 0.015, 0.043, 0.194, 0.745],
                "altitude_bande": [581, 530, 479, 429, 379],
            },
        }
        n_occupation = len(self.projet["physio"]["occupation"])
        n_occupation_bande = len(self.projet["physio"]["occupation_bande"])

        self.etat = {
            "eau_hydrogrammes": np.array(
                [
                    [0.0119659499257712, 0.0, 0.000836568732966587],
                    [0.00657326699679702, 0.0, 0.000457820350587645],
                    [0.00350988766532956, 0.0, 0.000243562528101483],
                    [0.00183560098889314, 0.0, 0.000126823084926892],
                    [0.000940041395578869, 0.0, 6.45595278510044e-05],
                    [0.000463627783143457, 0.00863160871921057, 3.18395122165191e-05],
                    [0.000219150439159671, 0.0, 1.48817782714734e-05],
                    [9.36765158139741e-05, 0.0, 6.26085573641646e-06],
                    [3.06877496825505e-05, 0.0, 2.01133487381504e-06],
                    [0.0, 0.0, 0.0],
                ]
            ),
            "neige_au_sol": neige_au_sol,
            "fonte": fonte,
            "nas_tot": 0,
            "fonte_tot": 0,
            "derniere_neige": 0,
            "gel": gel,
            "sol": [8.1390, 5.682],
            "nappe": 8.0817,
            "reserve": 0.0012,
            "mdj": {
                "couvert_neige": n_occupation * [0.0],
                "densite_neige": n_occupation * [0.0],
                "albedo_neige": n_occupation * [0.5],
                "neige_au_sol": n_occupation * [neige_au_sol],
                "fonte": n_occupation * [fonte],
                "gel": n_occupation * [gel],
                "sol": n_occupation * [0.0],
                "energie_neige": n_occupation * [0.0],
                "energie_glace": n_occupation * [0.0],
            },
            "alt": {
                "couvert_neige": n_occupation_bande * [0.0],
                "densite_neige": n_occupation_bande * [0.0],
                "albedo_neige": n_occupation_bande * [0.5],
                "neige_au_sol": n_occupation_bande * [neige_au_sol],
                "fonte": n_occupation_bande * [fonte],
                "gel": n_occupation_bande * [gel],
                "sol": n_occupation_bande * [0.0],
                "energie_neige": n_occupation_bande * [0.0],
                "energie_glace": n_occupation_bande * [0.0],
            },
            "mh_vol": 24565661.441,
            "ratio_MH": 0.0093,
            "mh_surf": 2456.566,
            "mhumide": 0.9305,
            "ratio_qbase": 0,
            "cumdegGel": 0,
            "obj_gel": -200,
            "dernier_gel": 0,
            "reservoir_epaisseur_glace": 0,
            "reservoir_energie_glace": 0,
            "reservoir_superficie": 438,
            "reservoir_superficie_glace": 0,
            "reservoir_superficie_ref": 438,
            "eeg": np.zeros(5000),
            "ratio_bassin": 1,
            "ratio_reservoir": 0,
            "ratio_fixe": 1,
        }

    def test_hsami2_noyau(self):
        s, etat, delta = hsami2_noyau(self.projet, self.etat)
        self.assertIn("Qtotal", s)
        self.assertIn("Qbase", s)
        self.assertIn("Qinter", s)
        self.assertIn("Qsurf", s)
        self.assertIn("Qreservoir", s)
        self.assertIn("Qglace", s)
        self.assertIn("ETP", s)
        self.assertIn("ETRtotal", s)
        self.assertIn("ETRsublim", s)
        self.assertIn("ETRPsurN", s)
        self.assertIn("ETRintercept", s)
        self.assertIn("ETRtranspir", s)
        self.assertIn("ETRreservoir", s)
        self.assertIn("ETRmhumide", s)
        self.assertIn("total", delta)
        self.assertIn("glace", delta)
        self.assertIn("interception", delta)
        self.assertIn("ruissellement", delta)
        self.assertIn("vertical", delta)
        self.assertIn("horizontal", delta)

    @patch("warnings.warn")
    def test_occupation_warning(self, mock_warn):
        self.projet["modules"]["een"] = "mdj"
        self.projet["physio"]["occupation"] = [0.50, 0.41]

        _s, _etat, _delta = hsami2_noyau(self.projet, self.etat)

        # Check if the warning was issued
        mock_warn.assert_called_once_with("La somme des occupations nest pas égale à 1", stacklevel=2)

    @patch("warnings.warn")
    def test_occupation_bande_warning(self, mock_warn):
        self.projet["modules"]["een"] = "alt"
        self.projet["physio"]["occupation_bande"] = [0.043, 0.194, 0.745]

        _s, _etat, _delta = hsami2_noyau(self.projet, self.etat)

        # Check if the warning was issued
        mock_warn.assert_called_once_with("La somme des occupations nest pas égale à 1", stacklevel=2)

    def test_hsami2_noyau_module1(self):
        self.projet["etp_bassin"] = "mcguinness_bordne"
        self.projet["etp_reservoir"] = "mcguinness_bordne"
        self.projet["een"] = "mdj"
        self.projet["infiltration"] = "green_ampt"
        self.projet["sol"] = "3couches"
        self.projet["qbase"] = "dingman"
        self.projet["radiation"] = "mdj"
        self.projet["mhumide"] = 1
        self.projet["reservoir"] = 1
        self.projet["glace_reservoir"] = "stefan"

        s, etat, delta = hsami2_noyau(self.projet, self.etat)
        self.assertIn("Qtotal", s)
        self.assertIn("Qbase", s)
        self.assertIn("Qinter", s)
        self.assertIn("Qsurf", s)
        self.assertIn("Qreservoir", s)
        self.assertIn("Qglace", s)
        self.assertIn("ETP", s)
        self.assertIn("ETRtotal", s)
        self.assertIn("ETRsublim", s)
        self.assertIn("ETRPsurN", s)
        self.assertIn("ETRintercept", s)
        self.assertIn("ETRtranspir", s)
        self.assertIn("ETRreservoir", s)
        self.assertIn("ETRmhumide", s)
        self.assertIn("total", delta)
        self.assertIn("glace", delta)
        self.assertIn("interception", delta)
        self.assertIn("ruissellement", delta)
        self.assertIn("vertical", delta)
        self.assertIn("horizontal", delta)

    def test_hsami2_noyau_module2(self):
        self.projet["etp_bassin"] = "priestley_taylor"
        self.projet["etp_reservoir"] = "hargreaves"
        self.projet["een"] = "alt"
        self.projet["infiltration"] = "scs_cn"
        self.projet["sol"] = "3couches"
        self.projet["qbase"] = "dingman"
        self.projet["radiation"] = "mdj"
        self.projet["mhumide"] = 1
        self.projet["reservoir"] = 1
        self.projet["glace_reservoir"] = "mylake"

        s, etat, delta = hsami2_noyau(self.projet, self.etat)
        self.assertIn("Qtotal", s)
        self.assertIn("Qbase", s)
        self.assertIn("Qinter", s)
        self.assertIn("Qsurf", s)
        self.assertIn("Qreservoir", s)
        self.assertIn("Qglace", s)
        self.assertIn("ETP", s)
        self.assertIn("ETRtotal", s)
        self.assertIn("ETRsublim", s)
        self.assertIn("ETRPsurN", s)
        self.assertIn("ETRintercept", s)
        self.assertIn("ETRtranspir", s)
        self.assertIn("ETRreservoir", s)
        self.assertIn("ETRmhumide", s)
        self.assertIn("total", delta)
        self.assertIn("glace", delta)
        self.assertIn("interception", delta)
        self.assertIn("ruissellement", delta)
        self.assertIn("vertical", delta)
        self.assertIn("horizontal", delta)

        class TestEtpGlaceInterception:
            def setup_method(self):
                self.projet = {
                    "date": [1950, 5, 8, 0, 0, 0],
                    "pas": 1,
                    "nb_pas_par_jour": 1,
                    "superficie": [2640, 438],
                    "param": [
                        0.5,
                        0.0,
                        0.10,
                        0.05,
                        -4,
                        -4,
                        -2,
                        1.10,
                        1,
                        5,
                        1,
                        0.05,
                        10,
                        8,
                        0.25,
                        0.20,
                        0.01,
                        0.0,
                        0.40,
                        0.50,
                        0.70,
                        1,
                        0.30,
                        25,
                        -3,
                        1,
                        0.6,
                        0.01,
                        0.1,
                        0.01,
                        0.01,
                        -4,
                        -2,
                        -2,
                        0,
                        0.02,
                        4,
                        4,
                        -3,
                        5,
                        10,
                        0.05,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.7,
                        1,
                        0.1,
                        -2,
                    ],
                    "meteo": {
                        "bassin": [-4.4, 12.2, 0.1, 0.0, 0.5, -1.0],
                        "reservoir": [-4.4, 12.2, 0.1, 0.0, 0.5, -1.0],
                    },
                    "modules": {
                        "etp_bassin": "hsami",
                        "etp_reservoir": "hsami",
                        "een": "hsami",
                        "infiltration": "hsami",
                        "sol": "hsami",
                        "qbase": "hsami",
                        "radiation": "hsami",
                        "mhumide": 0,
                        "reservoir": 0,
                        "glace_reservoir": 0,
                    },
                    "physio": {
                        "latitude": 47.1943,
                        "altitude": 390.90,
                        "albedo_sol": 0.7,
                        "i_orientation_bv": 1,
                        "pente_bv": 1.8,
                        "occupation": [0.083, 0.503, 0.4140],
                        "niveau": 359.17,
                        "coeff": [-0.0119, 52.095, -16814],
                        "samax": 242.970,
                        "occupation_bande": [0.003, 0.015, 0.043, 0.194, 0.745],
                        "altitude_bande": [581, 530, 479, 429, 379],
                    },
                }
                self.etat = {
                    "eau_hydrogrammes": np.zeros((10, 3)),
                    "neige_au_sol": 0.0,
                    "fonte": 0.0,
                    "nas_tot": 0,
                    "fonte_tot": 0,
                    "derniere_neige": 0,
                    "gel": 0.0,
                    "sol": [8.1390, 5.682],
                    "nappe": 8.0817,
                    "reserve": 0.0012,
                    "mdj": {
                        "couvert_neige": [0.0, 0.0, 0.0],
                        "densite_neige": [0.0, 0.0, 0.0],
                        "albedo_neige": [0.5, 0.5, 0.5],
                        "neige_au_sol": [0.0, 0.0, 0.0],
                        "fonte": [0.0, 0.0, 0.0],
                        "gel": [0.0, 0.0, 0.0],
                        "sol": [0.0, 0.0, 0.0],
                        "energie_neige": [0.0, 0.0, 0.0],
                        "energie_glace": [0.0, 0.0, 0.0],
                    },
                    "alt": {
                        "couvert_neige": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "densite_neige": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "albedo_neige": [0.5, 0.5, 0.5, 0.5, 0.5],
                        "neige_au_sol": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "fonte": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "gel": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "sol": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "energie_neige": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "energie_glace": [0.0, 0.0, 0.0, 0.0, 0.0],
                    },
                    "mh_vol": 24565661.441,
                    "ratio_MH": 0.0093,
                    "mh_surf": 2456.566,
                    "mhumide": 0.9305,
                    "ratio_qbase": 0,
                    "cumdegGel": 0,
                    "obj_gel": -200,
                    "dernier_gel": 0,
                    "reservoir_epaisseur_glace": 0,
                    "reservoir_energie_glace": 0,
                    "reservoir_superficie": 438,
                    "reservoir_superficie_glace": 0,
                    "reservoir_superficie_ref": 438,
                    "eeg": np.zeros(5000),
                    "ratio_bassin": 1,
                    "ratio_reservoir": 0,
                    "ratio_fixe": 1,
                }
                self.bilan = {}

            def test_etp_glace_interception_default(self):
                (
                    etat,
                    eau_surface,
                    demande_eau,
                    etps,
                    etr,
                    apport_vertical,
                    glace_vers_reservoir,
                    bassin_vers_reservoir,
                    bilan,
                ) = etp_glace_interception(
                    self.projet,
                    self.projet["param"],
                    self.projet["modules"],
                    self.projet["physio"],
                    self.projet["superficie"],
                    self.projet["meteo"],
                    self.projet["nb_pas_par_jour"],
                    self.etat,
                    self.bilan,
                )
                assert isinstance(etat, dict)
                assert isinstance(eau_surface, float)
                assert isinstance(demande_eau, float)
                assert isinstance(etps, list)
                assert isinstance(etr, list)
                assert isinstance(apport_vertical, list)
                assert isinstance(glace_vers_reservoir, float)
                assert isinstance(bassin_vers_reservoir, float)
                assert isinstance(bilan, dict)

            def test_etp_glace_interception_with_glace_reservoir(self):
                self.projet["modules"]["glace_reservoir"] = "stefan"
                (
                    etat,
                    eau_surface,
                    demande_eau,
                    etps,
                    etr,
                    apport_vertical,
                    glace_vers_reservoir,
                    bassin_vers_reservoir,
                    bilan,
                ) = etp_glace_interception(
                    self.projet,
                    self.projet["param"],
                    self.projet["modules"],
                    self.projet["physio"],
                    self.projet["superficie"],
                    self.projet["meteo"],
                    self.projet["nb_pas_par_jour"],
                    self.etat,
                    self.bilan,
                )
                assert isinstance(etat, dict)
                assert isinstance(eau_surface, float)
                assert isinstance(demande_eau, float)
                assert isinstance(etps, list)
                assert isinstance(etr, list)
                assert isinstance(apport_vertical, list)
                assert isinstance(glace_vers_reservoir, float)
                assert isinstance(bassin_vers_reservoir, float)
                assert isinstance(bilan, dict)

                # "niveau" not in physio
                self.projet["physio"] = {
                    "latitude": 47.1943,
                    "altitude": 390.90,
                    "albedo_sol": 0.7,
                    "i_orientation_bv": 1,
                    "pente_bv": 1.8,
                    "occupation": [0.083, 0.503, 0.4140],
                    # "niveau": 359.17,
                    "coeff": [-0.0119, 52.095, -16814],
                    "samax": 242.970,
                    "occupation_bande": [0.003, 0.015, 0.043, 0.194, 0.745],
                    "altitude_bande": [581, 530, 479, 429, 379],
                }
                (
                    etat,
                    eau_surface,
                    demande_eau,
                    etps,
                    etr,
                    apport_vertical,
                    glace_vers_reservoir,
                    bassin_vers_reservoir,
                    bilan,
                ) = etp_glace_interception(
                    self.projet,
                    self.projet["param"],
                    self.projet["modules"],
                    self.projet["physio"],
                    self.projet["superficie"],
                    self.projet["meteo"],
                    self.projet["nb_pas_par_jour"],
                    self.etat,
                    self.bilan,
                )
                assert isinstance(etat, dict)
                assert isinstance(etps, list)
                assert isinstance(etr, list)

                # "niveau" = ''
                self.projet["physio"] = {
                    "latitude": 47.1943,
                    "altitude": 390.90,
                    "albedo_sol": 0.7,
                    "i_orientation_bv": 1,
                    "pente_bv": 1.8,
                    "occupation": [0.083, 0.503, 0.4140],
                    "niveau": "",
                    "coeff": [-0.0119, 52.095, -16814],
                    "samax": 242.970,
                    "occupation_bande": [0.003, 0.015, 0.043, 0.194, 0.745],
                    "altitude_bande": [581, 530, 479, 429, 379],
                }
                (
                    etat,
                    eau_surface,
                    demande_eau,
                    etps,
                    etr,
                    apport_vertical,
                    glace_vers_reservoir,
                    bassin_vers_reservoir,
                    bilan,
                ) = etp_glace_interception(
                    self.projet,
                    self.projet["param"],
                    self.projet["modules"],
                    self.projet["physio"],
                    self.projet["superficie"],
                    self.projet["meteo"],
                    self.projet["nb_pas_par_jour"],
                    self.etat,
                    self.bilan,
                )
                assert isinstance(etat, dict)
                assert isinstance(etps, list)
                assert isinstance(etr, list)


if __name__ == "__main__":
    unittest.main()
