import sys
import unittest

import numpy as np

sys.path.append("../src/hsamiplus")
from hsami_glace import hsami_glace


class TestHsamiGlace(unittest.TestCase):
    def setUp(self):
        self.modules = {
            "reservoir": 1,
            "glace_reservoir": "stefan",  # 'my_lake',
            "een": "mdj",
        }
        self.superficie = [2640.0, 438.0]
        self.etats = {
            "reservoir_epaisseur_glace": 0,
            "reservoir_superficie_glace": 0,
            "reservoir_superficie_ref": 438.0,
            "reservoir_superficie": 438.0,
            "ratio_reservoir": 0,
            "ratio_bassin": 1,
            "ratio_fixe": 1,
            "eeg": np.zeros(5000),
            "neige_au_sol": 4.50,
            "dernier_gel": 0,
            "cumdegGel": -530.2250,
            "obj_gel": -200.0,
            "hsami": {
                "couvert_neige": [0.1, 0.1, 0.1],
                "densite_neige": [0.02, 0.02, 0.02],
            },
            "mdj": {
                "couvert_neige": [0.1, 0.1, 0.1],
                "densite_neige": [0.02, 0.02, 0.02],
            },
        }
        self.meteo = {
            "bassin": [-15.30, -1.90, 0.0, 0.0, 0.5, -1.0],
            "reservoir": [-15.30, -1.90, 0.0, 0.0, 0.5, -1.0],
        }

        self.physio = {
            "niveau": 358.940,
            "coeff": [-0.0119, 52.095, -16814],
            "occupation_bande": [0.083, 0.503, 0.414],
        }

        self.param = [0] * 50  # Assuming 50 parameters for simplicity
        self.param[47] = 0.10  # Coefficient pour calcul du volume max du MHE (hmax)

    def test_hsami_glace_no_reservoir(self):
        self.modules["reservoir"] = 0
        glace_vers_reservoir, bassin_vers_reservoir, etats = hsami_glace(self.modules, self.superficie, self.etats)
        self.assertEqual(glace_vers_reservoir, 0)
        self.assertEqual(bassin_vers_reservoir, 0)
        self.assertEqual(etats["reservoir_epaisseur_glace"], 0)
        self.assertEqual(etats["reservoir_superficie_glace"], 0)
        self.assertEqual(etats["ratio_reservoir"], 0)
        self.assertEqual(etats["ratio_bassin"], 1)
        self.assertEqual(etats["ratio_fixe"], 1)

    def test_hsami_glace_with_reservoir_stefan(self):
        self.modules["glace_reservoir"] = "stefan"

        glace_vers_reservoir, bassin_vers_reservoir, etats = hsami_glace(
            self.modules,
            self.superficie,
            self.etats,
            self.meteo,
            self.physio,
            self.param,
        )
        self.assertIsInstance(glace_vers_reservoir, float)
        self.assertIsInstance(bassin_vers_reservoir, float)
        self.assertIsInstance(etats, dict)

    def test_hsami_glace_with_reservoir_mylake(self):
        self.modules["glace_reservoir"] = "my_lake"

        glace_vers_reservoir, bassin_vers_reservoir, etats = hsami_glace(
            self.modules,
            self.superficie,
            self.etats,
            self.meteo,
            self.physio,
            self.param,
        )
        self.assertIsInstance(glace_vers_reservoir, float)
        self.assertIsInstance(bassin_vers_reservoir, float)
        self.assertIsInstance(etats, dict)


if __name__ == "__main__":
    unittest.main()
