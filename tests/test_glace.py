import sys
import unittest

import numpy as np

from hsamiplus.hsami_glace import conductivite_neige, hsami_glace


class TestHsamiGlace(unittest.TestCase):
    def setUp(self):
        self.modules = {
            "reservoir": 1,
            "glace_reservoir": "stefan",  # 'my_lake',
            "een": "mdj",
        }
        self.superficie = [2640.0, 438.0]
        self.etats = {
            "reservoir_epaisseur_glace": 0.0,
            "reservoir_superficie_glace": 0.0,
            "reservoir_superficie_ref": 438.0,
            "reservoir_superficie": 438.0,
            "ratio_reservoir": 0.0,
            "ratio_bassin": 1.0,
            "ratio_fixe": 1.0,
            "eeg": np.zeros(5000),
            "neige_au_sol": 4.50,
            "dernier_gel": 0.0,
            "cumdeggel": -530.2250,
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
        glace_vers_reservoir, bassin_vers_reservoir, etats = hsami_glace(
            self.modules, self.superficie, self.etats
        )
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
        self.assertEqual(glace_vers_reservoir, 0)
        self.assertIsInstance(bassin_vers_reservoir, float)
        self.assertIsInstance(etats, dict)

        # superficie_fixe
        glace_vers_reservoir, bassin_vers_reservoir, etats = hsami_glace(
            self.modules,
            self.superficie,
            self.etats,
        )
        self.assertEqual(glace_vers_reservoir, 0)
        self.assertIsInstance(bassin_vers_reservoir, float)
        self.assertIsInstance(etats, dict)

        # delta_glace < 0, il y a restitution, line 124
        self.etats["reservoir_superficie_glace"] = 450.0
        glace_vers_reservoir, bassin_vers_reservoir, etats = hsami_glace(
            self.modules,
            self.superficie,
            self.etats,
            self.meteo,
            self.physio,
            self.param,
        )
        self.assertEqual(glace_vers_reservoir, 0)
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

        # superficie_fixe
        glace_vers_reservoir, bassin_vers_reservoir, etats = hsami_glace(
            self.modules,
            self.superficie,
            self.etats,
        )
        self.assertEqual(glace_vers_reservoir, 0)
        self.assertIsInstance(bassin_vers_reservoir, float)
        self.assertIsInstance(etats, dict)

        # t_a <= 0, et epaisseur_glace > 0 : line 322
        self.etats["reservoir_epaisseur_glace"] = 5.2

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

        # t_a > 0, il fait "chaud" et epaisseur_glace > 0 : line 351
        self.meteo = {
            "bassin": [0.30, 0.90, 2.0, 0.0, 0.5, -1.0],
            "reservoir": [0.30, 0.90, 2.0, 0.0, 0.5, -1.0],
        }
        self.etats["reservoir_epaisseur_glace"] = 5.2

        glace_vers_reservoir, bassin_vers_reservoir, etats = hsami_glace(
            self.modules,
            self.superficie,
            self.etats,
            self.meteo,
            self.physio,
            self.param,
        )
        self.assertEqual(glace_vers_reservoir, 0)
        self.assertEqual(bassin_vers_reservoir, 0)
        self.assertIsInstance(etats, dict)

        self.modules["een"] = "unknown"
        with self.assertRaises(ValueError) as context:
            hsami_glace(
                self.modules,
                self.superficie,
                self.etats,
                self.meteo,
                self.physio,
                self.param,
            )
        self.assertTrue(r"Le module 'my_lake' pour la glace" in str(context.exception))

    def test_hsami_glace_raises_error(self):
        self.modules["glace_reservoir"] = "unknown"

        with self.assertRaises(ValueError) as context:
            hsami_glace(
                self.modules,
                self.superficie,
                self.etats,
                self.meteo,
                self.physio,
                self.param,
            )
        self.assertTrue(
            "modules.glace_reservoir doit être 'stefan' ou 'my_lake'"
            in str(context.exception)
        )

    def test_hsami_glace_conductivite_neige(self):
        # Test cases with known inputs and expected outputs
        test_cases = [
            (100, 0.36969 - 0.36435 + 0.04735 + 0.01135 - 0.00848),
            (200, 0.36969 - 0.20566 - 0.04035 + 0.02491 - 0.00231),
            (300, 0.36969 - 0.04697 - 0.06757 + 0.00618 + 0.0048),
            (400, 0.36969 + 0.11172 - 0.03429 - 0.01366 - 0.00355),
            (500, 0.36969 + 0.27040 + 0.05948 - 0.00342 - 0.00607),
        ]

        for densite, expected in test_cases:
            with self.subTest(densite=densite):
                result = conductivite_neige(densite)
                self.assertAlmostEqual(result, expected, places=5)


if __name__ == "__main__":
    unittest.main()
