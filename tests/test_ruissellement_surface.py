import sys
import unittest

import numpy as np

sys.path.append("../src/hsamiplus")

from hsami_ruissellement_surface import hsami_ruissellement_surface


class TestHsamiRuissellementSurface(unittest.TestCase):
    def setUp(self):
        self.nb_pas = 1
        self.param = [0] * 50
        self.param[8] = 0.1  # effet_gel
        self.param[9] = 10  # effet_sol
        self.param[10] = 0.5  # seuil_min
        self.param[12] = 10  # sol_max for 'hsami'
        self.param[39] = 10  # layer thickness for '3couches'
        self.param[44] = 0.2  # total porosity for '3couches'
        self.etat = {"gel": 0, "sol": [5.0, np.nan]}
        self.eau_surface = 12.80
        self.modules = {"infiltration": "hsami", "sol": "hsami"}

    def test_hsami_ruissellement_surface_hsami(self):
        self.modules["infiltration"] = "hsami"
        ruissellement_surface, infiltration = hsami_ruissellement_surface(
            self.nb_pas, self.param, self.etat, self.eau_surface, self.modules
        )

        self.assertIsInstance(ruissellement_surface, float)
        self.assertIsInstance(infiltration, float)

    def test_hsami_ruissellement_surface_green_ampt(self):
        self.modules["infiltration"] = "green_ampt"
        ruissellement_surface, infiltration = hsami_ruissellement_surface(
            self.nb_pas, self.param, self.etat, self.eau_surface, self.modules
        )

        self.assertIsInstance(ruissellement_surface, float)
        self.assertIsInstance(infiltration, float)

    def test_hsami_ruissellement_surface_scs_cn(self):
        self.modules["infiltration"] = "scs_cn"

        ruissellement_surface, infiltration = hsami_ruissellement_surface(
            self.nb_pas, self.param, self.etat, self.eau_surface, self.modules
        )

        self.assertIsInstance(ruissellement_surface, float)
        self.assertIsInstance(infiltration, float)

    def test_hsami_ruissellement_surface_hsami_no_gel(self):
        self.etat["gel"] = 0
        ruissellement_surface, infiltration = hsami_ruissellement_surface(
            self.nb_pas, self.param, self.etat, self.eau_surface, self.modules
        )

        self.assertIsInstance(ruissellement_surface, float)
        self.assertIsInstance(infiltration, float)

    def test_hsami_ruissellement_surface_hsami_with_gel(self):
        self.etat["gel"] = 1
        ruissellement_surface, infiltration = hsami_ruissellement_surface(
            self.nb_pas, self.param, self.etat, self.eau_surface, self.modules
        )

        self.assertIsInstance(ruissellement_surface, float)
        self.assertIsInstance(infiltration, float)

    def test_hsami_ruissellement_surface_hsami_3couches(self):
        self.modules["sol"] = "3couches"
        ruissellement_surface, infiltration = hsami_ruissellement_surface(
            self.nb_pas, self.param, self.etat, self.eau_surface, self.modules
        )

        self.assertIsInstance(ruissellement_surface, float)
        self.assertIsInstance(infiltration, float)


if __name__ == "__main__":
    unittest.main()
