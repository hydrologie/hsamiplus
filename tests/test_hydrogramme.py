import sys
import unittest

import numpy as np

from hsamiplus.hsami_hydrogramme import hsami_hydrogramme


class TestHsamiHydrogramme(unittest.TestCase):
    def setUp(self):
        self.nb_pas = 1
        self.param = [0] * 50  # Assuming 50 parameters for simplicity
        self.param[19] = 0.5  # Mode hydrogramme surface
        self.param[20] = 0.7  # Forme hydrogramme surface
        self.param[21] = 1.0  # Mode hydrogramme intermédiaire
        self.param[22] = 0.3  # Forme hydrogramme intermédiaire
        self.memoire = 10

    def test_hsami_hydrogramme_basic(self):
        expected_shape = (2, self.memoire)

        result_1 = hsami_hydrogramme(
            self.param[19], self.param[20], self.nb_pas, self.memoire / self.nb_pas
        )
        result_2 = hsami_hydrogramme(
            self.param[21], self.param[22], self.nb_pas, self.memoire / self.nb_pas
        )
        hydrogrammes = np.vstack((result_1, result_2))

        self.assertEqual(hydrogrammes.shape, expected_shape)

    def test_hsami_hydrogramme_different_shapes(self):
        mode = [self.param[19], self.param[19] * 0.7]
        forme = self.param[20]
        pas_temps_par_jour = self.nb_pas
        memoire = 13
        expected_shape = (2, 13)

        result = hsami_hydrogramme(mode, forme, pas_temps_par_jour, memoire)

        self.assertEqual(result.shape, expected_shape)
        self.assertAlmostEqual(np.sum(result), 2.0, places=5)

    def test_hsami_hydrogramme_zero_memory(self):
        mode = 0.6
        forme = 0.5
        pas_temps_par_jour = 1
        memoire = 0
        expected_shape = (1, 0)

        result = hsami_hydrogramme(mode, forme, pas_temps_par_jour, memoire)

        self.assertEqual(result.shape, expected_shape)
        self.assertEqual(np.sum(result), 0.0)


if __name__ == "__main__":
    unittest.main()
