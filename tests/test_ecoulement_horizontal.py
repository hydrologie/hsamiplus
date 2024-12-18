import sys
import unittest

import numpy as np

from hsamiplus import hsami_ecoulement_horizontal


class TestHsamiEcoulementHorizontal(unittest.TestCase):
    def setUp(self):
        self.nb_pas = 1
        self.vidange_reserve_inter = 0.4
        self.reserve_inter = 0.0347
        self.eau_hydrogrammes = np.array(
            [
                [0.0129814932962203, 0.00, 3.79175437638145e-05],
                [0.00675656652285787, 0.00, 2.02081628919923e-05],
                [0.00338291591605612, 0.00, 1.06398364130294e-05],
                [0.00174271295401074, 0.00, 5.55443453642384e-06],
                [4.77753129261467e-05, 0.00, 2.62755611203238e-06],
                [1.56447950380494e-05, 0.0347508631297638, 1.14388422942792e-06],
                [8.09593706334796e-06, 0.250425889369940, 4.00127308106300e-07],
                [4.17134497258392e-06, 0.00, 5.04713828096151e-08],
                [
                    0.00,
                    0.129574571561564,
                    0.00,
                ],
                [
                    0.00,
                    0.00,
                    0.00,
                ],
            ]
        )

        self.hydrogrammes = np.array(
            [
                [0.414152217701721, 0.200169538986265],
                [0.262128566452744, 0.182565471486093],
                [0.150016889269974, 0.152741719790161],
                [0.0823877374304353, 0.123353351187427],
                [0.0442359077236253, 0.0977092347541947],
                [0.0234143573478871, 0.0764542515625204],
                [0.0122717777487500, 0.0593194797561407],
                [0.00638555316902955, 0.0457410966890291],
                [0.00330442402379796, 0.0351045964745774],
                [0.00170256913203510, 0.0268412593135917],
            ]
        )
        self.apport_vertical = [0.0548254922621093, 0, 0, -0.116328327325140, 0, 0]
        self.modules = {"mhumide": 1}

        self.apport = [0.05482549, 0.0347, 0.01298149, -0.116328327, 0, 3.791754376e-05]

    def test_hsami_ecoulement_horizontal(self):
        apport, reserve_inter, eau_hydrogrammes = hsami_ecoulement_horizontal(
            self.nb_pas,
            self.vidange_reserve_inter,
            self.reserve_inter,
            self.eau_hydrogrammes,
            self.hydrogrammes,
            self.apport_vertical,
            self.modules,
        )

        # Check the shape of the outputs
        self.assertEqual(len(apport), 6)
        self.assertEqual(eau_hydrogrammes.shape, (10, 3))

        # Check the types of the outputs
        self.assertIsInstance(apport, list)
        self.assertIsInstance(reserve_inter, float)
        self.assertIsInstance(eau_hydrogrammes, np.ndarray)

        # Check the values of the outputs
        self.assertAlmostEqual(apport[0], self.apport[0], places=2)
        self.assertAlmostEqual(apport[1], self.apport[1], places=2)
        self.assertAlmostEqual(apport[2], self.apport[2], places=2)
        self.assertAlmostEqual(apport[3], self.apport[3], places=2)
        self.assertAlmostEqual(apport[4], self.apport[4], places=2)
        self.assertAlmostEqual(apport[5], self.apport[5], places=2)
        self.assertAlmostEqual(
            np.mean(eau_hydrogrammes), np.mean(self.eau_hydrogrammes), places=2
        )

    def test_hsami_ecoulement_horizontal_no_mhumide(self):
        self.modules["mhumide"] = 0
        apport, reserve_inter, eau_hydrogrammes = hsami_ecoulement_horizontal(
            self.nb_pas,
            self.vidange_reserve_inter,
            self.reserve_inter,
            self.eau_hydrogrammes,
            self.hydrogrammes,
            self.apport_vertical,
            self.modules,
        )

        # Check the types of the outputs
        self.assertIsInstance(apport, list)
        self.assertIsInstance(reserve_inter, float)
        self.assertIsInstance(eau_hydrogrammes, np.ndarray)


if __name__ == "__main__":
    unittest.main()
