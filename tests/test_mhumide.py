import sys
import unittest

import numpy as np

from hsamiplus.hsami_mhumide import hsami_mhumide


class TestHsamiMhumide(unittest.TestCase):
    def setUp(self):
        self.apport = [0.0553, 0.1455, 0.1865, 0.7883, 0]
        self.param = [0] * 50
        self.param[47] = 1.0  # hmax
        self.param[48] = 0.1  # p_norm
        self.param[49] = -2.0  # ksat (10^param[49])
        self.etat = {
            "mh_vol": 2.423423914e07,
            "mh_surf": 2.42342e03,
            "ratio_MH": 0.0092,
            "ratio_qbase": 0.0,
            "mhumide": 0.9180,
        }
        self.demande = 0.1317
        self.etr = np.array([0.0, 0.0, 0.1317, 0.0, 0.1317])
        self.physio = {"samax": 242.97}
        self.superficie = [2640, 438]

    def test_hsami_mhumide(self):
        apport, etat, etr = hsami_mhumide(
            self.apport,
            self.param,
            self.etat,
            self.demande,
            self.etr,
            self.physio,
            self.superficie,
        )

        # Check if the function returns the expected types
        self.assertIsInstance(apport, list)
        self.assertIsInstance(etat, dict)
        self.assertIsInstance(etr, np.ndarray)

        # Check if the output values are within expected ranges
        self.assertTrue(all(isinstance(x, float) for x in apport[:3]))
        self.assertTrue("mh_vol" in etat)
        self.assertTrue("mh_surf" in etat)
        self.assertTrue("ratio_MH" in etat)
        self.assertTrue("ratio_qbase" in etat)
        self.assertTrue("mhumide" in etat)
        self.assertTrue(all(isinstance(x, float) for x in etr))

        # offre_evap > demande
        self.demande = 52.423
        apport, etat, etr = hsami_mhumide(
            self.apport,
            self.param,
            self.etat,
            self.demande,
            self.etr,
            self.physio,
            self.superficie,
        )

        self.assertTrue(all(isinstance(x, float) for x in apport[:3]))
        self.assertTrue("mh_vol" in etat)
        self.assertTrue("ratio_qbase" in etat)

        # v_actuel > v_max
        self.demande = 0.1317
        self.etat["mh_surf"] = 2450.0
        self.etat["mh_vol"] = 2.45e08

        apport, etat, etr = hsami_mhumide(
            self.apport,
            self.param,
            self.etat,
            self.demande,
            self.etr,
            self.physio,
            self.superficie,
        )

        self.assertTrue(all(isinstance(x, float) for x in apport[:3]))
        self.assertTrue("mh_vol" in etat)


if __name__ == "__main__":
    unittest.main()
