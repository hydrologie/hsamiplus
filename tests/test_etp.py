import unittest

import numpy as np

from hsamiplus.hsami_etp import (
    etp_chaleur_lat_vaporisation,
    etp_duree_jour,
    etp_m_courbe_pression,
    etp_p,
    etp_rayonnement_et,
    etp_rayonnement_g,
    etp_rayonnement_net,
    etp_rayonnement_temps_clair,
    etp_td_linacre,
    hsami_etp,
)


class TestHsamiEtp(unittest.TestCase):
    def setUp(self):
        self.pas = 1
        self.nb_pas = 1
        self.jj = 120
        self.t_min = 1.9000  # - 15.3000 / 1.9000
        self.t_max = 15.3000  # -1.9000 / 15.3000
        lat = 47.1943
        alt = 390.9
        albedo = 0.7
        self.physio = {"latitude": lat, "altitude": alt, "albedo_sol": albedo}

        self.DL = etp_duree_jour(self.jj, lat)
        self.Re = etp_rayonnement_et(lat, self.jj)
        self.rg = etp_rayonnement_g(self.Re, lat, self.jj, self.t_min, self.t_max)
        self.m = etp_m_courbe_pression(self.t_min, self.t_max)
        self.p = etp_p(lat, self.jj)
        self.lamda = etp_chaleur_lat_vaporisation(self.t_min, self.t_max)
        self.rgo = etp_rayonnement_temps_clair(self.Re, alt)
        self.Rn = etp_rayonnement_net(self.t_min, self.t_max, self.rg, self.rgo, albedo)

    def test_hsami_etp_hsami(self):
        params = {
            "pas": self.pas,
            "nb_pas": self.nb_pas,
            "jj": self.jj,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "modules": "hsami",
            "physio": self.physio,
        }
        expected = (
            0.00065 * 2.54 * 9 / 5 * (params["t_max"] - params["t_min"]) * np.exp(0.019 * (params["t_min"] * 9 / 5 + params["t_max"] * 9 / 5 + 64))
        )
        result = hsami_etp(**params)
        self.assertAlmostEqual(result, expected, places=4)

    def test_hsami_etp_blaney_criddle(self):
        params = {
            "pas": self.pas,
            "nb_pas": self.nb_pas,
            "jj": self.jj,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "modules": "blaney_criddle",
            "physio": self.physio,
        }
        t_a = (params["t_min"] + params["t_max"]) / 2
        k = 0.85
        expected = k * self.p * (0.46 * t_a + 8.13) / 10
        expected = max(0, expected)
        result = hsami_etp(**params)
        self.assertAlmostEqual(result, expected, places=4)

    def test_hsami_etp_hamon(self):
        params = {
            "pas": self.pas,
            "nb_pas": self.nb_pas,
            "jj": self.jj,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "modules": "hamon",
            "physio": self.physio,
        }
        t_a = (params["t_min"] + params["t_max"]) / 2
        es = 0.6108 * np.exp(17.27 * t_a / (t_a + 237.3))
        expected = 2.1 * self.DL**2 * es / (t_a + 273.3) / 10
        result = hsami_etp(**params)
        self.assertAlmostEqual(result, expected, places=4)

    def test_hsami_etp_linacre(self):
        params = {
            "pas": self.pas,
            "nb_pas": self.nb_pas,
            "jj": self.jj,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "modules": "linacre",
            "physio": self.physio,
        }
        # T_a not 0
        t_a = (params["t_min"] + params["t_max"]) / 2
        th = t_a + 0.006 * params["physio"]["altitude"]
        td = etp_td_linacre(params["t_max"], params["t_min"])
        lat = params["physio"]["latitude"] * 180 / np.pi
        expected = (500 * th / (100 - lat) + 15 * (t_a - td)) / (80 - t_a) / 10
        expected = max(0, expected)
        result = hsami_etp(**params)
        self.assertAlmostEqual(result, expected, places=4)

        # t-a = 0
        params["t_min"] = -2.0
        params["t_max"] = -1.0
        expected = 0
        result = hsami_etp(**params)
        self.assertAlmostEqual(result, expected, places=4)

    def test_hsami_etp_kharrufa(self):
        params = {
            "pas": self.pas,
            "nb_pas": self.nb_pas,
            "jj": self.jj,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "modules": "kharrufa",
            "physio": self.physio,
        }
        t_a = (params["t_min"] + params["t_max"]) / 2
        t_a = max(0, t_a)
        expected = 0.34 * self.p * t_a ** (1.3) / 10
        result = hsami_etp(**params)
        self.assertAlmostEqual(result, expected, places=4)

    def test_hsami_etp_mohyse(self):
        params = {
            "pas": self.pas,
            "nb_pas": self.nb_pas,
            "jj": self.jj,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "modules": "mohyse",
            "physio": self.physio,
        }
        t_a = (params["t_min"] + params["t_max"]) / 2
        delta = 0.41 * np.sin((params["jj"] - 80) / 365 * 2 * np.pi)
        expected = 1 / np.pi * np.arccos(-np.tan(params["physio"]["latitude"]) * np.tan(delta)) * np.exp((17.3 * t_a) / (238 + t_a)) / 10
        result = hsami_etp(**params)
        self.assertAlmostEqual(result, expected, places=4)

    def test_hsami_etp_romanenko(self):
        params = {
            "pas": self.pas,
            "nb_pas": self.nb_pas,
            "jj": self.jj,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "modules": "romanenko",
            "physio": self.physio,
        }
        t_a = (params["t_min"] + params["t_max"]) / 2
        ea = 0.6108 * np.exp((17.27 * t_a) / (t_a + 237.3))
        ed = 0.6108 * np.exp((17.27 * params["t_min"]) / (params["t_min"] + 237.3))
        expected = 0.0045 * (1 + t_a / 25) ** 2 * (1 - ed / ea) * 100
        result = hsami_etp(**params)
        self.assertAlmostEqual(result, expected, places=4)

    def test_hsami_etp_makkink(self):
        params = {
            "pas": self.pas,
            "nb_pas": self.nb_pas,
            "jj": self.jj,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "modules": "makkink",
            "physio": self.physio,
        }
        psi = 0.066
        expected = ((self.m / (self.m + psi)) * (0.61 * self.rg / self.lamda) - 0.12) / 10
        expected = max(0, expected)
        result = hsami_etp(**params)
        self.assertAlmostEqual(result, expected, places=4)

    def test_hsami_etp_turc(self):
        params = {
            "pas": self.pas,
            "nb_pas": self.nb_pas,
            "jj": self.jj,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "modules": "turc",
            "physio": self.physio,
        }

        # t_a not 0
        t_a = (params["t_min"] + params["t_max"]) / 2
        k = 0.35
        expected = k * (self.rg + 2.094) * (t_a / (t_a + 15)) / 10
        expected = max(0, expected)
        result = hsami_etp(**params)
        self.assertAlmostEqual(result, expected, places=4)

        # t-a = 0
        params["t_min"] = -1.0
        params["t_max"] = 1.0
        expected = 0
        result = hsami_etp(**params)
        self.assertAlmostEqual(result, expected, places=4)

    def test_hsami_etp_mcguinness_bordne(self):
        params = {
            "pas": self.pas,
            "nb_pas": self.nb_pas,
            "jj": self.jj,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "modules": "mcguinness_bordne",
            "physio": self.physio,
        }
        t_a = (params["t_min"] + params["t_max"]) / 2
        rho_w = 100
        expected = (self.rg / (self.lamda * rho_w) * (t_a + 5) / 68) * 100
        expected = max(0, expected)
        result = hsami_etp(**params)
        self.assertAlmostEqual(result, expected, places=4)

    def test_hsami_etp_abtew(self):
        params = {
            "pas": self.pas,
            "nb_pas": self.nb_pas,
            "jj": self.jj,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "modules": "abtew",
            "physio": self.physio,
        }

        # t-a not 0
        expected = 0.53 * self.rg / self.lamda / 10
        expected = max(0, expected)
        result = hsami_etp(**params)
        self.assertAlmostEqual(result, expected, places=4)

        # t-a = 0
        params["t_min"] = -1.0
        params["t_max"] = 1.0
        expected = 0
        result = hsami_etp(**params)
        self.assertAlmostEqual(result, expected, places=4)

    def test_hsami_etp_hargreaves(self):
        params = {
            "pas": self.pas,
            "nb_pas": self.nb_pas,
            "jj": self.jj,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "modules": "hargreaves",
            "physio": self.physio,
        }

        # t-a not 0
        t_a = (params["t_min"] + params["t_max"]) / 2
        expected = 0.0135 * (0.16 * self.Re * np.sqrt(params["t_max"] - params["t_min"])) * 0.4082 * (t_a + 17.8) / 10
        expected = max(0, expected)
        result = hsami_etp(**params)
        self.assertAlmostEqual(result, expected, places=4)

        # t_max - t_min < 0
        params["t_min"] = 1.0
        params["t_max"] = -1.0
        expected = 0
        result = hsami_etp(**params)
        self.assertAlmostEqual(result, expected, places=4)

    def test_hsami_etp_priestley_taylor(self):
        params = {
            "pas": self.pas,
            "nb_pas": self.nb_pas,
            "jj": self.jj,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "modules": "priestley_taylor",
            "physio": self.physio,
        }
        psi = 0.066
        rho_w = 1000
        ct = 1.26

        expected = ct * self.m * self.Rn / (self.lamda * rho_w * (self.m + psi)) * 100
        expected = max(0, expected)
        result = hsami_etp(**params)
        self.assertAlmostEqual(result, expected, places=4)


if __name__ == "__main__":
    unittest.main()
