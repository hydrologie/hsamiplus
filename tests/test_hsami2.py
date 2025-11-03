import json
import unittest
from pathlib import Path

import numpy as np

from hsamiplus.hsami2 import (
    hsami2,
    hsami_etat_initial,
    hsami_simulation,
    modules_par_defaut,
    set_default_module,
)


class TestHsami2(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parent.parent.absolute() / "data"
        filename = "projet.json"

        self.etp_modules = [
            "hsami",
            "blaney_criddle",
            "hamon",
            "linacre",
            "kharrufa",
            "mohyse",
            "romanenko",
            "makkink",
            "turc",
            "mcguinness_bornde",
            "abtew",
            "hargreaves",
            "priestley-taylor",
        ]
        self.een_modules = ["hsami", "dj", "mdj", "alt"]
        self.infiltration_modules = ["hsami", "green_ampt", "scs_cn"]
        self.sol_modules = ["hsami", "3couches"]
        self.qbase_modules = ["hsami", "dingman"]
        self.radiation_modules = ["hsami", "mdj"]
        self.mhumide_modules = [0, 1]
        self.reservoir_modules = [0, 1]
        self.glace_reservoir_modules = [0, "stefan", "mylake"]

        with Path.open(Path(path) / filename) as file:
            self.projet = json.load(file)

        self.s, self.etats, self.deltas = hsami2(self.projet)

    def test_hsami2_required_fields(self):
        required_fields = [
            "superficie",
            "param",
            "memoire",
            "physio",
            "modules",
            "meteo",
            "dates",
            "nb_pas_par_jour",
        ]
        for field in required_fields:
            self.assertIn(field, self.projet)

    def test_set_default_modules(self):
        modules = {}
        valeurs = {
            "etp_bassin": "hsami",
            "etp_reservoir": "hsami",
            "een": "hsami",
            "infiltration": "hsami",
            "qbase": "hsami",
            "sol": "hsami",
            "radiation": "hsami",
            "reservoir": 0,
            "mhumide": 0,
            "glace_reservoir": 0,
        }

        for key, value in valeurs.items():
            set_default_module(modules, key, value)

        self.assertEqual(modules["etp_bassin"], "hsami")
        self.assertEqual(modules["etp_reservoir"], "hsami")
        self.assertEqual(modules["een"], "hsami")
        self.assertEqual(modules["infiltration"], "hsami")
        self.assertEqual(modules["qbase"], "hsami")
        self.assertEqual(modules["sol"], "hsami")
        self.assertEqual(modules["radiation"], "hsami")
        self.assertEqual(modules["reservoir"], 0)
        self.assertEqual(modules["mhumide"], 0)
        self.assertEqual(modules["glace_reservoir"], 0)

    def test_modules_par_defaut(self):
        modules = self.projet["modules"]
        modules_par_defaut(modules)

        self.assertEqual(modules["etp_bassin"], "hsami")
        self.assertEqual(modules["etp_reservoir"], "hsami")
        self.assertEqual(modules["een"], "hsami")
        self.assertEqual(modules["infiltration"], "hsami")
        self.assertEqual(modules["qbase"], "hsami")
        self.assertEqual(modules["sol"], "hsami")
        self.assertEqual(modules["radiation"], "hsami")
        self.assertEqual(modules["reservoir"], 0)
        self.assertEqual(modules["mhumide"], 1)
        self.assertEqual(modules["glace_reservoir"], "stefan")

    def test_hsami_etat_initial(self):
        # Dictionnaire états entrants
        etat = {}

        etat["eau_hydrogrammes"] = np.zeros((int(self.projet["memoire"]), 3))

        if self.projet["modules"]["een"] in ["mdj", "alt"]:
            if self.projet["modules"]["een"] == "mdj":
                n = len(self.projet["physio"]["occupation"])
            if self.projet["modules"]["een"] == "alt":
                n = len(self.projet["physio"]["occupation_bande"])

            etat["modules"] = {}

            etat[self.projet["modules"]["een"]] = {
                "couvert_neige": [0] * n,
                "densite_neige": [0] * n,
                "albedo_neige": [0.9] * n,
                "neige_au_sol": [0] * n,
                "fonte": [0] * n,
                "gel": [0] * n,
                "sol": [0] * n,
                "energie_neige": [0] * n,
                "energie_glace": 0,
            }

        etat["neige_au_sol"] = 0
        etat["fonte"] = 0
        etat["nas_tot"] = 0
        etat["fonte_tot"] = 0
        etat["derniere_neige"] = 0
        etat["gel"] = 0
        etat["nappe"] = self.projet["param"][13]
        etat["reserve"] = 0

        if self.projet["modules"]["sol"] == "hsami":
            etat["sol"] = np.array([self.projet["param"][11], np.nan])

        elif self.projet["modules"]["sol"] == "3couches":
            etat["sol"] = np.array(
                [
                    self.projet["param"][42] * self.projet["param"][39],
                    self.projet["param"][43] * self.projet["param"][40],
                ]
            )

        if self.projet["modules"]["mhumide"] == 1:
            if self.projet["physio"]["samax"] == 0:
                raise ValueError(
                    "La superficie maximale du milieu humide \
                                équivalent est égale à 0."
                )

            etat["mh_surf"] = self.projet["param"][48] * self.projet["physio"]["samax"] * 100
            etat["mh_vol"] = self.projet["param"][48] * (self.projet["param"][47] * self.projet["physio"]["samax"] * 100 * 10000)
            etat["ratio_MH"] = etat["mh_surf"] / (self.projet["superficie"][0] * 100)

        if self.projet["modules"]["mhumide"] == 0:
            etat["mh_vol"] = 0
            etat["ratio_MH"] = 0
            etat["mh_surf"] = 1

        etat["mhumide"] = etat["mh_vol"] * etat["ratio_MH"] / (etat["mh_surf"] * 100)
        etat["ratio_qbase"] = 0

        # Glace/réservoir
        etat["cumdegGel"] = 0
        etat["obj_gel"] = -200
        etat["dernier_gel"] = 0
        etat["reservoir_epaisseur_glace"] = 0
        etat["reservoir_energie_glace"] = 0
        etat["reservoir_superficie"] = self.projet["superficie"][1]
        etat["reservoir_superficie_glace"] = 0
        etat["reservoir_superficie_ref"] = etat["reservoir_superficie"]
        etat["eeg"] = np.zeros(5000)
        etat["ratio_bassin"] = 1
        etat["ratio_reservoir"] = 0
        etat["ratio_fixe"] = 1

        etat_initial = hsami_etat_initial(
            self.projet,
            self.projet["param"],
            self.projet["modules"],
            self.projet["physio"],
            self.projet["superficie"],
            etat,
        )
        self.assertIsInstance(etat_initial, dict)
        self.assertIn("eau_hydrogrammes", etat_initial)
        self.assertIn("neige_au_sol", etat_initial)
        self.assertIn("fonte", etat_initial)
        self.assertIn("nas_tot", etat_initial)
        self.assertIn("fonte_tot", etat_initial)
        self.assertIn("derniere_neige", etat_initial)
        self.assertIn("gel", etat_initial)
        self.assertIn("nappe", etat_initial)
        self.assertIn("reserve", etat_initial)

    def test_hsami_simulation(self):
        etat = {}

        etat["eau_hydrogrammes"] = np.zeros((int(self.projet["memoire"]), 3))

        if self.projet["modules"]["een"] in ["mdj", "alt"]:
            if self.projet["modules"]["een"] == "mdj":
                n = len(self.projet["physio"]["occupation"])
            if self.projet["modules"]["een"] == "alt":
                n = len(self.projet["physio"]["occupation_bande"])

            etat["modules"] = {}

            etat[self.projet["modules"]["een"]] = {
                "couvert_neige": [0] * n,
                "densite_neige": [0] * n,
                "albedo_neige": [0.9] * n,
                "neige_au_sol": [0] * n,
                "fonte": [0] * n,
                "gel": [0] * n,
                "sol": [0] * n,
                "energie_neige": [0] * n,
                "energie_glace": 0,
            }

        etat["neige_au_sol"] = 0
        etat["fonte"] = 0
        etat["nas_tot"] = 0
        etat["fonte_tot"] = 0
        etat["derniere_neige"] = 0
        etat["gel"] = 0
        etat["nappe"] = self.projet["param"][13]
        etat["reserve"] = 0

        if self.projet["modules"]["sol"] == "hsami":
            etat["sol"] = np.array([self.projet["param"][11], np.nan])

        elif self.projet["modules"]["sol"] == "3couches":
            etat["sol"] = np.array(
                [
                    self.projet["param"][42] * self.projet["param"][39],
                    self.projet["param"][43] * self.projet["param"][40],
                ]
            )

        if self.projet["modules"]["mhumide"] == 1:
            if self.projet["physio"]["samax"] == 0:
                raise ValueError(
                    "La superficie maximale du milieu humide \
                                équivalent est égale à 0."
                )

            etat["mh_surf"] = self.projet["param"][48] * self.projet["physio"]["samax"] * 100
            etat["mh_vol"] = self.projet["param"][48] * (self.projet["param"][47] * self.projet["physio"]["samax"] * 100 * 10000)
            etat["ratio_MH"] = etat["mh_surf"] / (self.projet["superficie"][0] * 100)

        if self.projet["modules"]["mhumide"] == 0:
            etat["mh_vol"] = 0
            etat["ratio_MH"] = 0
            etat["mh_surf"] = 1

        etat["mhumide"] = etat["mh_vol"] * etat["ratio_MH"] / (etat["mh_surf"] * 100)
        etat["ratio_qbase"] = 0

        # Glace/réservoir
        etat["cumdegGel"] = 0
        etat["obj_gel"] = -200
        etat["dernier_gel"] = 0
        etat["reservoir_epaisseur_glace"] = 0
        etat["reservoir_energie_glace"] = 0
        etat["reservoir_superficie"] = self.projet["superficie"][1]
        etat["reservoir_superficie_glace"] = 0
        etat["reservoir_superficie_ref"] = etat["reservoir_superficie"]
        etat["eeg"] = np.zeros(5000)
        etat["ratio_bassin"] = 1
        etat["ratio_reservoir"] = 0
        etat["ratio_fixe"] = 1

        etat_initial = hsami_etat_initial(
            self.projet,
            self.projet["param"],
            self.projet["modules"],
            self.projet["physio"],
            self.projet["superficie"],
            etat,
        )
        nb_pas_total = len(self.projet["meteo"]["bassin"])

        etats = {}
        f = list(etat.keys())
        for i_f in range(len(f)):
            etats[f[i_f]] = []

        s = {
            "Qtotal": [],
            "Qbase": [],
            "Qinter": [],
            "Qsurf": [],
            "Qreservoir": [],
            "Qglace": [],
            "ETP": [],
            "ETRtotal": [],
            "ETRsublim": [],
            "ETRPsurN": [],
            "ETRintercept": [],
            "ETRtranspir": [],
            "ETRreservoir": [],
            "ETRmhumide": [],
            "Qmh": [],
        }

        deltas = {
            "total": [],
            "glace": [],
            "interception": [],
            "ruissellement": [],
            "vertical": [],
            "mhumide": [],
            "horizontal": [],
        }

        etat = hsami_etat_initial(
            self.projet,
            self.projet["param"],
            self.projet["modules"],
            self.projet["physio"],
            self.projet["superficie"],
            etat_initial,
        )

        s, etats, deltas = hsami_simulation(
            self.projet,
            self.projet["param"],
            self.projet["modules"],
            self.projet["physio"],
            self.projet["superficie"],
            etat,
            nb_pas_total,
            s,
            etats,
            deltas,
        )

        self.assertIsInstance(self.s, dict)
        self.assertIsInstance(self.etats, dict)
        self.assertIsInstance(self.deltas, dict)

    def test_hsami2_modules(self):
        self.assertTrue(
            self.projet["modules"]["etp_bassin"] in self.etp_modules,
            "Le module nest disponible !",
        )
        self.assertTrue(
            self.projet["modules"]["etp_reservoir"] in self.etp_modules,
            "Le module nest disponible !",
        )
        self.assertTrue(
            self.projet["modules"]["een"] in self.een_modules,
            "Le module nest disponible !",
        )
        self.assertTrue(
            self.projet["modules"]["infiltration"] in self.infiltration_modules,
            "Le module nest disponible !",
        )
        self.assertTrue(
            self.projet["modules"]["qbase"] in self.qbase_modules,
            "Le module nest disponible !",
        )
        self.assertTrue(
            self.projet["modules"]["sol"] in self.sol_modules,
            "Le module nest disponible !",
        )
        self.assertTrue(
            self.projet["modules"]["radiation"] in self.radiation_modules,
            "Le module nest disponible !",
        )
        self.assertTrue(
            self.projet["modules"]["reservoir"] in self.reservoir_modules,
            "Le module nest disponible !",
        )
        self.assertTrue(
            self.projet["modules"]["mhumide"] in self.mhumide_modules,
            "Le module nest disponible !",
        )
        self.assertTrue(
            self.projet["modules"]["glace_reservoir"] in self.glace_reservoir_modules,
            "Le module nest disponible !",
        )

    def test_hsami2_output_structure(self):
        self.assertIsInstance(self.s, dict)
        self.assertIsInstance(self.etats, dict)
        self.assertIsInstance(self.deltas, dict)

    def test_hsami2_simulation_length(self):
        nb_pas_total = len(self.projet["meteo"]["bassin"])
        self.assertEqual(len(self.s["Qtotal"]), nb_pas_total)
        self.assertEqual(len(self.etats["neige_au_sol"]), nb_pas_total)
        self.assertEqual(len(self.deltas["total"]), nb_pas_total)

    def test_hsami2_etats(self):
        etats = self.etats
        self.assertIn("eau_hydrogrammes", etats)
        self.assertIn("neige_au_sol", etats)
        self.assertIn("fonte", etats)
        self.assertIn("nas_tot", etats)
        self.assertIn("fonte_tot", etats)
        self.assertIn("derniere_neige", etats)
        self.assertIn("gel", etats)
        self.assertIn("nappe", etats)
        self.assertIn("reserve", etats)

    def test_hsami2_simulation_output(self):
        s = self.s
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
        self.assertIn("Qmh", s)


if __name__ == "__main__":
    unittest.main()
