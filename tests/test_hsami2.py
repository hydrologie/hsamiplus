import json
import os
import sys
import unittest
from pathlib import Path

from hsamiplus.hsami2 import hsami2


class TestHsami2(unittest.TestCase):
    def setUp(self):

        path = Path.resolve(Path.parent(Path.parent(__file__))) / "data"
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

    def test_hsami2_modules(self):
        self.assertTrue(
            self.projet["modules"]["etp_bassin"] in self.etp_modules,
            "Le module n" "est disponible !",
        )
        self.assertTrue(
            self.projet["modules"]["etp_reservoir"] in self.etp_modules,
            "Le module n" "est disponible !",
        )
        self.assertTrue(
            self.projet["modules"]["een"] in self.een_modules,
            "Le module n" "est disponible !",
        )
        self.assertTrue(
            self.projet["modules"]["infiltration"] in self.infiltration_modules,
            "Le module n" "est disponible !",
        )
        self.assertTrue(
            self.projet["modules"]["qbase"] in self.qbase_modules,
            "Le module n" "est disponible !",
        )
        self.assertTrue(
            self.projet["modules"]["sol"] in self.sol_modules,
            "Le module n" "est disponible !",
        )
        self.assertTrue(
            self.projet["modules"]["radiation"] in self.radiation_modules,
            "Le module n" "est disponible !",
        )
        self.assertTrue(
            self.projet["modules"]["reservoir"] in self.reservoir_modules,
            "Le module n" "est disponible !",
        )
        self.assertTrue(
            self.projet["modules"]["mhumide"] in self.mhumide_modules,
            "Le module n" "est disponible !",
        )
        self.assertTrue(
            self.projet["modules"]["glace_reservoir"] in self.glace_reservoir_modules,
            "Le module n" "est disponible !",
        )

    def test_hsami2_output_structure(self):
        self.assertIsInstance(self.S, dict)
        self.assertIsInstance(self.etats, dict)
        self.assertIsInstance(self.deltas, dict)

    def test_hsami2_simulation_length(self):
        nb_pas_total = len(self.projet["meteo"]["bassin"])
        self.assertEqual(len(self.S["Qtotal"]), nb_pas_total)
        self.assertEqual(len(self.etats["neige_au_sol"]), nb_pas_total)
        self.assertEqual(len(self.deltas["total"]), nb_pas_total)

    def test_hsami2_etat(self):
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
