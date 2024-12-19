import sys
import unittest

from hsamiplus.hsami_ecoulement_vertical import (
    hsami_ecoulement_vertical,
    scs_cn,
    vidange_nappe,
)


class TestHsamiEcoulementVertical(unittest.TestCase):
    def setUp(self):
        self.nb_pas = 1.0
        self.param = [0] * 50  # Assuming 50 parameters for simplicity
        self.param[11] = 0.05  # sol_min
        self.param[12] = 10  # sol_max
        self.param[13] = 8.0  # nappe_max
        self.param[14] = 0.25  # portion_ruissellement_surface
        self.param[15] = 0.2  # portion_ruissellement_sol_max
        self.param[16] = 0.01  # taux_vidange_sol_min
        self.param[17] = 0.008  # taux_vidange_nappe
        self.param[26] = 0.06  # coeff. de récession
        self.param[23] = 25  # Curve Number (CN)
        self.param[24] = 0.30  # Puissance de la cond. hydraulique
        self.param[27] = 0.01  # specific yield
        self.param[36] = 0.0  # Indice de distribution de la taille des pores
        self.param[37] = 0.02  # pore-size distribution index (adim.)
        self.param[38] = 4.0  # cond. hyd. sat. (cm/j)
        self.param[39] = 10.0  # épaisseur des couches (cm)
        self.param[40] = 30.0  # épaisseur des couches (cm)
        self.param[41] = 5.0  # Point de flétrissement permanent
        self.param[42] = 10.0
        self.param[43] = 0.05  # capacité au champ (cm/cm)
        self.param[44] = 0.20  # Porosité couche 1
        self.param[45] = 0.20  # Porosité couche 2

        self.etat = {
            "sol": [5.8012, 1.52],  # [5.8012, np.nan]
            "nappe": 7.5889,
            "gel": 0,
            "neige_au_sol": 0,
        }
        self.offre = 0.0
        self.demande = 0.1163
        self.modules = {"sol": "hsami", "qbase": "hsami", "infiltration": "hsami"}
        self.ruissellement_surface = 0.0
        self.apport_vertical = [0.0, 0.0, 0.0, -0.1163, 0.0]
        self.etr = [0.0, 0.0, 0.0, 0.1163]

    def test_hsami_ecoulement_vertical_hsami(self):
        apport, etat, etr = hsami_ecoulement_vertical(
            self.nb_pas,
            self.param,
            self.etat,
            self.offre,
            self.demande,
            self.modules,
            self.ruissellement_surface,
            self.apport_vertical,
            self.etr,
        )
        self.assertIsNotNone(apport)
        self.assertIsNotNone(etat)
        self.assertIsNotNone(etr)

        # Check the shape of the outputs
        self.assertEqual(len(apport), 5)
        self.assertEqual(len(etr), 4)

        # Check infilitration
        self.modules["infiltration"] = "green_ampt"
        apport, etat, etr = hsami_ecoulement_vertical(
            self.nb_pas,
            self.param,
            self.etat,
            self.offre,
            self.demande,
            self.modules,
            self.ruissellement_surface,
            self.apport_vertical,
            self.etr,
        )
        self.assertIsNotNone(apport)
        self.assertIsNotNone(etat)
        self.assertIsNotNone(etr)

        # Check the shape of the outputs
        self.assertEqual(len(apport), 5)
        self.assertEqual(len(etr), 4)

        self.modules["infiltration"] = "scs_cn"
        apport, etat, etr = hsami_ecoulement_vertical(
            self.nb_pas,
            self.param,
            self.etat,
            self.offre,
            self.demande,
            self.modules,
            self.ruissellement_surface,
            self.apport_vertical,
            self.etr,
        )
        self.assertIsNotNone(apport)
        self.assertIsNotNone(etat)
        self.assertIsNotNone(etr)

        # Check the shape of the outputs
        self.assertEqual(len(apport), 5)
        self.assertEqual(len(etr), 4)

        # Check qbase
        self.modules["qbase"] = "dingman"
        apport, etat, etr = hsami_ecoulement_vertical(
            self.nb_pas,
            self.param,
            self.etat,
            self.offre,
            self.demande,
            self.modules,
            self.ruissellement_surface,
            self.apport_vertical,
            self.etr,
        )
        self.assertIsNotNone(apport)
        self.assertIsNotNone(etat)
        self.assertIsNotNone(etr)

        # Check the shape of the outputs
        self.assertEqual(len(apport), 5)
        self.assertEqual(len(etr), 4)

        self.modules["qbase"] = "hsami"
        apport, etat, etr = hsami_ecoulement_vertical(
            self.nb_pas,
            self.param,
            self.etat,
            self.offre,
            self.demande,
            self.modules,
            self.ruissellement_surface,
            self.apport_vertical,
            self.etr,
        )
        self.assertIsNotNone(apport)
        self.assertIsNotNone(etat)
        self.assertIsNotNone(etr)

        # Check the shape of the outputs
        self.assertEqual(len(apport), 5)
        self.assertEqual(len(etr), 4)

    def test_hsami_ecoulement_vertical_3couches(self):
        self.modules["sol"] = "3couches"
        apport, etat, etr = hsami_ecoulement_vertical(
            self.nb_pas,
            self.param,
            self.etat,
            self.offre,
            self.demande,
            self.modules,
            self.ruissellement_surface,
            self.apport_vertical,
            self.etr,
        )
        self.assertIsNotNone(apport)
        self.assertIsNotNone(etat)
        self.assertIsNotNone(etr)

        # Check the shape of the outputs
        self.assertEqual(len(apport), 5)
        self.assertEqual(len(etr), 4)

        # Check infilitration
        self.modules["infiltration"] = "green_ampt"
        apport, etat, etr = hsami_ecoulement_vertical(
            self.nb_pas,
            self.param,
            self.etat,
            self.offre,
            self.demande,
            self.modules,
            self.ruissellement_surface,
            self.apport_vertical,
            self.etr,
        )
        self.assertIsNotNone(apport)
        self.assertIsNotNone(etat)
        self.assertIsNotNone(etr)

        # Check the shape of the outputs
        self.assertEqual(len(apport), 5)
        self.assertEqual(len(etr), 4)

        self.modules["infiltration"] = "scs_cn"
        apport, etat, etr = hsami_ecoulement_vertical(
            self.nb_pas,
            self.param,
            self.etat,
            self.offre,
            self.demande,
            self.modules,
            self.ruissellement_surface,
            self.apport_vertical,
            self.etr,
        )
        self.assertIsNotNone(apport)
        self.assertIsNotNone(etat)
        self.assertIsNotNone(etr)

        # Check the shape of the outputs
        self.assertEqual(len(apport), 5)
        self.assertEqual(len(etr), 4)

    def test_vidange_nappe(self):
        apport, nappe, sol = vidange_nappe(
            self.apport_vertical,
            self.etat["nappe"],
            self.param[17],
            self.param[13],
            self.nb_pas,
            self.modules,
            self.param,
            self.etat["sol"],
        )

        self.assertIsNotNone(apport)
        self.assertIsNotNone(nappe)
        self.assertIsNotNone(sol)

        # Check the shape of the outputs
        self.assertEqual(len(apport), 5)

    def test_sc_cn(self):
        infiltration, ruissellement = scs_cn(self.offre, self.param[23])

        self.assertIsNotNone(infiltration)
        self.assertIsNotNone(ruissellement)


if __name__ == "__main__":
    unittest.main()
