import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd

from hsamiplus.hsami_input import (
    make_project,
    meteohsami,
    paramshsami,
    physiohsami,
    writejson,
)


class TestHsamiInput(unittest.TestCase):

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="ID: 1\nnb_pas_par_jour: 24\nmemoire: 10\nsuperficie_bv: 1000\nsuperficie_reservoir: 500\n"
        "fichier_meteo_bv: meteo_bv.csv\nfichier_meteo_reservoir: meteo_res.csv\n",
    )
    def test_make_project(self, mock_file):
        data_dir = "data"
        basin_file = "bassin_versant_info.txt"
        param_file = "parametres.txt"
        projet_file = "projet.json"

        with patch("hsamiplus.hsami_input.Path.open", mock_file):
            with patch(
                "hsamiplus.hsami_input.paramshsami",
                return_value=(["param1", "param2"], pd.DataFrame()),
            ):
                with patch(
                    "hsamiplus.hsami_input.physiohsami",
                    return_value={"latitude": 45.0, "altitude": 300},
                ):
                    with patch(
                        "hsamiplus.hsami_input.meteohsami",
                        return_value=({"bassin": [], "reservoir": []}, []),
                    ):
                        with patch("hsamiplus.hsami_input.writejson") as mock_writejson:
                            make_project(data_dir, basin_file, param_file, projet_file)
                            mock_writejson.assert_called_once()

    def test_physiohsami(self):
        donnees_bv = {
            "latitude_bv": 45.0,
            "altitude_bv": 300,
            "albedo_sol": 0.23,
            "indice_orientation_bv": 2,
            "pente_bv": 3.0,
            "occupation_bv": [0.1, 0.2, 0.7],
            "coeff_reservoir": [0.1, 0.2, 0.3],
            "surface_maximale_mhe": 100,
            "niveau_reservoir": 5.4,
        }
        expected_result = {
            "latitude": 45.0,
            "altitude": 300,
            "albedo_sol": 0.23,
            "i_orientation_bv": 2,
            "pente_bv": 3.0,
            "occupation": [0.1, 0.2, 0.7],
            "coeff": [0.1, 0.2, 0.3],
            "samax": 100,
            "niveau": 5.4,
        }
        result = physiohsami(donnees_bv)
        self.assertDictEqual(result, expected_result)

    @patch("pandas.read_csv")
    def test_meteohsami(self, mock_read_csv):
        mock_df = pd.DataFrame(
            {
                "dates": pd.date_range(start="10/10/2010", periods=5),
                "tmin": [1.1, -1.0, -2.1, -4.0, 4.4],
                "tmax": [15.3, 7.4, 9.9, 10.8, 7.1],
                "pluie": [0.0, 0.0, 0.0, 0.1, 0.9],
                "neige": [0.0, 0.0, 0.0, 0.0, 0.0],
                "soleil": [0.5, 0.5, 0.5, 0.5, 0.5],
                "een": [-1.0, -1.0, -1.0, -1.0, -1.0],
                "value": [1, 2, 3, 4, 5],
            }
        ).set_index("dates")
        mock_read_csv.return_value = mock_df

        data_dir = "data"
        fichier_meteo_bv = "meteo_bv.csv"
        fichier_meteo_reservoir = "meteo_res.csv"

        expected_meteo = {
            "bassin": mock_df.to_numpy().tolist(),
            "reservoir": mock_df.to_numpy().tolist(),
        }
        expected_dates = [
            [2010, 10, 10, 0, 0],
            [2010, 10, 11, 0, 0],
            [2010, 10, 12, 0, 0],
            [2010, 10, 13, 0, 0],
            [2010, 10, 14, 0, 0],
        ]

        meteo, dates = meteohsami(data_dir, fichier_meteo_bv, fichier_meteo_reservoir)
        self.assertEqual(meteo, expected_meteo)
        self.assertEqual(dates, expected_dates)

    @patch("pandas.read_csv")
    def test_paramshsami(self, mock_read_csv):
        mock_df = pd.DataFrame(
            {
                "Nom": ["param1", "param2"],
                "min": [0, 0],
                "default": [1, 2],
                "max": [10, 20],
            }
        )
        mock_read_csv.return_value = mock_df

        param_file = "parametres.txt"
        expected_params = [1, 2]
        expected_df = mock_df

        params, df_param = paramshsami(param_file)
        self.assertEqual(params, expected_params)
        pd.testing.assert_frame_equal(df_param, expected_df)

    @patch("hsamiplus.hsami_input.Path.open", new_callable=mock_open)
    def test_writejson(self, mock_file_open):
        filename = "test.json"
        dict_var = {"key": "value"}

        writejson(filename, dict_var)

        mock_file_open.assert_called_with(filename, "w")
        mock_file_open().write.assert_called_once_with(json.dumps(dict_var, indent=4))


if __name__ == "__main__":
    unittest.main()
