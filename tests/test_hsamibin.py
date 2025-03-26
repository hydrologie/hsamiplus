import datetime
import json
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from hsamiplus.hsamibin import hsamibin


class TestHsamibin(unittest.TestCase):
    def setUp(self):
        self.path = Path(__file__).parent.parent.absolute() / "data"
        self.filename = "projet.json"
        self.mock_s = {
            "Qtotal": [12.0, 11.7, 11.5],
            "ETP": [0.033, 0.04, 0.039],
        }
        self.mock_etats = {
            "sol": [[9.15, 1.34], [9.11, 1.26], [9.078, 1.14]],
            "neige_au_sol": [5.35, 5.33, 5.31],
        }
        self.mock_deltas = {
            "total": [0.0, 0.0, 0.0],
            "vertical": [0.0, 0.0, 0.0],
        }

    @patch(
        "pathlib.Path.open",
        new_callable=mock_open,
        read_data='{"test_key": "test_value"}',
    )
    def test_load_projet_json(self, mock_file):
        # Mock the json.load function
        with Path.open(Path(self.path) / self.filename) as file:
            projet = json.load(file)

        # Check if the projet file is called correctly
        self.assertEqual(projet.get("test_key"), "test_value")
        mock_file.assert_called_once_with(Path(self.path) / self.filename)

    def test_hsamibin_execution(self):
        # Run hsamibin
        s, etats, deltas = hsamibin(self.path, self.filename)

        # Check if the return values of hsamibin are correct
        self.assertIsInstance(s, dict)
        self.assertIsInstance(etats, dict)
        self.assertIsInstance(deltas, dict)

    @patch("pathlib.Path.open", new_callable=mock_open)
    def test_write_output_file(self, mock_file):
        # Date
        date = datetime.date(2025, 1, 1)

        # Perform snippet logic
        output = {
            "s": self.mock_s,
            "etats": self.mock_etats,
            "deltas": self.mock_deltas,
        }
        output_json = json.dumps(output)
        output_file = "output_" + date.strftime("%d_%m_%Y") + ".json"

        with Path.open(Path(self.path) / output_file, "w") as f:
            f.write(output_json)

        # Check calls
        mock_file.assert_called_once_with(
            Path(self.path) / "output_01_01_2025.json", "w"
        )

        # Check if the output file was written correctly
        mock_file().write.assert_called_once_with(output_json)


if __name__ == "__main__":
    unittest.main()
