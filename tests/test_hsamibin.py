import datetime
import json
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

from hsamiplus import hsamibin


class TestHsamibin(unittest.TestCase):
    @patch("hsamibin.hsami2")
    @patch("builtins.open", new_callable=mock_open, read_data='{"key": "value"}')
    @patch("os.path.join", side_effect=lambda *args: "/".join(args))
    def vrtest_hsamibin(self, mock_path_join, mock_open, mock_hsami2):
        # Mock the return value of hsami2
        mock_hsami2.return_value = ("S_value", "etats_value", "deltas_value")

        # Define the path and filename
        path = "test_path"
        filename = "test_file.json"

        # Call the function
        s, etats, deltas = hsamibin(path, filename)

        # Check that hsami2 was called with the correct parameters
        mock_hsami2.assert_called_once_with({"key": "value"})

        # Check that the output file was written correctly
        date_str = datetime.date.today().strftime("%d_%m_%Y")
        output_filename = f"output_{date_str}.json"
        mock_open.assert_called_with(f"test_path/{output_filename}", "w")
        handle = mock_open()
        handle.write.assert_called_once_with(
            json.dumps(
                {"S": "S_value", "etats": "etats_value", "deltas": "deltas_value"}
            )
        )

        # Check the return values
        self.assertEqual(S, "S_value")
        self.assertEqual(etats, "etats_value")
        self.assertEqual(deltas, "deltas_value")


if __name__ == "__main__":
    unittest.main()
