import unittest
import os
# Adjust import path if necessary. Assuming 'object_detector' is in PYTHONPATH
# or tests are run from the project root.
from object_detector.main import load_model

class TestMain(unittest.TestCase):

    def test_load_default_model(self):
        """
        Tests if the default model from TensorFlow Hub can be loaded.
        This test requires an internet connection.
        """
        model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
        model = load_model(model_url)
        self.assertIsNotNone(model, "Model should not be None if loading is successful.")
        self.assertTrue(callable(model), "Loaded model should be callable.")

if __name__ == '__main__':
    # This allows running the test file directly,
    # but 'python -m unittest discover tests' is preferred.
    unittest.main()
