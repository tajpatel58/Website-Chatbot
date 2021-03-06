import requests
import unittest

class LocalAPITestCase(unittest.TestCase):
    def test_API(self):
        response = requests.post("http://localhost:8080/ping")
        status_code = response.status_code
        self.assertEqual(status_code, 200)

unittest.main(argv=[''], verbosity=2, exit=False)