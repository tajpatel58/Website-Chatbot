import requests
import unittest

class APITestCase(unittest.TestCase):
    def test_API(self):
        response = requests.post("http://localhost:8080/predictions/chatbot",{"message" : "Hi Babe"})
        status_code = response.status_code
        self.assertEqual(status_code, 200)


unittest.main(argv=[''], verbosity=2, exit=False)