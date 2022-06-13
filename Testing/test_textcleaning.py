import unittest
from Scripts.text_cleaning import clean_text
from nltk.stem import PorterStemmer

class TextCleaningTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.ps = PorterStemmer()

    def test_clean_text(self):
        text_example = 'WHAT a GREAT day'
        expected_output = ['what', 'a', 'great', 'day']
        cleaned_message = clean_text(text_example, self.ps)
        self.assertEqual(expected_output, cleaned_message)
    
    def tearDown(self) -> None:
        del self.ps