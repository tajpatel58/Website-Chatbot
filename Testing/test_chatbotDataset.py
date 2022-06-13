import unittest
from Scripts.chat_bot_dataset import ChatbotDataset

class DatasetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = ChatbotDataset()
    
    def test_label_mapping(self):
        labels = list(self.dataset.label_mapping.keys())
        all_labels = ['greeting', 'working_day', 'feelings_or_actions', 'goodbye', 
             'day_intents', 'food_intents', 'food_response', 
             'morning_greeting', 'goodbye_night']
        self.assertEqual(labels, all_labels)

    def test_bag(self):
        expected_words_in_bag = ['pasta', 'hello', 'relax']
        for word in expected_words_in_bag:
            self.assertIn(word, self.dataset.bag)


unittest.main(argv=[''], verbosity=2, exit=False)