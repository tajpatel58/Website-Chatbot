#Import Packages:
import unittest
from Scripts.chatbot import Chatbot

class ChatbotTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.chatbot = Chatbot('/Users/tajsmac/Documents/Girlfriend-Chatbot/Models/chat_model.pth')
    
    def test_response(self):
        incoming_message = 'Hello'
        response = self.chatbot.respond(incoming_message)
        possible_responses = [
                "Hello Babe",
                "How is my darling?",
                "Hello, how are you?", 
                "Hey, What you up to?"
             ]
        self.assertIn(response, possible_responses)
    
    def tearDown(self) -> None:
        del self.chatbot