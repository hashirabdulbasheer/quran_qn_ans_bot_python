#
# This chatbot is designed to provide relevant verses from the Noble Quran
# that may address a question, along with a potential answer derived
# from those verses using an AI-powered language model
#
# uxquran.com
#

import os
import openai
from ai_chat import QuranOpenChat

openai.api_key  = os.environ['OPENAI_API_KEY']

chat = QuranOpenChat("input.json")
chat.initialize()

while True:
    try:
        print("")
        inp = input("Enter question (type 'quit' to exit):").strip()
        if inp == "quit":
            break

        answer = chat.get_answer(inp + " from the quran context")
        print(answer)
        
    except Exception as error:
        break 

print("Thank You")