
import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
chat_model = os.getenv("GROQ_MODEL")

if not api_key:
    raise ValueError("API key initialization Error")

if not chat_model:
    raise ValueError("Model initialization error")

client = Groq(api_key=api_key)