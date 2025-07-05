import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def ask_gemini_question(image_path, question):
    image = Image.open(image_path).convert("RGB")
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
        You are a visual reasoning assistant.

        Answer the following question based ONLY on the uploaded image.

        If the question is unrelated to the image (e.g., about who you are, what time it is, or unrelated trivia), respond with:
        "I'm only able to answer questions about the image content."

        Question: "{question}"
        """


    asked_question = prompt.format(question = question)
    response = model.generate_content([image, asked_question])
    return response.text
