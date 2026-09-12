from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-3.1-flash-lite")

result1 = llm.invoke("What is the capital of Bangladesh")
result2 = llm.invoke("Who beat thanos?")

print(result1)
print(result2)