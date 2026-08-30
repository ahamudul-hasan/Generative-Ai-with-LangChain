from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite', temperature=1.5, max_completion_tokens=10)

result = model.invoke("Write a 5 line poem on football")

print(result.content)