from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2', output_dimensionality=10)

documents = [
    "I am ironman",
    "Life is pretty hard",
    "Sky is the limit"
]

result = embedding.embed_documents(documents)

print(str(result))