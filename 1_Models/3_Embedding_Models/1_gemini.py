from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2', output_dimensionality=10)

result = embedding.embed_query("Dhaka is the capital of Bangladesh")

print(str(result))