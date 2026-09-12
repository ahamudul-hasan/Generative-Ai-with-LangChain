from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(model='gemini-embedding-2', output_dimensionality=300)

documents = [
    "Dhaka is the capital of Bangladesh and is known for its busy streets, rich history, and vibrant culture.",
    "Cox's Bazar is a popular tourist destination in Bangladesh and is famous for having one of the world's longest natural sea beaches.",
    "Jamdani is a traditional Bangladeshi handwoven textile famous for its intricate patterns and delicate craftsmanship.",
    "Sonargaon is an ancient city in Bangladesh known for its historical architecture, folk art, and cultural heritage.",
    "Bangladesh is famous for its diverse cuisine, including biryani, hilsa fish, pitha, and bhorta."
]

query = 'Tell me about Jamdani'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("Similarity score is ", score)