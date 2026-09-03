from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-5.3",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)