from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash', 
    temperature=1.5
)

url = 'http://amazon.com/ASUS-Certified-Cybenetics-Intelligent-Stabilizer/dp/B0DQSMMCSH/ref=sr_1_1?sr=8-1'

loader = WebBaseLoader(url)

docs = loader.load()

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'question':'What is the product that we are talking about?', 'text': docs[0].page_content})

print(result)