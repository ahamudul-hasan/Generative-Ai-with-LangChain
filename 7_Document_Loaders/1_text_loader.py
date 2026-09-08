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

loader = TextLoader('football.txt', encoding='utf-8')

docs = loader.load()

prompt = PromptTemplate(
    template='Write a summary for the followning peom - \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'poem':docs[0].page_content})

print(result)