from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite', temperature=1.5)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic':'football'})

print(result)

chain.get_graph().print_ascii()