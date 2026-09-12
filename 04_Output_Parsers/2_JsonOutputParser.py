from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-5.3",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

templete = PromptTemplate(
    template="Give me the name, age and city of a fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = templete | model | parser

result = chain.invoke({})

print(result)