from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableParallel
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
model2 = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and  quiz into a single document \n note -> {notes} and {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

with open(Path(__file__).resolve().parent / "text.txt", "r", encoding="utf-8") as file:
    text = file.read()

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()