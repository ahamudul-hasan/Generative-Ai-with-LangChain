from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash', 
    temperature=1.5
)

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

report_gen_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>500, prompt2 | model | parser),
    RunnablePassthrough() # Default
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

result = final_chain.invoke({'topic': 'USA vs Iran'})

print(result)