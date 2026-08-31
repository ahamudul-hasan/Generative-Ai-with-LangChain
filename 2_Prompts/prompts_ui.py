from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite', temperature=1.5, max_completion_tokens=10)

st.header('Research Tool')

user_input = st.text_input('Enter your prompt')

if st.button('Summarize'):
    result = mod