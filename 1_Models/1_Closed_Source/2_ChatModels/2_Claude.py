from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model='claude-fable-5')

result = model.invoke('What is the capital of Dhaka?')

print(result.content)