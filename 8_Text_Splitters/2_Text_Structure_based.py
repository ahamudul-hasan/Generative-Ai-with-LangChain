from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

text = """The primary objective of the prompt router is to predict the minimal capable model tier for any incoming query
prior to full model invocation. A fundamental design constraint is that the computational overhead of the routing
decision—both in execution latency (∆troute) and consumed energy (∆Eroute)—must be negligible compared to the
primary inference pass. If routing overhead were significant, the energy savings achieved by directing queries to
smaller models would be diminished.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)