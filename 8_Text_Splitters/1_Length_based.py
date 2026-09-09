from langchain_classic.text_splitter import CharacterTextSplitter

text = """The primary objective of the prompt router is to predict the minimal capable model tier for any incoming query
prior to full model invocation. A fundamental design constraint is that the computational overhead of the routing
decision—both in execution latency (∆troute) and consumed energy (∆Eroute)—must be negligible compared to the
primary inference pass. If routing overhead were significant, the energy savings achieved by directing queries to
smaller models would be diminished."""

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
) 

result = splitter.split_text(text)

print(result)