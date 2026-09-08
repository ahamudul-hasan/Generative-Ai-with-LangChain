from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('Data/Green.pdf')

docs = loader.lazy_load()
first_doc = next(docs)

print(first_doc.page_content)
print(first_doc.metadata)