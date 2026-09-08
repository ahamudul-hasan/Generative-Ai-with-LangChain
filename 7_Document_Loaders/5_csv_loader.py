from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='Data/Social_Network_Ads.csv')

data = loader.load()

print(data[0])