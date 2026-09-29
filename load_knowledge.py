from langchain_community.document_loaders import TextLoader

loader = TextLoader("knowledge/career_guide.txt")

documents = loader.load()

print("Number of documents:", len(documents))
print("\nFirst document:")
print(documents[0].page_content[:500])