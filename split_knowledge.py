from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("knowledge/career_guide.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Original documents:", len(documents))
print("Total chunks:", len(chunks))

print("\nFirst chunk:")
print(chunks[0].page_content)