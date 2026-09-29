from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

question = "What skills are needed for a Generative AI Developer?"

results = vector_store.similarity_search(
    question,
    k=3
)

print("\nRelevant knowledge:\n")

for i, doc in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print(doc.page_content)
    print()