from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# Load local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load our existing Chroma database
vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Create the RAG prompt
rag_prompt = ChatPromptTemplate.from_template("""
You are an AI Career Agent.

Use the provided career knowledge to answer the user's question.

Rules:
- Use the knowledge provided below.
- Do not invent facts.
- If the knowledge does not contain the answer, say that the information is not available in the knowledge base.
- Give practical, beginner-friendly explanations.
- Clearly distinguish facts from recommendations.

Career knowledge:
{context}

User question:
{question}

Answer:
""")

# User question
question = "What skills are needed for a Generative AI Developer?"

# Retrieve relevant information
results = vector_store.similarity_search(question, k=3)

# Combine retrieved chunks
context = "\n\n".join(
    document.page_content for document in results
)

# Create final prompt
final_prompt = rag_prompt.invoke({
    "context": context,
    "question": question
})

print("RAG pipeline working!")
print("\nRetrieved information:")
print(context)

print("\n" + "=" * 60)
print("FINAL PROMPT SENT TO THE AI:")
print("=" * 60)

print(final_prompt.to_string())