from langchain_core.prompts import ChatPromptTemplate

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

print("RAG prompt created successfully!")

test_prompt = rag_prompt.invoke({
    "context": "Generative AI developers commonly use Python, APIs, SQL, RAG, embeddings, vector databases, and AI agents.",
    "question": "What skills are useful for a Generative AI Developer?"
})

print("\nGenerated prompt:")
print(test_prompt.to_string())