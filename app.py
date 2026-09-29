import os

import streamlit as st
from dotenv import load_dotenv
from google import genai

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Career Agent",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


# ============================================================
# TITLE
# ============================================================

st.title("🤖 AI Career Agent")

st.write(
    "Your AI-powered career assistant using "
    "RAG, LangChain, Chroma and Gemini."
)


# ============================================================
# LOAD RAG COMPONENTS
# ============================================================

@st.cache_resource
def load_rag_components():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    return vector_store


vector_store = load_rag_components()


# ============================================================
# LOAD GEMINI
# ============================================================

@st.cache_resource
def load_gemini():

    if not api_key:
        return None

    return genai.Client(api_key=api_key)


client = load_gemini()


# ============================================================
# LOCAL FALLBACK MODEL
# ============================================================

@st.cache_resource
def load_local_model():

    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model


def generate_local_answer(prompt):

    tokenizer, model = load_local_model()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=180,
        temperature=0.7,
        do_sample=True
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer


# ============================================================
# RAG PROMPT
# ============================================================

rag_prompt = ChatPromptTemplate.from_template("""
You are an AI Career Agent.

Use the provided career knowledge to answer the user's question.

Rules:
- Use the provided knowledge as your primary source.
- Do not invent facts.
- If the knowledge does not contain the answer, clearly say so.
- Give practical and beginner-friendly explanations.
- Clearly distinguish information from recommendations.

Career knowledge:
{context}

User question:
{question}

Answer:
""")


# ============================================================
# USER QUESTION
# ============================================================

st.subheader("Ask your career question")

question = st.text_area(
    "Question",
    placeholder="Example: What skills should I learn for a Generative AI Developer?",
    height=120
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button("🚀 Ask Career Agent", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")
        st.stop()


    # --------------------------------------------------------
    # STEP 1: SEARCH KNOWLEDGE BASE
    # --------------------------------------------------------

    with st.spinner("🔎 Searching career knowledge..."):

        results = vector_store.similarity_search(
            question,
            k=3
        )


    context = "\n\n".join(
        document.page_content
        for document in results
    )


    # --------------------------------------------------------
    # BUILD PROMPT
    # --------------------------------------------------------

    final_prompt = rag_prompt.invoke({
        "context": context,
        "question": question
    })


    prompt_text = final_prompt.to_string()


    # --------------------------------------------------------
    # STEP 2: TRY GEMINI
    # --------------------------------------------------------

    answer = None
    used_model = None

    if client:

        with st.spinner("🤖 Generating answer with Gemini..."):

            try:

                interaction = client.interactions.create(
                    model="gemini-3.8-flash",
                    input=prompt_text,
                    generation_config={
                        "thinking_level": "low"
                    }
                )

                answer = interaction.output_text
                used_model = "Gemini"


            except Exception as e:

                error_message = str(e)

                if (
                    "429" in error_message
                    or "RESOURCE_EXHAUSTED" in error_message
                    or "quota" in error_message.lower()
                ):

                    st.info(
                        "Gemini quota is currently unavailable. "
                        "Switching to the local AI model..."
                    )

                else:

                    st.info(
                        "Gemini is currently unavailable. "
                        "Switching to the local AI model..."
                    )


    # --------------------------------------------------------
    # STEP 3: LOCAL FALLBACK
    # --------------------------------------------------------

    if answer is None:

        with st.spinner(
            "🧠 Generating answer with local AI model..."
        ):

            try:

                answer = generate_local_answer(
                    prompt_text
                )

                used_model = "Local Hugging Face Model"

            except Exception as e:

                st.error(
                    "The local AI model could not be loaded."
                )

                with st.expander("Technical details"):
                    st.code(str(e))

                st.stop()


    # --------------------------------------------------------
    # DISPLAY ANSWER
    # --------------------------------------------------------

    st.subheader("💡 AI Career Agent")

    st.success(
        f"Response generated using: **{used_model}**"
    )

    st.write(answer)


    # --------------------------------------------------------
    # RETRIEVED KNOWLEDGE
    # --------------------------------------------------------

    with st.expander("📚 Retrieved career knowledge"):

        for i, document in enumerate(results, 1):

            st.markdown(f"**Source {i}**")

            st.write(document.page_content)