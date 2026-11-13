
from pathlib import Path
import re

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_DB_PATH = (
    PROJECT_ROOT / "artifacts" / "policy_faiss_index"
)


# Load embedding model
print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Embedding model loaded successfully!")


# Load FAISS database
print("\nLoading FAISS database...")

vector_store = FAISS.load_local(
    str(VECTOR_DB_PATH),
    embedding_model,
    allow_dangerous_deserialization=True
)

print("FAISS database loaded!")
print("Stored vectors:", vector_store.index.ntotal)


# Clean extracted PDF text
def clean_text(text):

    text = text.replace("\x7f", "\n- ")
    text = text.replace("\uf0a7", "\n- ")
    text = text.replace("\u25a0", "")

    text = re.sub(r"[\x00-\x08\x0b-\x1f]", " ", text)

    return text.strip()


# Retrieve policy information
def ask_policy_question(question):

    results = vector_store.similarity_search(
        question,
        k=3
    )

    if not results:
        print("No relevant information found.")
        return

    seen = set()

    for index, document in enumerate(results, start=1):

        content = clean_text(document.page_content)

        if content in seen:
            continue

        seen.add(content)

        page = document.metadata.get("page")

        print(f"\nSource {index}")

        if page is not None:
            print("PDF page:", page + 1)

        print("-" * 50)
        print(content)
        print("-" * 50)

    print("\nThese are retrieved excerpts, not an LLM-generated answer.")


# Interactive question loop
if __name__ == "__main__":

    print("\nMOTORAI POLICY ASSISTANT")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nEnter your policy question: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Policy assistant stopped.")
            break

        if not question:
            print("Please enter a question.")
            continue

        ask_policy_question(question)
