
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ---------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDF_FILE = (
    PROJECT_ROOT
    / "policy_documents"
    / "MotorAI_Sample_Motor_Policy.pdf"
)

VECTOR_DB_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "policy_faiss_index"
)


# ---------------------------------------
# 2. LOAD PDF DOCUMENT
# ---------------------------------------

def ingest_policy_document():

    print("========== PDF LOADING ==========")

    if not PDF_FILE.exists():
        raise FileNotFoundError(
            f"Policy PDF not found: {PDF_FILE}"
        )

    loader = PyPDFLoader(str(PDF_FILE))

    documents = loader.load()

    print("PDF loaded successfully!")
    print("Total PDF pages:", len(documents))


    # ---------------------------------------
    # 3. SPLIT DOCUMENT INTO CHUNKS
    # ---------------------------------------

    print("\n========== TEXT CHUNKING ==========")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    print("Total text chunks:", len(chunks))

    if not chunks:
        raise ValueError("No text chunks were generated.")

    print("\nSample chunk:")
    print(chunks[0].page_content[:500])


    # ---------------------------------------
    # 4. CREATE HUGGINGFACE EMBEDDINGS
    # ---------------------------------------

    print("\n========== CREATING EMBEDDINGS ==========")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Embedding model loaded successfully!")


    # ---------------------------------------
    # 5. STORE EMBEDDINGS IN FAISS
    # ---------------------------------------

    print("\n========== CREATING FAISS DATABASE ==========")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    VECTOR_DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store.save_local(str(VECTOR_DB_PATH))

    print("FAISS database created successfully!")

    print("Total stored vectors:", vector_store.index.ntotal)

    print("FAISS location:", VECTOR_DB_PATH)


    # ---------------------------------------
    # 6. TEST VECTOR SEARCH
    # ---------------------------------------

    print("\n========== TEST VECTOR SEARCH ==========")

    question = "What documents are required for a motor insurance claim?"

    results = vector_store.similarity_search(
        question,
        k=3
    )

    print("Question:", question)

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")

        print("Page:", result.metadata.get("page", "Unknown"))

        print("Content:")
        print(result.page_content[:500])


    print("\n========== INGESTION COMPLETED ==========")


if __name__ == "__main__":
    ingest_policy_document()
