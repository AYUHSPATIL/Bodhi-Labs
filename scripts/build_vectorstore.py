import logging
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# Imp paths
DATA_PATH = "data/processed/data_processed.xlsx"
CHROMA_DIR = "vectorstore/chroma"
COLLECTION_NAME = "question_bank"
EMBED_MODEL = "abhinand/MedEmbed-small-v0.1"


# Embedding Fine-tuned on Medical Data
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device":"cpu"},
    encode_kwargs={"normalize_embeddings":True}
)


# Loads Documents
def load_documents(xlsx_path: str):
    df = pd.read_excel(xlsx_path)

    documents = []
    ids = []

    for _, row in df.iterrows():
        metadata = {
            "question_id": row["question_id"],
            "correct_option_id": row["correct_option_id"],
            "tag_ids": row["tag_ids"],
            "domain": row["domain"],
            "competency_name": row["competency_name"],
            "competency_area": row["competency_area"],
            "competency_definition": row["competency_definition"],
            "type": row["type"],
        }

        documents.append(
            Document(
                page_content=row["content"],
                metadata=metadata
            )
        )

        # Use question_id as vector ID
        ids.append(str(row["question_id"]))

    return documents, ids


logger.info("Loading processed documents")
documents, ids = load_documents(DATA_PATH)

logger.info(f"Creating Chroma vectorstore ({len(documents)} vectors)")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    ids=ids,
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR
)

logger.info("Vectorstore successfully built")
