import json
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Metadata
from sentence_transformers import SentenceTransformer

CHEMIN_FAQ = Path("data/faq_service_client.jsonl")
CHEMIN_CHROMA = Path("data/chroma_db")
NOM_COLLECTION = "faq_service_client"
MODELE_EMBEDDING = "all-MiniLM-L6-v2"


def load_faq(chemin: Path = CHEMIN_FAQ) -> list[dict[str, str]]:
    lines = []
    with open(chemin, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def get_collection(chemin_chroma: Path = CHEMIN_CHROMA) -> Collection:
    client = chromadb.PersistentClient(path=chemin_chroma)

    collection = client.get_or_create_collection(
        name=NOM_COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    return collection


def populate_collection(
    collection: Collection,
    faq: list[dict],
    modele: SentenceTransformer,
) -> None:
    if collection.count() > 0:
        return  # Already populated
    ids = [entree["id"] for entree in faq]
    questions = [entree["question"] for entree in faq]
    documents = [entree["reponse"] for entree in faq]
    metadata: list[Metadata] = [
        {"id": e["id"], "categorie": e["categorie"], "question": e["question"]}
        for e in faq
    ]

    to_add = modele.encode(questions, normalize_embeddings=True).tolist()

    collection.add(ids=ids, embeddings=to_add, documents=documents, metadatas=metadata)
    return


def build_base() -> Collection:
    faq = load_faq()
    collection = get_collection()
    modele = SentenceTransformer(MODELE_EMBEDDING)
    populate_collection(collection, faq, modele)
    return collection


if __name__ == "__main__":
    collection = build_base()
    print(f"Collection '{collection.name}' : {collection.count()} documents indexés.")
