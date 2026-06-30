import json
from pathlib import Path

from chromadb.api.models.Collection import Collection
from chromadb.api.types import Metadata
from sentence_transformers import SentenceTransformer


def split_document(
    texte: str,
    taille_chunk: int = 200,
    chevauchement: int = 30,
) -> list[str]:
    splitted = texte.split()
    if not splitted:
        return []

    pas = taille_chunk - chevauchement
    if pas <= 0:
        raise ValueError("Chevauchement doit être < taille chunk")

    chunks = []
    for i in range(0, len(splitted), pas):
        windows = splitted[i : i + taille_chunk]
        chunks.append(" ".join(windows))
        if i + taille_chunk >= len(splitted):
            break
    return chunks


def index_long_documents(
    chemin_jsonl: Path,
    collection: Collection,
    modele_embedding: SentenceTransformer,
    taille_chunk: int = 200,
    chevauchement: int = 30,
) -> None:
    if collection.count() > 0:
        return  # garde anti doubloon
    documents_long = []
    with open(chemin_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            documents_long.append(json.loads(line))

    ids: list[str] = []
    chunks_tous: list[str] = []
    metadatas: list[Metadata] = []

    for doc in documents_long:
        chunks = split_document(doc["texte"], taille_chunk, chevauchement)
        for numero, chunk in enumerate(chunks):
            ids.append(f"{doc['id']}__chunk_{numero}")
            chunks_tous.append(chunk)
            metadatas.append({"id_parent": doc["id"], "numero_chunk": numero})
    if not chunks_tous:
        return

    embeddings = modele_embedding.encode(
        chunks_tous, normalize_embeddings=True
    ).tolist()
    collection.add(
        ids=ids, embeddings=embeddings, documents=chunks_tous, metadatas=metadatas
    )


if __name__ == "__main__":
    # Test rapide de split_document sans toucher à ChromaDB
    texte = " ".join(str(i) for i in range(100))  # "0 1 2 ... 99" = 100 "mots"
    chunks = split_document(texte, taille_chunk=30, chevauchement=0)
    print(f"{len(chunks)} chunks (attendu : 4)")
    for i, c in enumerate(chunks):
        mots = c.split()
        print(f"  chunk {i}: {len(mots)} mots  [{mots[0]}..{mots[-1]}]")
