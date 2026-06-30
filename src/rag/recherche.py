from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

SEUIL_SIMILARITE = 0.3


def find_documents(
    requete: str,
    collection: Collection,
    modele_embedding: SentenceTransformer,
    top_k: int = 3,
    seuil: float = SEUIL_SIMILARITE,
) -> list[dict]:
    vecteur = modele_embedding.encode([requete], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=vecteur, n_results=top_k)
    ids = res["ids"][0]
    documents = res["documents"]
    distances = res["distances"]
    metadata = res["metadatas"]
    if documents is None or distances is None or metadata is None:
        return []

    similar_embedding = []
    for id_, texte, distance, meta in zip(ids, documents[0], distances[0], metadata[0]):
        similarite = 1 - distance
        if similarite >= seuil:
            similar_embedding.append(
                {
                    "id": id_,
                    "text": texte,
                    "distance": distance,
                    "metadatas": meta,
                    "score": similarite,
                }
            )
    return similar_embedding


if __name__ == "__main__":
    # Petit test manuel rapide
    from src.rag.base_connaissance import MODELE_EMBEDDING, build_base

    collection = build_base()
    modele = SentenceTransformer(MODELE_EMBEDDING)

    for requete in ["Je veux retourner un article", "Quelle est la météo demain ?"]:
        print(f"\nRequête : {requete!r}")
        for doc in find_documents(requete, collection, modele):
            print(f"  [{doc['id']}] score={doc['score']:.3f} :: {doc['text'][:60]}...")
