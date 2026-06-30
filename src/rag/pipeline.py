from chromadb.api.models.Collection import Collection
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.modele import generer
from src.prompts import construire_prompt
from src.rag.recherche import SEUIL_SIMILARITE, find_documents
from src.rag.reranking import rerank_documents


def generer_avec_rag(
    requete: str,
    collection: Collection,
    modele_embedding: SentenceTransformer,
    cross_encoder: CrossEncoder,
    top_k_recherche: int = 10,
    top_k_final: int = 3,
    seuil: float = SEUIL_SIMILARITE,
    max_new_tokens: int = 256,
) -> dict:
    candidats = find_documents(
        requete, collection, modele_embedding, top_k=top_k_recherche, seuil=seuil
    )

    if not candidats:
        return {
            "reponse": "Désolé, je ne dispose pas d'information pour répondre à cette question.",
            "sources": [],
            "documents": [],
        }

    documents = rerank_documents(requete, candidats, cross_encoder, top_k_final)

    prompt = construire_prompt(requete, documents)

    reponse = generer(prompt, max_new_tokens=max_new_tokens)

    return {
        "reponse": reponse,
        "sources": [d["id"] for d in documents],
        "documents": documents,
    }


if __name__ == "__main__":
    from src.rag.base_connaissance import MODELE_EMBEDDING, build_base
    from src.rag.reranking import MODELE_RERANKING

    collection = build_base()
    modele = SentenceTransformer(MODELE_EMBEDDING)
    reranker = CrossEncoder(MODELE_RERANKING)

    for requete in [
        "Comment retourner un article ?",
        "Quelle est la météo demain ?",  # hors-domaine -> doit refuser
    ]:
        print(f"\n=== Requête : {requete!r} ===")
        res = generer_avec_rag(requete, collection, modele, reranker)
        print("Réponse :", res["reponse"])
        print("Sources :", res["sources"])
