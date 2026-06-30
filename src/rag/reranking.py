from sentence_transformers import CrossEncoder

MODELE_RERANKING = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank_documents(
    requete: str,
    candidats: list[dict],
    cross_encoder: CrossEncoder,
    top_k_final: int = 3,
) -> list[dict]:
    if not candidats:
        return []
    paires = [[requete, c["text"]] for c in candidats]
    scores = cross_encoder.predict(paires)
    for c, s in zip(candidats, scores, strict=True):
        c["score_reranking"] = float(s)
    candidats_tries = sorted(
        candidats, key=lambda c: c["score_reranking"], reverse=True
    )
    return candidats_tries[:top_k_final]


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    from src.rag.base_connaissance import MODELE_EMBEDDING, build_base
    from src.rag.recherche import find_documents

    collection = build_base()
    modele = SentenceTransformer(MODELE_EMBEDDING)
    reranker = CrossEncoder(MODELE_RERANKING)

    requete = "Je veux retourner un article"
    # top_k élargi pour donner du grain à moudre au reranker
    candidats = find_documents(requete, collection, modele, top_k=8)
    print(f"Avant reranking ({len(candidats)} candidats) :")
    for c in candidats:
        print(f"  [{c['id']}] bi={c['score']:.3f} :: {c['text'][:55]}...")

    tries = rerank_documents(requete, candidats, reranker, top_k_final=3)
    print(f"\nAprès reranking (top {len(tries)}) :")
    for c in tries:
        print(f"  [{c['id']}] rerank={c['score_reranking']:.3f} :: {c['text'][:55]}...")
