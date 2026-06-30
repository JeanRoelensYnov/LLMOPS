import pytest


@pytest.fixture(scope="module")
def modele_embedding():
    from sentence_transformers import SentenceTransformer

    from src.rag.base_connaissance import MODELE_EMBEDDING

    return SentenceTransformer(MODELE_EMBEDDING)


@pytest.fixture(scope="module")
def cross_encoder():
    from sentence_transformers import CrossEncoder

    from src.rag.reranking import MODELE_RERANKING

    return CrossEncoder(MODELE_RERANKING)


@pytest.fixture
def collection_faq(modele_embedding):
    """chromadb.Client() est partagé dans le process -> nom UNIQUE par test (uuid)."""
    import uuid

    import chromadb

    from src.rag.base_connaissance import load_faq, populate_collection

    client = chromadb.Client()
    collection = client.create_collection(
        f"test_faq_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"}
    )
    populate_collection(collection, load_faq(), modele_embedding)
    return collection


def test_recherche_pertinence(collection_faq, modele_embedding):
    from src.rag.recherche import find_documents

    res = find_documents(
        "Comment retourner un article ?", collection_faq, modele_embedding
    )
    assert res
    assert res[0]["id"] == "faq-01"


def test_recherche_structure(collection_faq, modele_embedding):
    from src.rag.recherche import find_documents

    res = find_documents("remboursement", collection_faq, modele_embedding)
    premier = res[0]
    for cle in ("id", "text", "distance", "metadatas", "score"):
        assert cle in premier


def test_recherche_hors_domaine(collection_faq, modele_embedding):
    from src.rag.recherche import find_documents

    res = find_documents(
        "Quelle est la capitale de l'Australie ?",
        collection_faq,
        modele_embedding,
        seuil=0.9,
    )
    assert res == []


def test_rerank_respecte_top_k(collection_faq, modele_embedding, cross_encoder):
    from src.rag.recherche import find_documents
    from src.rag.reranking import rerank_documents

    candidats = find_documents(
        "retour article", collection_faq, modele_embedding, top_k=8
    )
    tries = rerank_documents("retour article", candidats, cross_encoder, top_k_final=3)
    assert len(tries) <= 3


def test_rerank_ajoute_score_et_trie(collection_faq, modele_embedding, cross_encoder):
    from src.rag.recherche import find_documents
    from src.rag.reranking import rerank_documents

    candidats = find_documents(
        "retour article", collection_faq, modele_embedding, top_k=8
    )
    tries = rerank_documents("retour article", candidats, cross_encoder, top_k_final=5)
    scores = [d["score_reranking"] for d in tries]
    assert scores == sorted(scores, reverse=True)


def test_rerank_liste_vide(cross_encoder):
    """rerank_documents([]) doit renvoyer [] sans appeler le modèle."""
    from src.rag.reranking import rerank_documents

    assert rerank_documents("requete", [], cross_encoder) == []


@pytest.mark.slow
def test_pipeline_structure(collection_faq, modele_embedding, cross_encoder):
    """generer_avec_rag renvoie un dict cohérent avec une réponse non vide."""
    from src.rag.pipeline import generer_avec_rag

    res = generer_avec_rag(
        "Comment retourner un article ?",
        collection_faq,
        modele_embedding,
        cross_encoder,
    )
    assert len(res["reponse"]) > 0
    assert isinstance(res["sources"], list)
    assert len(res["sources"]) > 0
