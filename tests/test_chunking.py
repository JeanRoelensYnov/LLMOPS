import pytest

from src.rag.chunking import split_document

# ---------------------------------------------------------------------------
# Tests de split_document (logique pure, pas de ChromaDB ni de modèle).
# ---------------------------------------------------------------------------


def test_split_nombre_de_chunks():
    """100 mots, taille 30, chevauchement 0 -> pas=30 -> 4 chunks (30,30,30,10)."""
    texte = " ".join(str(i) for i in range(100))
    chunks = split_document(texte, taille_chunk=30, chevauchement=0)
    assert len(chunks) == 4


def test_split_chevauchement():
    """Avec chevauchement, deux chunks consécutifs partagent des mots."""
    texte = " ".join(str(i) for i in range(100))
    chunks = split_document(texte, taille_chunk=30, chevauchement=10)
    # pas = 30 - 10 = 20. chunk 0 = mots[0:30], chunk 1 = mots[20:50].
    # Les mots d'indices 20..29 sont donc dans LES DEUX premiers chunks.
    mots_chunk0 = set(chunks[0].split())
    mots_chunk1 = set(chunks[1].split())
    assert mots_chunk0 & mots_chunk1


def test_split_texte_vide():
    assert split_document("") == []


def test_split_chevauchement_invalide():
    with pytest.raises(ValueError):
        split_document("a b c d e", taille_chunk=10, chevauchement=10)


def test_split_dernier_chunk_complet():
    """Le dernier chunk ne doit pas être perdu (texte plus court qu'un chunk)."""
    texte = "un deux trois"  # 3 mots, plus petit que taille_chunk
    chunks = split_document(texte, taille_chunk=30, chevauchement=0)
    assert len(chunks) == 1
    assert chunks[0] == texte


# ---------------------------------------------------------------------------
# Test d'index_long_documents avec une collection ChromaDB EN MÉMOIRE.
# ---------------------------------------------------------------------------


@pytest.fixture
def collection_memoire():
    """Collection ChromaDB éphémère (en RAM), recréée à chaque test."""
    import chromadb

    client = chromadb.Client()
    # nom unique pour éviter toute collision entre tests
    return client.create_collection("test_chunks", metadata={"hnsw:space": "cosine"})


@pytest.fixture(scope="module")
def modele_embedding():
    """Modèle d'embedding chargé une seule fois pour tout le module (coûteux)."""
    from sentence_transformers import SentenceTransformer

    from src.rag.base_connaissance import MODELE_EMBEDDING

    return SentenceTransformer(MODELE_EMBEDDING)


def test_index_long_documents(collection_memoire, modele_embedding, tmp_path):
    """Indexe 1 doc long et vérifie que les chunks portent bien l'id_parent."""
    import json

    # On fabrique un petit JSONL temporaire (tmp_path = dossier temp fourni par pytest).
    chemin = tmp_path / "docs.jsonl"
    doc = {
        "id": "doc-test",
        "titre": "T",
        "texte": " ".join(str(i) for i in range(100)),
    }
    chemin.write_text(json.dumps(doc) + "\n", encoding="utf-8")

    from src.rag.chunking import index_long_documents

    index_long_documents(
        chemin,
        collection_memoire,
        modele_embedding,
        taille_chunk=30,
        chevauchement=0,
    )
    # 100 mots chunks 30 [30, 30, 30, 10]
    assert collection_memoire.count() == 4
    res = collection_memoire.get(include=["metadatas"])
    assert all(doc["id_parent"] == "doc-test" for doc in res["metadatas"])
