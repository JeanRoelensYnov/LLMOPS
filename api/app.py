import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag.pipeline import generer_avec_rag

logger = logging.getLogger(__name__)
logging.basicConfig(filename="app.log", encoding="utf-8", level=logging.DEBUG)

ressources: dict = {}
CHEMIN_LOG_REQUETES = Path("logs/requests.jsonl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.debug("Loading necessary ressources...")
    from sentence_transformers import CrossEncoder, SentenceTransformer

    from src.modele import get_modele
    from src.rag.base_connaissance import MODELE_EMBEDDING, build_base
    from src.rag.reranking import MODELE_RERANKING
    from src.tokeniseur import get_tokenizer

    ressources["collection"] = build_base()
    ressources["modele_embedding"] = SentenceTransformer(MODELE_EMBEDDING)
    ressources["cross_encoder"] = CrossEncoder(MODELE_RERANKING)

    # Pre load both tokenizer and model
    get_tokenizer()
    get_modele()
    logger.debug("Ressources loaded")
    yield
    logger.debug("Stopping application, flushing ressources...")
    ressources.clear()
    logger.debug("Ressources cleared, bye.")


app = FastAPI(
    title="Assistant Service Client RAG",
    description="API d'un assistant de service client basé sur RAG (Qwen2.5-1.5B).",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "modeles_charges": "collection" in ressources.keys()}


class GenerateRequest(BaseModel):
    question: str = Field(
        # https://stackoverflow.com/questions/772124/what-does-the-ellipsis-object-do
        ...,
        min_length=1,
        description="La question de l'utilisateur.",
    )
    max_new_tokens: int = Field(
        256,
        ge=1,
        le=500,
        description="Nombre max de tokens générés.",
    )


class GenerateResponse(BaseModel):
    reponse: str
    sources: list[str]


def journaliser_requete(question, reponse, sources, duree_s, max_new_tokens):
    CHEMIN_LOG_REQUETES.parent.mkdir(parents=True, exist_ok=True)
    ligne = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "reponse": reponse,
        "sources": sources,
        "nb_sources": len(sources),
        "duree_s": round(duree_s, 3),
        "max_new_tokens": max_new_tokens,
    }
    with open(CHEMIN_LOG_REQUETES, "a", encoding="utf-8") as f:
        f.write(json.dumps(ligne, ensure_ascii=False) + "\n")


@app.post("/generate", response_model=GenerateResponse)
def generate(requete: GenerateRequest) -> GenerateResponse:

    if "collection" not in ressources.keys():
        raise HTTPException(
            status_code=503,
            detail="Modèles non chargés, le service n'est pas encore prêt.",
        )
    try:
        start = perf_counter()
        resultat = generer_avec_rag(
            requete.question,
            ressources["collection"],
            ressources["modele_embedding"],
            ressources["cross_encoder"],
            max_new_tokens=requete.max_new_tokens,
        )
        duree = perf_counter() - start
    except Exception as e:
        logger.exception(f"Error while generating : {e}")
        raise HTTPException(
            status_code=500, detail="Internal error while generating."
        ) from e
    journaliser_requete(
        requete.question,
        resultat["reponse"],
        resultat["sources"],
        duree,
        requete.max_new_tokens,
    )
    return GenerateResponse(reponse=resultat["reponse"], sources=resultat["sources"])
