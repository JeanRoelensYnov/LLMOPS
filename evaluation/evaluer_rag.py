import json
import sys
from pathlib import Path

from rouge_score import rouge_scorer
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.rag.base_connaissance import MODELE_EMBEDDING, build_base
from src.rag.recherche import find_documents
from src.rag.reranking import MODELE_RERANKING

CHEMIN_JEU = Path("evaluation/jeu_evaluation.jsonl")


def charger_jeu(chemin: Path = CHEMIN_JEU) -> list[dict]:
    """Charge le jeu d'évaluation (une ligne JSON par cas)."""
    cas = []
    with open(chemin, "r", encoding="utf-8") as f:
        for ligne in f:
            cas.append(json.loads(ligne))
    return cas


def ids_recuperes(
    question: str,
    collection,
    modele: SentenceTransformer,
    top_k: int = 5,
) -> list[str]:
    """Renvoie la liste ORDONNÉE des ids récupérés pour une question.

    On met seuil=0.0 pour NE PAS filtrer : en évaluation de recherche, on veut le
    classement complet du top_k (sinon un doc pertinent un peu loin serait coupé et
    fausserait Recall@5 / MRR).
    """
    docs = find_documents(question, collection, modele, top_k=top_k, seuil=0.0)
    return [d["id"] for d in docs]


def evaluer_recherche(
    jeu: list[dict],
    collection,
    modele: SentenceTransformer,
) -> dict:
    ks = (1, 3, 5)
    recalls: dict[int, list[float]] = {k: [] for k in ks}
    reciprocal_ranks = []
    for cas in jeu:
        pertinents = set(cas["docs_pertinents"])
        recuperes = ids_recuperes(cas["question"], collection, modele)
        for k in ks:
            trouves = pertinents & set(recuperes[:k])
            recalls[k].append(len(trouves) / len(pertinents))
        rr = 0.0  # Reciprocal Rank
        for rang, id_ in enumerate(recuperes, start=1):
            if id_ in pertinents:
                rr = 1.0 / rang
                break
        reciprocal_ranks.append(rr)

    resultats = {f"recall@{k}": sum(recalls[k]) / len(jeu) for k in ks}
    resultats["mrr"] = sum(reciprocal_ranks) / len(jeu)
    resultats["n_cas"] = len(jeu)
    return resultats


_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

PROMPT_SYSTEME_SANS_RAG = (
    "Tu es un assistant de service client. Réponds à la question de l'utilisateur "
    "en français, de manière concise."
)


def rouge_l(reference: str, hypothese: str) -> float:
    return _scorer.score(reference, hypothese)["rougeL"].fmeasure


def evaluer_generation(
    jeu: list[dict],
    collection,
    modele: SentenceTransformer,
    cross_encoder: CrossEncoder,
    max_new_tokens: int = 128,
) -> dict:
    from src.modele import generer
    from src.rag.pipeline import generer_avec_rag

    l_score_rag = []
    l_score_wo_rag = []
    for cas in jeu:
        ref = cas["reponse_reference"]
        # With RAG
        res = generer_avec_rag(
            cas["question"],
            collection,
            modele,
            cross_encoder,
            max_new_tokens=max_new_tokens,
        )
        score_rag = rouge_l(ref, res["reponse"])

        # Without RAG
        rep = generer(
            cas["question"],
            prompt_systeme=PROMPT_SYSTEME_SANS_RAG,
            max_new_tokens=max_new_tokens,
        )
        score_sans = rouge_l(ref, rep)

        l_score_rag.append(score_rag)
        l_score_wo_rag.append(score_sans)
    mean_score_rag = sum(l_score_rag) / len(l_score_rag)
    mean_score_wo_rag = sum(l_score_wo_rag) / len(l_score_wo_rag)
    return {
        "rougeL_rag": mean_score_rag,
        "rougeL_sans_rag": mean_score_wo_rag,
        "gain_absolu": mean_score_rag - mean_score_wo_rag,
    }


if __name__ == "__main__":
    jeu = charger_jeu()
    collection = build_base()
    modele = SentenceTransformer(MODELE_EMBEDDING)
    cross_encoder = CrossEncoder(MODELE_RERANKING)
    resultats_recherche = evaluer_recherche(jeu, collection, modele)
    resultats_generation = evaluer_generation(jeu, collection, modele, cross_encoder)
    rapport = {"recherche": resultats_recherche, "generation": resultats_generation}
    with open("evaluation/rapport_eval.json", "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2)

    alertes = []
    if resultats_recherche["recall@3"] < 0.75:
        alertes.append(f"Recall@3 trop bas : {resultats_recherche['recall@3']:.2f}")
    if resultats_generation["rougeL_rag"] < 0.30:
        alertes.append(
            f"ROUGE-L RAG trop bas : {resultats_generation['rougeL_rag']:.2f}"
        )

    if alertes:
        print("ECHEC:", *alertes, sep="\n ")
        sys.exit(1)
    print("OK: tous les seuils respectés.")
