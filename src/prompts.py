PROMPT_SYSTEME = (
    "Tu es un assistant de service client. Réponds à la question de l'utilisateur "
    "en t'appuyant UNIQUEMENT sur le contexte fourni ci-dessous. "
    "Si le contexte ne contient pas l'information nécessaire, dis clairement que tu "
    "ne disposes pas de cette information. Ne fabrique pas de réponse. "
    "Réponds en français, de manière concise et polie."
)


def construire_prompt(requete: str, documents: list[dict]) -> str:
    if not documents:
        contexte = "(aucun)"
    else:
        lignes = [f"[{i}] {doc['text']}" for i, doc in enumerate(documents, start=1)]
        contexte = "\n".join(lignes)
    return f"Contexte :\n{contexte}\n\nQuestion : {requete}\n\nRéponse :"


if __name__ == "__main__":
    docs = [
        {"text": "Vous pouvez retourner tout article dans un délai de 30 jours."},
        {"text": "Le remboursement est effectué sous 5 à 10 jours ouvrés."},
    ]
    print("=== Avec contexte ===")
    print(construire_prompt("Comment retourner un article ?", docs))
    print("\n=== Sans contexte ===")
    print(construire_prompt("Quelle est la météo ?", []))
