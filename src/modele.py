import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from src.prompts import PROMPT_SYSTEME
from src.tokeniseur import NOM_MODELE, get_tokenizer

_modele: PreTrainedModel | None = None


def get_modele() -> PreTrainedModel:
    """Charge (une seule fois) le modèle Qwen et le renvoie.

    Singleton paresseux comme le tokenizer. torch_dtype=float32 car on est en CPU
    (pas de float16/bf16 efficace sans GPU). .eval() désactive dropout & co (inférence).
    """
    global _modele
    if _modele is None:
        _modele = AutoModelForCausalLM.from_pretrained(
            NOM_MODELE,
            dtype=torch.float32,
        )
        _modele.eval()
    return _modele


def generer(
    contenu_utilisateur: str,
    prompt_systeme: str = PROMPT_SYSTEME,
    max_new_tokens: int = 256,
) -> str:
    """Génère une réponse à partir d'un message système + un message utilisateur.

    `contenu_utilisateur` = la sortie de construire_prompt() (contexte + question).
    """
    tokenizer = get_tokenizer()
    modele = get_modele()

    # 1. Structurer en rôles system/user, puis appliquer le chat template de Qwen.
    #    add_generation_prompt=True ajoute le marqueur qui invite le modèle à répondre.
    messages = [
        {"role": "system", "content": prompt_systeme},
        {"role": "user", "content": contenu_utilisateur},
    ]
    texte = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 2. Tokeniser le prompt complet en tenseurs.
    inputs = tokenizer(texte, return_tensors="pt")

    # 3. Génération déterministe (greedy : do_sample=False) -> reproductible pour l'éval.
    with torch.no_grad():
        sortie = modele.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # 4. La sortie contient le prompt + la réponse ; on ne garde que les tokens générés.
    nb_tokens_prompt = inputs["input_ids"].shape[1]
    tokens_generes = sortie[0][nb_tokens_prompt:]
    return tokenizer.decode(tokens_generes, skip_special_tokens=True).strip()


if __name__ == "__main__":
    from src.prompts import construire_prompt

    docs = [
        {"text": "Vous pouvez retourner tout article dans un délai de 30 jours."},
        {"text": "Le remboursement est effectué sous 5 à 10 jours ouvrés."},
    ]
    prompt = construire_prompt("Comment retourner un article ?", docs)
    print("Génération en cours (CPU, patiente quelques secondes)...\n")
    print(generer(prompt))
