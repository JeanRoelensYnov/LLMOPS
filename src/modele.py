import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from src.prompts import PROMPT_SYSTEME
from src.tokeniseur import NOM_MODELE, get_tokenizer

_modele: PreTrainedModel | None = None


def get_modele() -> PreTrainedModel:
    """
    singleton Qwen.
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
    tokenizer = get_tokenizer()
    modele = get_modele()
    messages = [
        {"role": "system", "content": prompt_systeme},
        {"role": "user", "content": contenu_utilisateur},
    ]
    texte = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(texte, return_tensors="pt")

    with torch.no_grad():
        sortie = modele.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

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
