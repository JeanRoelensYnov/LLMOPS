from transformers import AutoTokenizer, PreTrainedTokenizerBase

NOM_MODELE = "Qwen/Qwen2.5-1.5B-Instruct"

_tokenizer: PreTrainedTokenizerBase | None = None


def get_tokenizer() -> PreTrainedTokenizerBase:
    """Charge (une seule fois) le tokenizer de Qwen et le renvoie.

    Pattern singleton paresseux : le tokenizer n'est chargé qu'au premier appel,
    puis réutilisé. Évite de le recharger à chaque requête (coûteux).
    """
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(NOM_MODELE)
    return _tokenizer
