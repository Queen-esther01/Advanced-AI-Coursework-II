import warnings

import spacy

warnings.filterwarnings("ignore", message=".*word vectors.*")
warnings.filterwarnings("ignore", category=UserWarning, module="spacy")

_MODEL_CANDIDATES = ("en_core_web_sm", "en_core_web_md")
_nlp = None


def _load_spacy_model():
    last_error: OSError | None = None
    for name in _MODEL_CANDIDATES:
        try:
            return spacy.load(name)
        except OSError as exc:
            last_error = exc
    raise OSError(
        "No spaCy English model found. Install one with:\n"
        "  uv pip install "
        "https://github.com/explosion/spacy-models/releases/download/"
        "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
    ) from last_error


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = _load_spacy_model()
    return _nlp


class _NlpProxy:
    def __call__(self, text):
        return get_nlp()(text)

    def __getattr__(self, name):
        return getattr(get_nlp(), name)


nlp = _NlpProxy()


def lemmatize_and_clean(text):
    doc = nlp(text.lower())
    out = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            out.append(token.lemma_)
    return " ".join(out)


def build_time_date_data(time_sentences, date_sentences):
    labels = []
    sentences = []
    doc = nlp(time_sentences)
    for sent in doc.sents:
        labels.append("time")
        sentences.append(sent.text.lower().strip())
    doc = nlp(date_sentences)
    for sent in doc.sents:
        labels.append("date")
        sentences.append(sent.text.lower().strip())
    return labels, sentences
