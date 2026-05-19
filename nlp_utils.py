import spacy
import spacy.cli

import warnings
warnings.filterwarnings('ignore', message=".*word vectors.*")
warnings.filterwarnings('ignore', category=UserWarning, module='spacy')


try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

def lemmatize_and_clean(text):
    doc = nlp(text.lower())
    out = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            out.append(token.lemma_)
    return ' '.join(out)

def build_time_date_data(time_sentences, date_sentences):
    labels = []
    sentences = []
    doc = nlp(time_sentences)
    for sent in doc.sents:
        labels.append('time')
        sentences.append(sent.text.lower().strip())
    doc = nlp(date_sentences)
    for sent in doc.sents:
        labels.append('date')
        sentences.append(sent.text.lower().strip())
    return labels, sentences