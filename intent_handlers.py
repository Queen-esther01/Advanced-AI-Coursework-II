import random
from datetime import datetime
from config import final_chatbot
from nlp_utils import nlp, lemmatize_and_clean
from ticket_finder import ticket_response

# Load intentions once
intentions = None   # will be set by main after loading data
time_date_labels = None
time_date_sentences = None
universities = None

def init_intent_handlers(intentions_dict, labels, sentences):
    global intentions, time_date_labels, time_date_sentences, universities
    intentions = intentions_dict
    time_date_labels = labels
    time_date_sentences = sentences

# Keyword matching (greeting, goodbye, thanks)
def check_intention_by_keyword(sentence):
    for word in sentence.split():
        for intent_type in intentions:
            if word.lower() in intentions[intent_type]["patterns"]:
                print("BOT: " + random.choice(intentions[intent_type]["responses"]))
                if intent_type == 'greeting' and final_chatbot:
                    pass
                return intent_type
    return None


# Time/date response (similarity‑based)

def date_time_response(user_input):
    cleaned_user = lemmatize_and_clean(user_input)
    doc1 = nlp(cleaned_user)
    similarities = {}
    for idx, sent in enumerate(time_date_sentences):
        cleaned_sent = lemmatize_and_clean(sent)
        doc2 = nlp(cleaned_sent)
        sim = doc1.similarity(doc2)
        similarities[idx] = sim
    max_idx = max(similarities, key=similarities.get)
    if similarities[max_idx] > 0.75:
        if time_date_labels[max_idx] == 'time':
            print("BOT: It's", datetime.now().strftime('%H:%M:%S'))
        else:
            print("BOT: It's", datetime.now().strftime('%Y-%m-%d'))
        if final_chatbot:
            print("BOT: Now can you tell me where you want to go? (Hint: I need a ticket...)")
        return True
    return False


# Ticket intent gatekeeper (calls  state machine)

def handle_ticket_intent(user_input):
    return ticket_response(user_input)