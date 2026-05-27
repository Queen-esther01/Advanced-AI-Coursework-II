
import json
import pandas as pd
from config import INTENTIONS_PATH, SENTENCES_PATH, STATIONS_CSV_PATH

def load_intentions():
    with open(INTENTIONS_PATH, 'r') as f:
        return json.load(f)

def load_time_date_sentences():
    time_sentences = ''
    date_sentences = ''
    with open(SENTENCES_PATH, 'r') as file:
        for line in file:
            parts = line.strip().split(' | ')
            if parts[0] == 'time':
                time_sentences += ' ' + parts[1]
            elif parts[0] == 'date':
                date_sentences += ' ' + parts[1]
    return time_sentences.strip(), date_sentences.strip()

def load_stations():
    df = pd.read_csv(STATIONS_CSV_PATH)
    return df