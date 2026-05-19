import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Input files
INTENTIONS_PATH = os.path.join(DATA_DIR, 'intentions.json')
SENTENCES_PATH = os.path.join(DATA_DIR, 'sentences.txt')
STATIONS_CSV_PATH = os.path.join(DATA_DIR, 'StationNameAndCode.csv')

# Global flag used to enable extra prompts in the final chatbot
final_chatbot = True