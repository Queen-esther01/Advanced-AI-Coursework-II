from data_loader import load_intentions, load_time_date_sentences, load_stations
from nlp_utils import build_time_date_data
from intent_handlers import init_intent_handlers, check_intention_by_keyword, date_time_response, handle_ticket_intent


def main():
    # Load all data
    intentions = load_intentions()
    stations = load_stations()
    print("First 10 station names from CSV:", stations['NAME'].head(10).tolist())
    time_sentences, date_sentences = load_time_date_sentences()
    labels, sentences = build_time_date_data(time_sentences, date_sentences)

    # Initialise intent handlers with the loaded data
    init_intent_handlers(intentions, labels, sentences)

    print("BOT: Hi there! How can I help you?\n(If you want to exit, just type bye!)")

    flag = True
    while flag:
        user_input = input("You: ")

        # Check for goodbye
        intention = check_intention_by_keyword(user_input)
        if intention == 'goodbye':
            flag = False
            continue
        if intention is not None:
            continue

        # Ticket intent (Task 1)
        if handle_ticket_intent(user_input):
            continue

        # 4) Time / date
        if date_time_response(user_input):
            continue

        # 5) Fallback
        print("BOT: Sorry, I don't understand that. Please rephrase.")

if __name__ == '__bot_main__':
    main()