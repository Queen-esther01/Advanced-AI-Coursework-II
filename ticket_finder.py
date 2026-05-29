import re
from datetime import datetime, timedelta
from difflib import get_close_matches

from dateutil import parser

from data_loader import load_stations
from national_rail_api import get_ojp_client
from nlp_utils import nlp


def extract_station_from_sentence(text, context_hint=None):
    text_lower = text.lower()

    patterns = [
        r"to\s+([a-z\s]+?)(?:\s+on|\s+at|\s+for|\s+$|\.|$)",
        r"from\s+([a-z\s]+?)(?:\s+to|\s+$|\.|$)",
        r"going\s+to\s+([a-z\s]+?)(?:\s+on|\s+at|\s+for|\s+$|\.|$)",
        r"travelling\s+to\s+([a-z\s]+?)(?:\s+on|\s+at|\s+for|\s+$|\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            station_candidate = match.group(1).strip()
            station_candidate = re.sub(r"\s+(?:on|at|for)$", "", station_candidate)
            return station_candidate.title()

    # Then try spaCy NER (as fallback for simple "London" style inputs)
    doc = nlp(text_lower)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text.title()

    return None


def extract_date_from_sentence(text):
    # Simple regex for common date patterns
    date_pattern = r"\b(\d{1,2}(?:st|nd|rd|th)?\s+[a-z]+\s+\d{4})\b|\b(\d{4}-\d{2}-\d{2})\b|\b(\d{1,2}/\d{1,2}/\d{4})\b"
    match = re.search(date_pattern, text, re.IGNORECASE)
    if match:
        return match.group(0)
    return None


# Load stations
stations_df = load_stations()
station_names = stations_df["NAME"].tolist()


# Common station name overrides (for cases where fuzzy matching fails)
STATION_OVERRIDES = {
    "london waterloo": ("LONDON WATERLOO", "WAT"),
    "waterloo": ("LONDON WATERLOO", "WAT"),
    "london bridge": ("LONDON BRIDGE", "LBG"),
    "london victoria": ("LONDON VICTORIA", "VIC"),
    "london kings cross": ("LONDON KINGS CROSS", "KGX"),
    "london euston": ("LONDON EUSTON", "EUS"),
    "london paddington": ("LONDON PADDINGTON", "PAD"),
}


def get_station_details(station_name):
    print(f"DEBUG: get_station_details received: '{station_name}'")
    user_input_lower = station_name.lower().strip()
    print(f"DEBUG: lowercased: '{user_input_lower}'")

    if len(user_input_lower) < 3:
        return None, None

    # 0) Check common overrides first
    if user_input_lower in STATION_OVERRIDES:
        print(f"DEBUG: Override matched -> {STATION_OVERRIDES[user_input_lower]}")
        return STATION_OVERRIDES[user_input_lower]

    # 1) Fuzzy match against all stations
    lowercase_names = [name.lower() for name in station_names]
    matches = get_close_matches(user_input_lower, lowercase_names, n=1, cutoff=0.6)
    if matches:
        best = matches[0]
        idx = lowercase_names.index(best)
        official = station_names[idx]
        crs = stations_df[stations_df["NAME"] == official]["CRS"].values[0]
        print(f"DEBUG: Fuzzy matched -> {official} ({crs})")
        return official, crs

    # 2) Substring fallback (for longer inputs)
    if len(user_input_lower) >= 4:
        for i, name in enumerate(lowercase_names):
            if user_input_lower in name:
                official = station_names[i]
                crs = stations_df[stations_df["NAME"] == official]["CRS"].values[0]
                print(f"DEBUG: Substring matched -> {official} ({crs})")
                return official, crs

    return None, None


def parse_travel_date(date_string, default_hour=6):
    """
    Convert natural language date (including 'tomorrow', 'today', 'next Monday', etc.)
    into API-ready string with timezone offset.
    Returns None if parsing fails.
    """
    date_string = date_string.lower().strip()
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # 1) Direct relative keywords
    if date_string == "today":
        dt = today
    elif date_string == "tomorrow":
        dt = today + timedelta(days=1)
    elif date_string == "day after tomorrow":
        dt = today + timedelta(days=2)
    elif date_string == "next tomorrow":
        dt = today + timedelta(days=2)
    elif date_string == "next week":
        dt = today + timedelta(days=7)
    else:
        # 2) "next <weekday>"
        match = re.match(
            r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$",
            date_string,
        )
        if match:
            weekday_name = match.group(1)
            weekdays = [
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ]
            target_weekday = weekdays.index(weekday_name)
            days_ahead = (target_weekday - today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7  # next week, not today
            dt = today + timedelta(days=days_ahead)
        else:
            # 3) Try absolute date parsing (e.g., "15 july 2026")
            try:
                # Use a default datetime to avoid dateutil defaulting to current date
                dt = parser.parse(
                    date_string,
                    default=datetime.now().replace(
                        hour=0, minute=0, second=0, microsecond=0
                    ),
                )
            except Exception:
                # Also try removing ordinal suffixes (e.g., "15th" -> "15")
                cleaned = re.sub(r"(\d)(st|nd|rd|th)", r"\1", date_string)
                try:
                    dt = parser.parse(
                        cleaned,
                        default=datetime.now().replace(
                            hour=0, minute=0, second=0, microsecond=0
                        ),
                    )
                except Exception:
                    return None

    # Set time to default_hour
    dt = dt.replace(hour=default_hour, minute=0, second=0, microsecond=0)

    # Reject dates in the past (allow today only if the time is in the future? We'll keep simple: no past dates)
    if dt.date() < today.date():
        return None

    return dt.strftime("%Y-%m-%dT%H:%M:%S+01:00")


# validate station name
def validate_station(user_input):
    """
    Returns (official_name, crs_code) if recognised, else (None, None).
    """
    official, crs = get_station_details(user_input)
    if official:
        return official, crs
    else:
        print(
            "BOT: I'm not sure about that station. Please try again (e.g., 'London Waterloo')."
        )
        return None, None


# Ticket state machine
class TicketState:
    def __init__(self):
        self.station_choices = []  # list of (official_name, crs)
        self.choice_target = None  # 'origin' or 'destination'
        self.origin_name = None
        self.origin_crs = None
        self.dest_name = None
        self.dest_crs = None
        self.outbound_date_raw = None  # what the user typed
        self.outbound_date_api = None  # formatted for API
        self.return_date_raw = None
        self.return_date_api = None
        self.is_return = False
        self.temp_dest = None
        self.stage = "idle"  # idle, wait_origin, wait_destination, wait_outbound, ask_return, wait_return, search


state = TicketState()


def reset_ticket_state():
    global state
    state = TicketState()


def is_ticket_intent(user_input):
    user_lower = user_input.lower()
    # Keywords for ticket booking
    keywords = [
        "ticket",
        "cheapest",
        "cheap",
        "fare",
        "book",
        "buy",
        "journey",
        "travel",
    ]
    # Travel intent phrases
    phrases = ["want to go", "need to go", "travelling to", "going to"]
    if any(kw in user_lower for kw in keywords):
        return True
    if any(phrase in user_lower for phrase in phrases):
        return True
    return False


def process_ticket_input(user_input, ticket_state):
    from ticket_finder import ticket_response_streamlit

    reply = ticket_response_streamlit(user_input, ticket_state)
    if reply is None:
        reply = "I didn't understand that. Please try again."
    return reply


def search_national_rail_tickets(
    origin_crs, dest_crs, outward_datetime, is_return=False, inward_datetime=None
):
    """
    outward_datetime and inward_datetime must be in the format "YYYY-MM-DDTHH:MM:SS+01:00".
    Returns: (cheapest_price_in_pence, price_display_string, booking_link, notice_text)
    """
    if not outward_datetime:
        print("BOT: No valid outbound date provided.")
        return None, None, None, None

    print(f"DEBUG: origin CRS = {origin_crs}, dest CRS = {dest_crs}")
    print(f"DEBUG: outward_datetime = {outward_datetime}")
    if inward_datetime:
        print(f"DEBUG: inward_datetime = {inward_datetime}")

    if not origin_crs or not dest_crs:
        print("BOT: Sorry, I could not recognise the station names.")
        return None, None, None, None

    # Build the request dictionary
    request = {
        "origin": {"stationCRS": origin_crs},
        "destination": {"stationCRS": dest_crs},
        "realtimeEnquiry": "STANDARD",
        "outwardTime": {"departBy": outward_datetime},
        "directTrains": False,
        "fareRequestDetails": {
            "passengers": {"adult": 1, "child": 0},
            "fareClass": "ANY",
        },
    }
    if is_return and inward_datetime:
        request["inwardTime"] = {"departBy": inward_datetime}

    # Create the client and make the call
    try:
        client = get_ojp_client()
        response = client.service.RealtimeJourneyPlan(**request)
    except Exception as e:
        print(f"BOT: Something went wrong while contacting National Rail: {e}")
        return None, None, None, None

    if response.response != "Ok":
        print(f"BOT: National Rail returned an error: {response.responseDetails}")
        return None, None, None, None

    outward_journeys = response.outwardJourney
    if not outward_journeys:
        print("BOT: No journeys found for your dates. Please try a different time.")
        return None, None, None, None

    # --- Collect service bulletins (notices) ---
    notices = []
    for journey in outward_journeys:
        if hasattr(journey, "serviceBulletins") and journey.serviceBulletins:
            for bulletin in journey.serviceBulletins:
                title = getattr(bulletin, "title", "")
                description = getattr(bulletin, "description", "")
                if title and description:
                    notices.append(f"{title}: {description}")
                elif description:
                    notices.append(description)
                elif title:
                    notices.append(title)
    # Remove duplicates while preserving order
    unique_notices = []
    for n in notices:
        if n not in unique_notices:
            unique_notices.append(n)
    notice_text = "\n".join(unique_notices) if unique_notices else None

    # --- Find cheapest fare ---
    cheapest_price = None
    cheapest_desc = None
    for journey in outward_journeys:
        if hasattr(journey, "fare") and journey.fare:
            for fare in journey.fare:
                if not hasattr(fare, "totalPrice"):
                    continue
                price_pence = fare.totalPrice
                if cheapest_price is None or price_pence < cheapest_price:
                    cheapest_price = price_pence
                    cheapest_desc = fare.description

    if cheapest_price is None:
        print(
            "BOT: No fare information was returned. The service may be temporarily unavailable."
        )
        return None, None, None, notice_text

    price_pounds = cheapest_price / 100.0

    booking_link = build_national_rail_link(
        origin_crs, dest_crs, outward_datetime, is_return, inward_datetime
    )
    return (
        cheapest_price,
        f"£{price_pounds:.2f} ({cheapest_desc})",
        booking_link,
        notice_text,
    )


def search_and_present_tickets():
    print(
        f"\nBOT: Looking for the cheapest ticket from {state.origin_name} to {state.dest_name}..."
    )
    if state.is_return:
        print(
            f"     Outbound: {state.outbound_date_raw}, Return: {state.return_date_raw}"
        )
    else:
        print(f"     Outbound: {state.outbound_date_raw}")

    price_pence, price_display, link = search_national_rail_tickets(
        state.origin_crs,
        state.dest_crs,
        state.outbound_date_api,
        state.is_return,
        state.return_date_api if state.is_return else None,
    )

    if price_pence is None:
        print("BOT: Please try again with different stations or dates.")
        reset_ticket_state()
        return True

    print(f"BOT: The cheapest ticket I found is {price_display}.")
    print(f"BOT: You can book it at: {link}")
    print(
        "BOT: Would you like to book another ticket? (type 'reset' to start over, or 'bye' to exit)"
    )
    reset_ticket_state()
    return True


def build_national_rail_link(
    origin_crs,
    dest_crs,
    outward_datetime_api,
    is_return=False,
    return_datetime_api=None,
):
    """
    Build a realistic National Rail journey planner URL exactly as the website does.
    """
    if not outward_datetime_api:
        return "https://www.nationalrail.co.uk/"
    from datetime import datetime

    # Parse outward datetime
    outward_dt = datetime.fromisoformat(outward_datetime_api.replace("+01:00", ""))
    leaving_date = outward_dt.strftime("%d%m%y")
    leaving_hour = outward_dt.strftime("%H")
    leaving_min = outward_dt.strftime("%M")

    # Building the base URL parameters as a list to so it remains consostent
    params = [
        f"origin={origin_crs}",
        f"destination={dest_crs}",
        f"leavingType=departing",
        f"leavingDate={leaving_date}",
        f"leavingHour={leaving_hour}",
        f"leavingMin={leaving_min}",
        f"adults=1",
        f"extraTime=0",
    ]

    # Handle return journey parameters
    if is_return and return_datetime_api:
        # Insert the "type=return" at the beginning
        params.insert(0, "type=return")

        # Parse return datetime and add return parameters
        return_dt = datetime.fromisoformat(return_datetime_api.replace("+01:00", ""))
        return_date = return_dt.strftime("%d%m%y")
        return_hour = return_dt.strftime("%H")
        return_min = return_dt.strftime("%M")

        params.append(f"returnType=departing")
        params.append(f"returnDate={return_date}")
        params.append(f"returnHour={return_hour}")
        params.append(f"returnMin={return_min}")
    else:
        # For a single journey, insert "type=single" at the beginning
        params.insert(0, "type=single")

    # Build the final URL
    url = "https://www.nationalrail.co.uk/journey-planner/?" + "&".join(params) + "#O"
    return url


def ticket_response(user_input):
    global state

    # Handle post-ticket commands
    if user_input.lower() == "reset":
        reset_ticket_state()
        print("BOT: Conversation reset. How can I help you?")
        return True
    if state.stage == "idle" and user_input.lower() == "yes":
        # User wants another ticket – restart the process
        reset_ticket_state()
        print("BOT: Great! Where would you like to travel from?")
        state.stage = "wait_origin"
        return True

    # Start new ticket conversation
    if state.stage == "idle":
        if is_ticket_intent(user_input):
            print("BOT: Sure! Where are you travelling from?")
            state.stage = "wait_origin"
            return True

        # Also check for travel phrases like "I want to go to X"
        travel_match = re.search(
            r"(?:want to go to|need to go to|travelling to|going to)\s+([a-z\s]+)$",
            user_input.lower(),
        )
        if travel_match:
            # User expressed destination only – origin is still needed
            dest_candidate = travel_match.group(1).strip().title()

            print(
                f"BOT: I see you want to go to {dest_candidate}. Where will you be travelling from?"
            )
            # Store destination temporarily and await origin
            state.temp_dest = dest_candidate
            state.stage = "wait_origin"
            return True

    # Wait for origin
    if state.stage == "wait_origin":
        extracted = extract_station_from_sentence(user_input)
        if extracted:
            user_input_for_validation = extracted
        else:
            user_input_for_validation = user_input

        official, crs = validate_station(user_input_for_validation)
        if official:
            state.origin_name = official
            state.origin_crs = crs
            if state.temp_dest:
                temp_official, temp_crs = validate_station(state.temp_dest)
                state.dest_name = temp_official
                state.dest_crs = temp_crs
                print(
                    f"BOT: Alright. You're going from {official} (code: {crs}) to {state.temp_dest}. What is your outbound travel date? (e.g., 15th July 2026)"
                )
                state.stage = "wait_outbound"
            else:
                print(
                    f"BOT: Got it. {official} (code: {crs}). Where do you want to go?"
                )
                state.stage = "wait_destination"
        else:
            print(
                "BOT: I'm not sure about that station. Please try again (e.g., 'London Waterloo' or say 'I am going to Norwich')."
            )
        return True

    # Wait for destination
    if state.stage == "wait_destination":
        extracted = extract_station_from_sentence(user_input)
        if extracted:
            user_input_for_validation = extracted
        else:
            user_input_for_validation = user_input

        official, crs = validate_station(user_input_for_validation)
        if official:
            state.dest_name = official
            state.dest_crs = crs
            print(
                f"BOT: Thanks. {official} (code: {crs}). What is your outbound travel date? (e.g., 15th July 2026)"
            )
            state.stage = "wait_outbound"
        else:
            print(
                "BOT: I'm not sure about that station. Please try again (e.g., 'London Waterloo' or say 'I am going to Norwich')."
            )
        return True

    if state.stage == "wait_outbound":
        extracted_date = extract_date_from_sentence(user_input) or user_input
        api_date = parse_travel_date(extracted_date)
        if api_date:
            state.outbound_date_raw = user_input
            state.outbound_date_api = api_date
            print("BOT: Is this a return journey? (yes/no)")
            state.stage = "ask_return"
        else:
            print(
                "BOT: Sorry, I didn't understand that date. Please try again (e.g., '15 July 2026')."
            )
        return True

    # Ask return or single
    if state.stage == "ask_return":
        if user_input.lower() in ["yes", "y", "return"]:
            state.is_return = True
            print("BOT: What is your return date?")
            state.stage = "wait_return"
        else:
            state.is_return = False
            state.stage = "search"
            return search_and_present_tickets()
        return True

    if state.stage == "wait_return":
        extracted_date = extract_date_from_sentence(user_input) or user_input
        api_date = parse_travel_date(extracted_date)
        if api_date:
            state.return_date_raw = user_input
            state.return_date_api = api_date
            state.stage = "search"
            return search_and_present_tickets()
        else:
            print("BOT: Sorry, I didn't understand that date. Please try again.")
        return True

    return False


def search_and_present_tickets_streamlit(state):
    msg = f"\nLooking for the cheapest ticket from {state.origin_name} to {state.dest_name}...\n"
    if state.is_return:
        msg += f"     Outbound: {state.outbound_date_raw}, Return: {state.return_date_raw}\n"
    else:
        msg += f"     Outbound: {state.outbound_date_raw}\n"

    price_pence, price_display, link, notice = search_national_rail_tickets(
        state.origin_crs,
        state.dest_crs,
        state.outbound_date_api,
        state.is_return,
        state.return_date_api if state.is_return else None,
    )

    if price_pence is None:
        msg += "Please try again with different stations or dates.\n"
        if notice:
            msg += f"\n📢 **Notice from National Rail:**\n{notice}\n"
        # Reset state
        state.origin_name = None
        state.origin_crs = None
        state.dest_name = None
        state.dest_crs = None
        state.outbound_date_raw = None
        state.outbound_date_api = None
        state.return_date_raw = None
        state.return_date_api = None
        state.is_return = False
        state.temp_dest = None
        state.stage = "idle"
        return msg

    msg += f"The cheapest ticket I found is {price_display}.\n"
    msg += f"You can book it at: {link}\n"
    if notice:
        msg += f"\n📢 **Notice from National Rail:**\n{notice}\n"
    msg += "Would you like to book another ticket? (type 'reset' to start over, or 'bye' to exit)"
    # Reset state for next conversation
    state.origin_name = None
    state.origin_crs = None
    state.dest_name = None
    state.dest_crs = None
    state.outbound_date_raw = None
    state.outbound_date_api = None
    state.return_date_raw = None
    state.return_date_api = None
    state.is_return = False
    state.temp_dest = None
    state.stage = "idle"
    return msg


def get_station_candidates(query, max_results=10):
    """
    Return a list of (official_name, crs) for stations that match the query.
    Uses fuzzy matching and substring matching, removes duplicates,
    and sorts results with preference for stations whose name starts with the query.
    """
    query_lower = query.lower().strip()
    if len(query_lower) < 2:
        return []

    candidates = []
    lowercase_names = [name.lower() for name in station_names]

    # 1) Fuzzy matching (get up to 30 matches, we'll later trim)
    fuzzy_matches = get_close_matches(query_lower, lowercase_names, n=30, cutoff=0.4)
    for m in fuzzy_matches:
        idx = lowercase_names.index(m)
        official = station_names[idx]
        crs = stations_df[stations_df["NAME"] == official]["CRS"].values[0]
        candidates.append((official, crs))

    # 2) Substring matching (always add any station containing the query, even if fuzzy already added)
    if len(query_lower) >= 3:
        for i, name in enumerate(lowercase_names):
            if query_lower in name:
                official = station_names[i]
                crs = stations_df[stations_df["NAME"] == official]["CRS"].values[0]
                if (official, crs) not in candidates:
                    candidates.append((official, crs))
                # No early break; we collect all, then trim later

    # Remove duplicates (keep first occurrence)
    seen = set()
    unique = []
    for name, crs in candidates:
        key = (name, crs)
        if key not in seen:
            seen.add(key)
            unique.append(key)

    # Sort: priority to stations whose name starts with the query (case-insensitive)
    def sort_key(item):
        name, _ = item
        name_lower = name.lower()
        if name_lower.startswith(query_lower):
            return (0, name)  # starts with query, then alphabetical
        elif query_lower in name_lower:
            return (1, name)  # contains query, then alphabetical
        else:
            return (2, name)

    unique.sort(key=sort_key)
    return unique[:max_results]


def ticket_response_streamlit(user_input, state):
    # Handle post-ticket commands
    if user_input.lower() == "reset":
        state.origin_name = None
        state.origin_crs = None
        state.dest_name = None
        state.dest_crs = None
        state.outbound_date_raw = None
        state.outbound_date_api = None
        state.return_date_raw = None
        state.return_date_api = None
        state.is_return = False
        state.temp_dest = None
        state.station_choices = []
        state.choice_target = None
        state.stage = "idle"
        return "Conversation reset. How can I help you?"

    if state.stage == "idle" and user_input.lower() == "yes":
        state.origin_name = None
        state.origin_crs = None
        state.dest_name = None
        state.dest_crs = None
        state.outbound_date_raw = None
        state.outbound_date_api = None
        state.return_date_raw = None
        state.return_date_api = None
        state.is_return = False
        state.temp_dest = None
        state.station_choices = []
        state.choice_target = None
        state.stage = "wait_origin"
        return "Great! Where would you like to travel from?"

    # Start new ticket conversation
    if state.stage == "idle":
        travel_match = re.search(
            r"(?:want to go to|need to go to|travelling to|going to)\s+([a-z\s]+?)(?:\.|$)",
            user_input.lower(),
        )
        if travel_match:
            dest_candidate = travel_match.group(1).strip().title()
            state.temp_dest = dest_candidate
            state.stage = "wait_origin"
            return f"I see you want to go to {dest_candidate}. Where will you be travelling from?"
        if is_ticket_intent(user_input):
            state.stage = "wait_origin"
            return "Sure! Where are you travelling from?"

    # Wait for station choice (disambiguation)
    if state.stage == "wait_station_choice":
        try:
            choice_idx = int(user_input) - 1
            if 0 <= choice_idx < len(state.station_choices):
                official, crs = state.station_choices[choice_idx]
                if state.choice_target == "origin":
                    state.origin_name = official
                    state.origin_crs = crs
                    state.station_choices = []
                    state.choice_target = None
                    if state.temp_dest:
                        # We have a destination already from travel phrase
                        temp_official, temp_crs = get_station_details(state.temp_dest)
                        if temp_official:
                            state.dest_name = temp_official
                            state.dest_crs = temp_crs
                        else:
                            state.temp_dest = None
                            state.stage = "wait_destination"
                            return f"Got it. {official} (code: {crs}). Where do you want to go?"
                        state.stage = "wait_outbound"
                        return f"Alright. You're going from {official} (code: {crs}) to {state.temp_dest}. What is your outbound travel date? (e.g., 15th July 2026)"
                    else:
                        state.stage = "wait_destination"
                        return f"Got it. {official} (code: {crs}). Where do you want to go?"
                elif state.choice_target == "destination":
                    state.dest_name = official
                    state.dest_crs = crs
                    state.station_choices = []
                    state.choice_target = None
                    state.stage = "wait_outbound"
                    return f"Thanks. {official} (code: {crs}). What is your outbound travel date? (e.g., 15th July 2026)"
            else:
                return f"Please choose a number between 1 and {len(state.station_choices)}."
        except ValueError:
            # If user typed a station name instead of a number, try to match directly
            official, crs = get_station_details(user_input)
            if official:
                if state.choice_target == "origin":
                    state.origin_name = official
                    state.origin_crs = crs
                    state.station_choices = []
                    state.choice_target = None
                    if state.temp_dest:
                        temp_official, temp_crs = get_station_details(state.temp_dest)
                        if temp_official:
                            state.dest_name = temp_official
                            state.dest_crs = temp_crs
                        else:
                            state.temp_dest = None
                            state.stage = "wait_destination"
                            return f"Got it. {official} (code: {crs}). Where do you want to go?"
                        state.stage = "wait_outbound"
                        return f"Alright. You're going from {official} (code: {crs}) to {state.temp_dest}. What is your outbound travel date? (e.g., 15th July 2026)"
                elif state.choice_target == "destination":
                    state.dest_name = official
                    state.dest_crs = crs
                    state.station_choices = []
                    state.choice_target = None
                    state.stage = "wait_outbound"
                    return f"Thanks. {official} (code: {crs}). What is your outbound travel date? (e.g., 15th July 2026)"
            else:
                return "I didn't recognise that station. Please type the number from the list, or type the full station name."

    # Wait for origin
    if state.stage == "wait_origin":
        extracted = extract_station_from_sentence(user_input)
        if extracted:
            user_input_for_validation = extracted
        else:
            user_input_for_validation = user_input

        # Check if this is a single word (potential ambiguous city)
        is_single_word = " " not in user_input_for_validation.strip()

        if not is_single_word:
            # Multi‑word input: try direct match first
            official, crs = get_station_details(user_input_for_validation)
            if official:
                state.origin_name = official
                state.origin_crs = crs
                if state.temp_dest:
                    temp_official, temp_crs = get_station_details(state.temp_dest)
                    if temp_official:
                        state.dest_name = temp_official
                        state.dest_crs = temp_crs
                    else:
                        state.temp_dest = None
                        state.stage = "wait_destination"
                        return f"Got it. {official} (code: {crs}). Where do you want to go?"
                    state.stage = "wait_outbound"
                    return f"Alright. You're going from {official} (code: {crs}) to {state.temp_dest}. What is your outbound travel date? (e.g., 15th July 2026)"
                else:
                    state.stage = "wait_destination"
                    return f"Got it. {official} (code: {crs}). Where do you want to go?"
        # For single‑word input, go directly to candidate list (skip fuzzy match)
        candidates = get_station_candidates(user_input_for_validation)
        if candidates:
            # If only one candidate, we could use it directly, but for consistency we still show list?
            # Let's use it directly if only one.
            if len(candidates) == 1:
                official, crs = candidates[0]
                state.origin_name = official
                state.origin_crs = crs
                if state.temp_dest:
                    temp_official, temp_crs = get_station_details(state.temp_dest)
                    if temp_official:
                        state.dest_name = temp_official
                        state.dest_crs = temp_crs
                    else:
                        state.temp_dest = None
                        state.stage = "wait_destination"
                        return f"Got it. {official} (code: {crs}). Where do you want to go?"
                    state.stage = "wait_outbound"
                    return f"Alright. You're going from {official} (code: {crs}) to {state.temp_dest}. What is your outbound travel date? (e.g., 15th July 2026)"
                else:
                    state.stage = "wait_destination"
                    return f"Got it. {official} (code: {crs}). Where do you want to go?"
            else:
                state.station_choices = candidates
                state.choice_target = "origin"
                state.stage = "wait_station_choice"
                options = "\n".join(
                    [
                        f"{i + 1}. {name} ({crs})"
                        for i, (name, crs) in enumerate(candidates[:10])
                    ]
                )
                return f"There are several stations matching '{user_input_for_validation}'. Please choose one by number:\n{options}"
        else:
            return "I'm not sure about that station. Please try again (e.g., 'London Waterloo' or say 'I am going to Norwich')."

    # Wait for destination
    if state.stage == "wait_destination":
        extracted = extract_station_from_sentence(user_input)
        if extracted:
            user_input_for_validation = extracted
        else:
            user_input_for_validation = user_input

        is_single_word = " " not in user_input_for_validation.strip()

        if not is_single_word:
            official, crs = get_station_details(user_input_for_validation)
            if official:
                state.dest_name = official
                state.dest_crs = crs
                state.stage = "wait_outbound"
                return f"Thanks. {official} (code: {crs}). What is your outbound travel date? (e.g., 15th July 2026)"

        candidates = get_station_candidates(user_input_for_validation)
        if candidates:
            if len(candidates) == 1:
                official, crs = candidates[0]
                state.dest_name = official
                state.dest_crs = crs
                state.stage = "wait_outbound"
                return f"Thanks. {official} (code: {crs}). What is your outbound travel date? (e.g., 15th July 2026)"
            else:
                state.station_choices = candidates
                state.choice_target = "destination"
                state.stage = "wait_station_choice"
                options = "\n".join(
                    [
                        f"{i + 1}. {name} ({crs})"
                        for i, (name, crs) in enumerate(candidates[:10])
                    ]
                )
                return f"There are several stations matching '{user_input_for_validation}'. Please choose one by number:\n{options}"
        else:
            return "I'm not sure about that station. Please try again (e.g., 'London Waterloo' or say 'I am going to Norwich')."

    # Wait for outbound date
    if state.stage == "wait_outbound":
        extracted_date = extract_date_from_sentence(user_input) or user_input
        api_date = parse_travel_date(extracted_date)
        if api_date:
            state.outbound_date_raw = user_input
            state.outbound_date_api = api_date
            state.stage = "ask_return"
            return "Is this a return journey? (yes/no)"
        else:
            return "Sorry, I didn't understand that date. Please try again (e.g., '15 July 2026')."

    # Ask return or single
    if state.stage == "ask_return":
        answer = user_input.lower().strip()
        if answer in ["yes", "y", "return"]:
            state.is_return = True
            state.stage = "wait_return"
            return "What is your return date?"
        else:
            state.is_return = False
            state.stage = "search"
            return search_and_present_tickets_streamlit(state)

    # Wait for return date
    if state.stage == "wait_return":
        extracted_date = extract_date_from_sentence(user_input) or user_input
        api_date = parse_travel_date(extracted_date)
        if api_date:
            state.return_date_raw = user_input
            state.return_date_api = api_date
            state.stage = "search"
            return search_and_present_tickets_streamlit(state)
        else:
            return "Sorry, I didn't understand that date. Please try again (e.g., '15 July 2026')."

    return None
