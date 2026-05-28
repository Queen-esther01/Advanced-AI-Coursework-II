from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Token

from nlp_utils import nlp

_BLOCK_LEMMAS = frozenset({"block", "blockage", "blockade", "blocked"})
_DISRUPT_LEMMAS = frozenset({"disruption", "disrupt", "incident", "problem"})
_SEVERITY_FULL = frozenset(
    {"full", "complete", "entire", "total", "closed", "all", "both"}
)
_SEVERITY_PARTIAL = frozenset({"partial", "single", "one"})
_TIME_LABELS = frozenset({"morning", "afternoon", "evening", "night", "tonight", "now"})
_DURATION_UNITS = frozenset(
    {"minute", "minutes", "min", "mins", "hour", "hours", "hr", "hrs", "h"}
)
_INFO_TOPIC_LEMMAS = {
    "staff": frozenset(
        {"signaller", "signaler", "staff", "crew", "divert", "terminate"}
    ),
    "passengers": frozenset({"passenger", "bus", "replacement", "taxi", "customer"}),
    "contacts": frozenset({"contact", "phone", "number", "call"}),
    "routes": frozenset({"route", "diversion", "detour", "via"}),
}
_ROLE_PHRASES = {
    "station_staff": ["station staff"],
    "signaller": ["signaller", "signaler", "signalling"],
    "control": ["operations control", "control room", "controller", "control"],
}
_TYPOS = {"blocage": "blockage"}


@dataclass
class NlpExtraction:
    mentions_blockage: bool = False
    mentions_disruption: bool = False
    event_type: str | None = None
    route_from: str | None = None
    route_to: str | None = None
    station_phrase: str | None = None
    staff_role: str | None = None
    severity: str | None = None
    incident_time: str | None = None
    duration_minutes: int | None = None
    info_topics: list[str] = field(default_factory=list)
    two_station_answer: bool = False


@lru_cache(maxsize=1)
def _matchers() -> tuple[Matcher, PhraseMatcher]:
    matcher = Matcher(nlp.vocab)
    matcher.add(
        "ROUTE_BETWEEN",
        [
            [
                {"LOWER": {"IN": ["between", "btw"]}},
                {"IS_ALPHA": True, "OP": "+"},
                {"LOWER": "and"},
                {"IS_ALPHA": True, "OP": "+"},
            ],
        ],
    )
    matcher.add(
        "ROUTE_FROM_TO",
        [
            [
                {"LOWER": "from"},
                {"IS_ALPHA": True, "OP": "+"},
                {"LOWER": "to"},
                {"IS_ALPHA": True, "OP": "+"},
            ],
        ],
    )
    matcher.add(
        "DISRUPTION_AT",
        [
            [
                {"LEMMA": {"IN": list(_DISRUPT_LEMMAS)}},
                {"LOWER": "at"},
                {"IS_ALPHA": True, "OP": "+"},
            ],
        ],
    )
    matcher.add(
        "STATION_DISRUPTION",
        [
            [{"LOWER": "station"}, {"LEMMA": "disruption"}],
        ],
    )
    matcher.add(
        "LINE_BLOCKAGE",
        [
            [{"LOWER": "line"}, {"LEMMA": {"IN": list(_BLOCK_LEMMAS)}}],
            [{"LEMMA": {"IN": list(_BLOCK_LEMMAS)}}, {"LOWER": "between"}],
        ],
    )
    matcher.add(
        "TWO_STATIONS",
        [
            [{"LIKE_NUM": True}, {"LOWER": {"IN": ["station", "stations"]}}],
            [{"LOWER": "two"}, {"LOWER": {"IN": ["station", "stations"]}}],
        ],
    )

    phrase = PhraseMatcher(nlp.vocab, attr="LOWER")
    for role_id, phrases in _ROLE_PHRASES.items():
        patterns = [nlp.make_doc(p) for p in phrases]
        phrase.add(role_id.upper(), patterns)

    return matcher, phrase


def get_doc(text: str) -> Doc:
    return nlp(text)


def _subtree_text(token: Token) -> str:
    span = token.subtree
    return " ".join(t.text for t in span if not t.is_punct).strip()


def _phrase_excluding_conj(token: Token) -> str:
    parts = [t for t in token.subtree if t.dep_ not in ("conj", "cc")]
    return " ".join(t.text for t in parts if not t.is_punct).strip()


def _conj_phrase(head: Token) -> str | None:
    doc = head.doc
    conj: Token | None = None
    for child in head.children:
        if child.dep_ == "conj":
            conj = child
            break
    if conj is None:
        return None
    tokens = [conj]
    i = conj.i + 1
    stop = {"at", "on", "this", "has", "for", "since", ",", "?"}
    while i < len(doc):
        t = doc[i]
        if t.text.lower() in stop or t.dep_ == "prep":
            break
        tokens.append(t)
        i += 1
    return " ".join(t.text for t in tokens if not t.is_punct).strip()


def _normalise_token(tok: Token) -> str:
    raw = tok.text.lower()
    return _TYPOS.get(raw, tok.lemma_.lower())


def _has_block_language(doc: Doc) -> bool:
    for tok in doc:
        norm = _normalise_token(tok)
        if norm in _BLOCK_LEMMAS:
            return True
    return False


def _has_disruption_language(doc: Doc) -> bool:
    for tok in doc:
        if _normalise_token(tok) in _DISRUPT_LEMMAS:
            return True
    return False


def _route_from_between(doc: Doc) -> tuple[str | None, str | None]:
    for tok in doc:
        if tok.lemma_.lower() not in ("between", "btw"):
            continue
        for child in tok.children:
            if child.dep_ != "pobj":
                continue
            start_phrase = _phrase_excluding_conj(child)
            end_phrase = _conj_phrase(child)
            if start_phrase and end_phrase:
                return start_phrase, end_phrase
    return None, None


def _route_from_prep(doc: Doc) -> tuple[str | None, str | None]:
    from_tok = to_tok = None
    for tok in doc:
        low = tok.text.lower()
        if low == "from" and from_tok is None:
            from_tok = tok
        elif low == "to" and from_tok is not None and to_tok is None:
            to_tok = tok
    if from_tok is None or to_tok is None or to_tok.i <= from_tok.i:
        return None, None
    from_span = doc[from_tok.i + 1 : to_tok.i]
    to_span = doc[to_tok.i + 1 :]
    stop = {
        "and",
        "with",
        "is",
        "are",
        "both",
        "one",
        "full",
        "partial",
        "at",
        "this",
        "has",
    }
    to_tokens = []
    for t in to_span:
        if t.text.lower() in stop:
            break
        to_tokens.append(t)
    if not to_tokens:
        return None, None
    a = " ".join(t.text for t in from_span if not t.is_punct).strip()
    b = " ".join(t.text for t in to_tokens if not t.is_punct).strip()
    return (a or None), (b or None)


def _route_from_matcher(doc: Doc) -> tuple[str | None, str | None]:
    matcher, _ = _matchers()
    for _match_id, start, end in matcher(doc):
        label = nlp.vocab.strings[_match_id]
        span = doc[start:end]
        if label == "ROUTE_BETWEEN":
            if " and " in span.text.lower():
                mid = span.text.lower().find(" and ")
                if mid != -1:
                    left = (
                        span.text[:mid]
                        .replace("between", "")
                        .replace("btw", "")
                        .strip()
                    )
                    right = span.text[mid + 5 :].strip()
                    return left or None, right or None
        if label == "ROUTE_FROM_TO":
            text = span.text
            m = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+)$", text, re.I)
            if m:
                return m.group(1).strip(), m.group(2).strip()
    return None, None


def extract_route_phrases(
    text: str, *, doc: Doc | None = None
) -> tuple[str | None, str | None]:
    doc = doc or get_doc(text)
    for extractor in (_route_from_between, _route_from_prep, _route_from_matcher):
        a, b = extractor(doc)
        if a and b:
            return a, b
    return None, None


def _station_phrase_after_at(doc: Doc) -> str | None:
    matcher, _ = _matchers()
    for match_id, start, end in matcher(doc):
        if nlp.vocab.strings[match_id] == "DISRUPTION_AT":
            span = doc[start:end]
            for tok in span:
                if tok.text.lower() == "at" and tok.i + 1 < len(doc):
                    return _subtree_text(doc[tok.i + 1])
    for tok in doc:
        if tok.text.lower() == "at" and tok.head.lemma_.lower() in _DISRUPT_LEMMAS:
            for child in tok.head.children:
                if child.dep_ == "prep" and child == tok:
                    for pobj in tok.children:
                        if pobj.dep_ == "pobj":
                            return _subtree_text(pobj)
    return None


def extract_staff_role(
    text: str, *, doc: Doc | None = None, pending_staff_slot: bool = False
) -> str | None:
    doc = doc or get_doc(text)
    _, phrase = _matchers()
    for match_id, start, end in phrase(doc):
        role = nlp.vocab.strings[match_id].lower()
        if role == "station_staff":
            return "station_staff"
        if role == "signaller":
            return "signaller"
        if role == "control":
            return "control"
    for tok in doc:
        if tok.lemma_.lower() in ("signaller", "signaler"):
            return "signaller"
    lower = text.lower().strip()
    if re.search(r"\b(?:i\s*['']?m\s+(?:a\s+)?)?signaller\b", lower):
        return "signaller"
    if pending_staff_slot and lower in ("staff", "station"):
        return "station_staff"
    return None


def extract_severity(text: str, *, doc: Doc | None = None) -> str | None:
    doc = doc or get_doc(text)
    lemmas = {_normalise_token(t) for t in doc if not t.is_stop}
    text_l = text.strip().lower()
    if lemmas & _SEVERITY_PARTIAL and not lemmas & _SEVERITY_FULL:
        return "one_line_blocked"
    if lemmas & _SEVERITY_FULL or text_l in ("full", "partial"):
        if text_l == "partial" or "partial" in lemmas:
            return "one_line_blocked"
        return "both_lines_blocked"
    for chunk in doc.noun_chunks:
        cl = chunk.text.lower()
        if "one line" in cl or "partial" in cl:
            return "one_line_blocked"
        if "both lines" in cl or "full block" in cl:
            return "both_lines_blocked"
    return None


def extract_event_type(
    text: str,
    *,
    doc: Doc | None = None,
    route: tuple[str | None, str | None] | None = None,
) -> str | None:
    if route and route[0] and route[1]:
        return None
    doc = doc or get_doc(text)
    matcher, _ = _matchers()
    for match_id, start, end in matcher(doc):
        label = nlp.vocab.strings[match_id]
        if label == "STATION_DISRUPTION":
            return "station_disruption"
        if label in ("LINE_BLOCKAGE", "ROUTE_BETWEEN"):
            return "line_blockage"
        if label == "TWO_STATIONS":
            return "line_blockage"
    lemmas = {_normalise_token(t) for t in doc}
    if "disruption" in lemmas and "station" in lemmas:
        return "station_disruption"
    if _has_block_language(doc) or "between" in {t.text.lower() for t in doc}:
        return "line_blockage"
    if "disruption" in lemmas:
        return "station_disruption"
    return None


def extract_incident_time(text: str, *, doc: Doc | None = None) -> str | None:
    doc = doc or get_doc(text)
    for ent in doc.ents:
        if ent.label_ in ("TIME", "DATE"):
            return ent.text
    for tok in doc:
        if tok.text.lower() in _TIME_LABELS:
            return tok.text.lower()
    match = re.search(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b", text)
    if match:
        return match.group(0)
    match = re.search(r"\b([1-9]|1[0-2])\s*(am|pm)\b", text, re.I)
    if match:
        return match.group(0)
    return None


def extract_duration_minutes(text: str, *, doc: Doc | None = None) -> int | None:
    doc = doc or get_doc(text)
    best: tuple[int, int] | None = None
    for tok in doc:
        if tok.lemma_.lower() not in _DURATION_UNITS:
            continue
        amount = None
        for child in tok.children:
            if child.like_num or child.dep_ in ("nummod", "quantmod"):
                try:
                    amount = int(child.text)
                except ValueError:
                    continue
                break
        if amount is None:
            for left in doc[max(0, tok.i - 3) : tok.i]:
                if left.like_num:
                    try:
                        amount = int(left.text)
                    except ValueError:
                        continue
                    break
        if amount is None:
            continue
        minutes = (
            amount * 60
            if tok.lemma_.lower().startswith("hour")
            or tok.text.lower() in ("hr", "hrs", "h")
            else amount
        )
        if best is None or minutes > best[0]:
            best = (minutes, tok.i)
    return best[0] if best else None


def extract_info_topics(text: str, *, doc: Doc | None = None) -> list[str]:
    doc = doc or get_doc(text)
    lemmas = {_normalise_token(t) for t in doc}
    lower = text.lower()
    found: list[str] = []
    if lemmas & _INFO_TOPIC_LEMMAS["staff"] or "station staff" in lower:
        found.append("staff")
    if (
        lemmas & _INFO_TOPIC_LEMMAS["passengers"]
        or "replacement bus" in lower
        or "bus replacement" in lower
    ):
        found.append("passengers")
    if lemmas & _INFO_TOPIC_LEMMAS["contacts"]:
        found.append("contacts")
    if (
        lemmas & _INFO_TOPIC_LEMMAS["routes"]
        or "alternative route" in lower
        or " via " in lower
    ):
        found.append("routes")
    return found


def extract_two_station_answer(text: str, *, doc: Doc | None = None) -> bool:
    doc = doc or get_doc(text)
    matcher, _ = _matchers()
    for match_id, _, _ in matcher(doc):
        if nlp.vocab.strings[match_id] == "TWO_STATIONS":
            return True
    return False


def analyse_incident_text(text: str) -> NlpExtraction:
    doc = get_doc(text)
    route = extract_route_phrases(text, doc=doc)
    return NlpExtraction(
        mentions_blockage=_has_block_language(doc),
        mentions_disruption=_has_disruption_language(doc),
        event_type=extract_event_type(text, doc=doc, route=route),
        route_from=route[0],
        route_to=route[1],
        station_phrase=_station_phrase_after_at(doc),
        staff_role=extract_staff_role(text, doc=doc),
        severity=extract_severity(text, doc=doc),
        incident_time=extract_incident_time(text, doc=doc),
        duration_minutes=extract_duration_minutes(text, doc=doc),
        info_topics=extract_info_topics(text, doc=doc),
        two_station_answer=extract_two_station_answer(text, doc=doc),
    )
