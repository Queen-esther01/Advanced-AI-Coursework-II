from expert.plan_retriever import RetrievedChunk, format_context, source_labels
from llm_client import LLMClient

_LLM_FAILURE_MARKERS = (
    "I could not get a model response",
    "The AI service",
    "OPENROUTER_API_KEY",
)


def _llm_call_failed(reply: str) -> bool:
    return any(reply.startswith(marker) for marker in _LLM_FAILURE_MARKERS)


def _situation_header(*, event_type: str | None, station: str | None) -> str:
    if event_type == "station_disruption" and station:
        return f"**Station disruption at {station}** — from the indexed SWR station disruption plan:\n\n"
    if event_type == "line_blockage":
        return "**Line blockage** — from the indexed contingency plan:\n\n"
    return ""


def _chunks_are_mostly_boilerplate(chunks: list[RetrievedChunk]) -> bool:
    if not chunks:
        return False
    low_value = 0
    for chunk in chunks:
        section = (chunk.metadata.get("section") or "").lower()
        if any(term in section for term in ("introduction", "top tips")):
            low_value += 1
    return low_value >= max(1, len(chunks) // 2)


def _fallback_disruption_reply(chunks: list[RetrievedChunk]) -> str:
    context = format_context(chunks)
    return (
        "The AI summary could not be generated (often due to **rate limits** on the API). "
        "Below are the **retrieved plan sections** from the knowledge base.\n\n"
        "## Staff advice\n"
        "- Use the retrieved excerpts below for station staff actions.\n\n"
        "## Passenger advice\n"
        "- Use the retrieved excerpts below for passenger-facing steps.\n\n"
        "### Retrieved plan excerpts\n\n"
        f"{context}"
    )


DISRUPTION_SYSTEM_PROMPT = """
You are an expert assistant for South Western Railway operational staff handling an active disruption right now.

You will receive retrieved excerpts from official contingency and station disruption plans.
Answer ONLY using that context. If the context does not contain an item, say it is not in the retrieved plan sections.

Prioritise immediate operational steps (communications with control, station operation, crowd management,
alternative transport, passenger information). Do NOT pad the answer with generic onboarding such as
reading the emergency plan on a noticeboard, briefing volunteers on arrival, or "use this plan in conjunction with"
unless the user explicitly needs preparedness guidance.

Structure your reply with exactly these markdown headings:
## Staff advice
## Passenger advice

Under each heading use short bullet points with concrete actions for staff on the ground now.
Mention specific train headcodes, bus routes, or station names only when they appear in the context.

Do not tell the user to open another document or slide. Do not invent procedures.
""".strip()


def build_disruption_reply(
    chunks: list[RetrievedChunk],
    *,
    retrieval_query: str,
    info_topics: list[str],
    llm_client: LLMClient,
    station: str | None = None,
    event_type: str | None = None,
    staff_role: str | None = None,
    service_period: str | None = None,
    incident_time: str | None = None,
    derived_actions: list[str] | None = None,
) -> str:
    if not chunks:
        return "I could not find matching contingency plan sections in the knowledge base for that location. Please contact a member of staff for assistance."

    context = format_context(chunks)
    topics = ", ".join(info_topics) if info_topics else "staff and passengers"
    role_text = staff_role or "unspecified"
    period_text = service_period or "unspecified"
    time_text = incident_time or "unspecified"
    derived_text = ", ".join(derived_actions or []) or "none"
    user_prompt = (
        f"Staff request: {retrieval_query}\n"
        f"Focus areas: {topics}\n\n"
        f"Role context: {role_text}\n"
        f"Service period: {period_text}\n"
        f"Incident time: {time_text}\n"
        f"Derived required actions: {derived_text}\n\n"
        f"--- Retrieved plan excerpts ---\n{context}"
    )

    reply = llm_client.complete_reply(
        [{"role": "user", "content": user_prompt}],
        system_prompt=DISRUPTION_SYSTEM_PROMPT,
    )
    if _llm_call_failed(reply):
        reply = _fallback_disruption_reply(chunks)

    header = _situation_header(event_type=event_type, station=station)
    if header and not reply.startswith("**"):
        reply = header + reply

    if _chunks_are_mostly_boilerplate(chunks):
        reply += (
            "\n\n*Note: retrieved sections were mostly introductory material. "
            "Re-index the station plan or ask about **communication with control** or "
            "**practical operation** for more specific steps.*"
        )

    sources = source_labels(chunks[:3])
    if sources:
        reply += "\n\n---\n*Sources: " + "; ".join(sources) + "*"
    return reply
