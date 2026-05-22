from pathlib import Path

STATION_PLAN_IMAGE_PROMPT = """You catalogue images extracted from UK railway station disruption plan documents (Word).

Analyse the image and return JSON only with these keys:
- caption: one short sentence suitable as a markdown image caption
- suggested_filename: a descriptive kebab-case filename including the correct extension (e.g. aldershot-station-map.png)
- description: a clear paragraph explaining what is shown — labels, diagrams, maps, flows, or text visible in the image — and why it matters for station disruption response
- image_type: one of "station-diagram" | "track-schematic" | "flow-chart" | "table" | "contact-list" | "logo-or-branding" | "other"
- stations_or_junctions: list of any station or junction names visible (empty list if none)

Use British English. Be specific about visible content. Do not invent details that are not visible."""

CPT_PRESENTATION_IMAGE_PROMPT = """You catalogue full-slide renders from UK railway CPT (Contingency Plan Training) PowerPoint presentations used by South Western Railway (SWR).

Each slide is a composed layout: plan text, status boxes, track schematics (often with blocked sections in red), network maps, and contact tables together form one operational view. You will also receive the slide's extracted text — use it for plan names and status; focus your description on the visual layout and diagrams.

Analyse the full slide image and return JSON only with these keys:
- caption: one short sentence suitable as a markdown image caption
- suggested_filename: a descriptive kebab-case filename including the correct extension (e.g. plan-20-1-woking-guildford-slide.png)
- description: a clear paragraph describing the whole slide layout and how the visual elements relate to rail disruption response
- image_type: one of "slide-composite" | "track-schematic" | "network-map" | "station-diagram" | "flow-chart" | "table" | "logo-or-branding" | "other"
- stations_or_junctions: list of every station and junction name visible on the slide (empty list if none)
- blocked_section: describe which line or section is highlighted as blocked (typically in red), and direction if shown — otherwise null

Describe the slide as a whole:
- How text, schematic, and map regions relate to each other
- Name stations and junctions visible on track diagrams
- Note blocked/highlighted sections and operational meaning
- Mention depots, sidings, crossovers, or signal boxes if visible

Use British English. Be specific about visible content. Do not invent details that are not visible."""

CPT_SLIDE_GUIDE_PROMPT = """You write operator documentation for South Western Railway (SWR) CPT contingency slides.

This document is the operator's ONLY source of truth. They cannot open the PowerPoint or any other file. Never say "refer to", "see the plan", "check the map/list view", "consult", or "for details see …".

Inputs:
1. Slide text (authoritative) — every headcode, time, bus route, terminate/shunt, and passenger instruction must appear in your output
2. Visual summary — use only for geography (blocked section, line, stations on the diagram)

Write complete operator documentation, not a short summary. Use markdown with these sections (omit empty sections only if the slide truly has no content for them):

## Overview
Plan number, line name, status (e.g. both lines blocked), and what is blocked (from text + diagram).

## Geography
Blocked section, direction if known, stations/junctions on the schematic.

## Service alteration (step-by-step)
Numbered steps. Copy each Off Peak and Peak working from the slide text as its own step (keep exact train IDs, times, cancelled/diverted/via Cobham wording). Do not collapse multiple trains into one line.

## Alternative passenger journey (step-by-step)
Numbered steps for buses, taxis, walking routes, and station-specific passenger moves.

## Signallers (step-by-step)
Numbered actionable steps. If the slide says "See Principles of Service Alteration", copy those principles here as numbered steps — never tell the reader to look elsewhere.

## Station staff (step-by-step)
Numbered steps from "Additional Information for Station Staff".

## Passenger information (step-by-step)
Numbered steps including bus numbers, frequencies, and walking distances exactly as written.

Index or menu slides (e.g. list/map index, plan picker, only plan IDs and blocked/available labels, no Principles of Service Alteration, no signaller/station/passenger steps, no train headcodes): return an empty guide — do not write documentation for these.

Do not invent details missing from the inputs. Do not shorten operative slides to a headline.

Return JSON only with this key:
- guide: the full markdown operator document as above, or "" for index/menu slides

Use British English."""

_PROMPTS = {
    "station_plan": STATION_PLAN_IMAGE_PROMPT,
    "cpt_presentation": CPT_PRESENTATION_IMAGE_PROMPT,
}


def document_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".pptx", ".pptm"):
        return "cpt_presentation"
    if ext in (".docx", ".docm"):
        return "station_plan"
    raise ValueError(f"Unsupported document type for image prompts: {path}")


def image_description_prompt(kind: str) -> str:
    try:
        return _PROMPTS[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown document kind: {kind}") from exc


def slide_guide_prompt(kind: str) -> str:
    if kind == "cpt_presentation":
        return CPT_SLIDE_GUIDE_PROMPT
    raise ValueError(f"No slide guide prompt for document kind: {kind}")
