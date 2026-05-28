import base64
import json
import mimetypes
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Iterator

from openrouter import OpenRouter
from openrouter.types import UNSET


def image_to_data_url(path: str | Path) -> str:
    path = Path(path)
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _message_content_to_text(content: Any) -> str:
    if content is None or content is UNSET:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def _format_llm_error(exc: BaseException) -> str:
    msg = str(exc)
    lower = msg.lower()
    if "429" in msg or ("rate" in lower and "limit" in lower):
        return (
            "The AI service is temporarily **rate-limited** (too many requests). "
            "Wait a minute and try again, or use a different model in `LLMClient`."
        )
    if "401" in msg or "403" in msg or "unauthorized" in lower or "forbidden" in lower:
        return "The AI API key was **rejected**. Check `OPENROUTER_API_KEY` in your `.env` file."
    if "402" in msg or "insufficient" in lower and "credit" in lower:
        return (
            "The AI account has **insufficient credits** on OpenRouter. "
            "Top up the account or switch models."
        )
    api_message = re.search(r"'message':\s*'((?:\\'|[^'])*)'", msg)
    if api_message:
        detail = api_message.group(1).replace("\\'", "'")
        if "429" in detail or "rate" in detail.lower():
            return (
                "The AI service is temporarily **rate-limited**. "
                "Wait a minute and try again."
            )
        return f"The AI service returned an error: **{detail}**"
    if "validation failed" in lower or "error finding id" in lower:
        return (
            "The AI service returned an unexpected response "
            "(often **rate limiting** or a temporary outage). Wait and try again."
        )
    short = msg if len(msg) <= 280 else msg[:277] + "..."
    return f"I could not get a model response: {short}"


def _parse_json_response(text: str) -> dict[str, Any]:
    raw = text.strip()
    cleaned = raw
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError as exc:
                snippet = raw[:500].replace("\n", "\\n")
                raise ValueError(
                    f"Could not parse model JSON response: {exc}. Raw snippet: {snippet}"
                ) from exc
        else:
            snippet = raw[:500].replace("\n", "\\n")
            raise ValueError(
                f"Could not parse model JSON response. Raw snippet: {snippet}"
            )
    return _normalize_image_description(parsed)


def _normalize_image_description(data: Any) -> dict[str, Any]:
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                return item
    raise ValueError(
        f"Expected a JSON object for image description, got: {type(data).__name__}"
    )


class LLMClient:
    DEFAULT_MODEL = "openai/gpt-4o-mini"
    DEFAULT_VISION_MODEL = "google/gemini-2.0-flash-001"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        vision_model: str | None = None,
        system_prompt: str | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or self.DEFAULT_MODEL
        self.vision_model = vision_model or self.DEFAULT_VISION_MODEL
        self.system_prompt = system_prompt

    def stream_reply(self, messages: list[dict]) -> Iterator[str]:
        if not self.api_key:
            yield "OPENROUTER_API_KEY is not set, so I cannot call the LLM yet."
            return

        model_messages = messages
        if self.system_prompt:
            model_messages = [
                {"role": "system", "content": self.system_prompt},
                *messages,
            ]

        try:
            with OpenRouter(api_key=self.api_key) as client:
                response = client.chat.send(
                    model=self.model,
                    messages=model_messages,
                    stream=True,
                )
                with response as event_stream:
                    for chunk in event_stream:
                        if not chunk.choices:
                            continue
                        content = chunk.choices[0].delta.content
                        if content and content is not UNSET:
                            yield content
        except Exception as exc:
            yield _format_llm_error(exc)

    def stream_delay_assistant(self, messages: list[dict]) -> Iterator[str]:
        from task_2.delay_tool import tools as delay_tools
        from task_2.utils import MAX_TOOL_ROUNDS, execute_tool_call

        yield from self.stream_with_tools(
            messages,
            tools=delay_tools,
            execute_tool=execute_tool_call,
            max_rounds=MAX_TOOL_ROUNDS,
        )

    def stream_with_tools(
        self,
        messages: list[dict],
        *,
        tools: list[dict],
        execute_tool: Callable[[str, str], dict],
        max_rounds: int = 5,
    ) -> Iterator[str]:
        from task_2.utils import (
            clean_model_text,
            merge_stream_tool_calls,
            tool_calls_from_accumulator,
        )

        if not self.api_key:
            yield "OPENROUTER_API_KEY is not set, so I cannot call the LLM yet."
            return

        model_messages = list(messages)
        if self.system_prompt:
            model_messages = [
                {"role": "system", "content": self.system_prompt},
                *model_messages,
            ]

        try:
            with OpenRouter(api_key=self.api_key) as client:
                for _ in range(max_rounds):
                    content_parts: list[str] = []
                    tool_call_accumulator: dict = {}

                    response = client.chat.send(
                        model=self.model,
                        messages=model_messages,
                        stream=True,
                        tools=tools,
                    )
                    with response as event_stream:
                        for chunk in event_stream:
                            if not chunk.choices:
                                continue

                            delta = chunk.choices[0].delta
                            if delta.content and delta.content is not UNSET:
                                cleaned = clean_model_text(delta.content)
                                if cleaned:
                                    content_parts.append(cleaned)
                                    yield cleaned

                            if delta.tool_calls:
                                merge_stream_tool_calls(
                                    tool_call_accumulator, delta.tool_calls
                                )

                    tool_calls = tool_calls_from_accumulator(tool_call_accumulator)
                    if not tool_calls:
                        return

                    model_messages.append(
                        {
                            "role": "assistant",
                            "content": "".join(content_parts) or None,
                            "tool_calls": tool_calls,
                        }
                    )

                    for tool_call in tool_calls:
                        tool_result = execute_tool(
                            tool_call["function"]["name"],
                            tool_call["function"]["arguments"],
                        )
                        model_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": json.dumps(tool_result),
                            }
                        )

                yield (
                    "Sorry, I could not finish the tool request after several attempts."
                )
        except Exception as exc:
            yield _format_llm_error(exc)

    def complete_reply(
        self,
        messages: list[dict],
        *,
        system_prompt: str | None = None,
        max_tokens: int = 2048,
    ) -> str:
        if not self.api_key:
            return "OPENROUTER_API_KEY is not set, so I cannot call the LLM yet."

        prompt = system_prompt if system_prompt is not None else self.system_prompt
        model_messages = messages
        if prompt:
            model_messages = [{"role": "system", "content": prompt}, *messages]

        try:
            with OpenRouter(api_key=self.api_key) as client:
                response = client.chat.send(
                    model=self.model,
                    messages=model_messages,
                    max_tokens=max_tokens,
                )
            return _message_content_to_text(response.choices[0].message.content).strip()
        except Exception as exc:
            return _format_llm_error(exc)

    def describe_image(
        self,
        image_path: str | Path,
        *,
        prompt: str,
        section_title: str | None = None,
        original_filename: str | None = None,
        slide_text: str | None = None,
        max_tokens: int = 1500,
    ) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(path)

        context_parts = []
        if section_title:
            context_parts.append(f"Document section: {section_title}")
        if original_filename:
            context_parts.append(f"Original embedded filename: {original_filename}")
        if slide_text and slide_text.strip():
            context_parts.append(
                "Slide text extracted from the presentation (for context):\n"
                + slide_text.strip()
            )
        context = "\n".join(context_parts)

        user_text = prompt
        if context:
            user_text = f"{context}\n\n{user_text}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(path)},
                    },
                ],
            }
        ]

        with OpenRouter(api_key=self.api_key) as client:
            response = client.chat.send(
                model=self.vision_model,
                messages=messages,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )

        raw = _message_content_to_text(response.choices[0].message.content)
        data = _parse_json_response(raw)
        result = {k: v for k, v in data.items()}
        result.setdefault("caption", "")
        result.setdefault("suggested_filename", "")
        result.setdefault("description", "")
        result["caption"] = str(result["caption"]).strip()
        result["suggested_filename"] = str(result["suggested_filename"]).strip()
        result["description"] = str(result["description"]).strip()
        result["section"] = section_title or ""
        result["source_path"] = str(path)
        return result

    def generate_slide_guide(
        self,
        *,
        prompt: str,
        section_title: str,
        slide_text: str,
        image_description: dict[str, Any],
        max_tokens: int = 4096,
    ) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        visual_parts = [
            f"Caption: {image_description.get('caption', '')}",
            f"Description: {image_description.get('description', '')}",
        ]
        blocked = image_description.get("blocked_section")
        if blocked:
            visual_parts.append(f"Blocked section: {blocked}")
        stations = image_description.get("stations_or_junctions")
        if stations:
            if isinstance(stations, list):
                visual_parts.append(
                    "Stations/junctions: " + ", ".join(str(s) for s in stations)
                )
            else:
                visual_parts.append(f"Stations/junctions: {stations}")

        user_text = (
            f"Section: {section_title}\n\n"
            f"--- Slide text (authoritative) ---\n{slide_text.strip() or '(none)'}\n\n"
            f"--- Visual summary ---\n" + "\n".join(visual_parts) + f"\n\n{prompt}"
        )

        messages = [{"role": "user", "content": user_text}]
        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}, *messages]

        with OpenRouter(api_key=self.api_key) as client:
            response = client.chat.send(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )

        raw = _message_content_to_text(response.choices[0].message.content)
        data = _parse_json_response(raw)
        guide = str(data.get("guide", "")).strip()
        return {"guide": guide, "section": section_title}
