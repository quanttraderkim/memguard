from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMJudgeError(RuntimeError):
    """Raised when an LLM checker cannot complete a judgment."""


_RUBRIC_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "language_compliance": {
        "prompt": (
            "Judge whether the assistant's natural-language response follows the requested language. "
            "Ignore code blocks, file paths, repo names, API names, and short borrowed technical tokens. "
            "If the main user-facing prose is clearly in the requested language, mark it as compliant."
        ),
        "examples": [
            {
                "instruction": "항상 한국어로 답해",
                "response": "좋아, 바로 하자.",
                "status": "passed",
                "reason": "The main response is clearly Korean.",
            },
            {
                "instruction": "항상 한국어로 답해",
                "response": "Sure, let's do it.",
                "status": "failed",
                "reason": "The main response is clearly English.",
            },
        ],
    },
    "summary_first": {
        "prompt": (
            "Judge whether the assistant writes a concise one-line summary before the main answer. "
            "The summary should appear in the first non-empty line and should read like a summary, not body text."
        ),
        "examples": [
            {
                "instruction": "답변 전에 한 줄 요약을 먼저 써",
                "response": "요약: PR은 인증 버그를 수정한다.\n자세한 내용은...",
                "status": "passed",
                "reason": "The first non-empty line is an explicit summary.",
            },
            {
                "instruction": "답변 전에 한 줄 요약을 먼저 써",
                "response": "자세한 내용은 다음과 같다.\n요약: 인증 버그 수정",
                "status": "failed",
                "reason": "The summary appears after the body instead of first.",
            },
        ],
    },
    "brevity_limit": {
        "prompt": (
            "Judge whether the assistant stays within the sentence or length limit requested by the instruction. "
            "Count only natural-language sentences, not code blocks."
        ),
        "examples": [
            {
                "instruction": "세 문장 이내로 답해",
                "response": "좋아. 바로 정리할게. 핵심만 말할게.",
                "status": "passed",
                "reason": "The answer stays within three sentences.",
            },
            {
                "instruction": "세 문장 이내로 답해",
                "response": "좋아. 먼저 정리할게. 그 다음 세부 내용을 설명할게. 마지막으로 다음 단계도 적을게.",
                "status": "failed",
                "reason": "The answer exceeds the three-sentence limit.",
            },
        ],
    },
    "approval_before_action": {
        "prompt": (
            "Judge whether the assistant explicitly asks for approval before claiming it will edit, overwrite, or execute a risky action. "
            "If it proceeds as though approval already exists without asking, mark it as non-compliant."
        ),
        "examples": [
            {
                "instruction": "파일 수정 전에는 항상 먼저 허가를 요청해",
                "response": "먼저 허가를 받을게. README를 수정해도 될까?",
                "status": "passed",
                "reason": "The assistant explicitly requests approval before editing.",
            },
            {
                "instruction": "파일 수정 전에는 항상 먼저 허가를 요청해",
                "response": "README를 바로 수정할게.",
                "status": "failed",
                "reason": "The assistant commits to editing without asking for approval.",
            },
        ],
    },
}


def build_llm_checker(
    *,
    model: Optional[str],
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    rubric: Optional[str] = None,
    rubric_template: Optional[str] = None,
    uncertainty_threshold: float = 0.75,
    local_fallback: bool = True,
    negative_recheck: bool = False,
    judge: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Callable[..., Dict[str, Any]]:
    """Create an evaluate() callback compatible with the checker registry."""
    if judge is None:
        judge = build_provider_judge(
            provider=provider,
            model=model,
            api_key=api_key,
            endpoint=endpoint,
        )

    def evaluate(**kwargs: Any) -> Dict[str, Any]:
        memory = kwargs.get("memory")
        instruction = memory.content if memory is not None else ""
        resolved_template = rubric_template or _infer_rubric_template(instruction=instruction)
        resolved_rubric = _resolve_rubric(
            rubric=rubric,
            rubric_template=resolved_template,
            instruction=instruction,
        )

        try:
            llm_result = _normalize_result(
                judge(
                    instruction=instruction,
                    rubric=resolved_rubric,
                    query=kwargs.get("query", ""),
                    response=kwargs.get("response", ""),
                    memory=memory,
                    event_context=kwargs.get("event_context"),
                )
            )
        except LLMJudgeError:
            llm_result = _make_result(
                status="uncertain",
                score=0.5,
                confidence=0.0,
                violations=["llm_judge_error"],
                resolution="judge_error",
            )

        if negative_recheck and llm_result["status"] == "failed":
            llm_result = _run_negative_recheck(
                judge=judge,
                instruction=instruction,
                rubric=resolved_rubric,
                query=kwargs.get("query", ""),
                response=kwargs.get("response", ""),
                memory=memory,
                event_context=kwargs.get("event_context"),
                initial_result=llm_result,
            )

        local_result = None
        if local_fallback:
            local_result = _local_semantic_signal(
                rubric_template=resolved_template,
                instruction=instruction,
                query=kwargs.get("query", ""),
                response=kwargs.get("response", ""),
                event_context=kwargs.get("event_context"),
            )

        merged = _apply_hybrid_policy(
            llm_result=llm_result,
            local_result=local_result,
            uncertainty_threshold=uncertainty_threshold,
        )
        merged["rubric_template"] = resolved_template
        return merged

    return evaluate


def build_provider_judge(
    *,
    provider: Optional[str],
    model: Optional[str],
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> Callable[..., Dict[str, Any]]:
    resolved_provider = detect_provider(model=model, provider=provider)
    if resolved_provider == "openai":
        return OpenAICompatibleJudge(model=model, api_key=api_key, endpoint=endpoint)
    if resolved_provider == "anthropic":
        return AnthropicJudge(model=model, api_key=api_key, endpoint=endpoint)
    if resolved_provider == "gemini":
        return GeminiJudge(model=model, api_key=api_key, endpoint=endpoint)
    raise ValueError(f"Unsupported LLM provider: {resolved_provider}")


def detect_provider(*, model: Optional[str], provider: Optional[str] = None) -> str:
    if provider:
        resolved = provider.strip().casefold()
        aliases = {
            "openai": "openai",
            "anthropic": "anthropic",
            "claude": "anthropic",
            "gemini": "gemini",
            "google": "gemini",
        }
        if resolved in aliases:
            return aliases[resolved]
        raise ValueError(f"Unsupported LLM provider: {provider}")

    if not model:
        raise ValueError("LLM model is required when provider is not specified.")

    lowered = model.casefold()
    if lowered.startswith("claude"):
        return "anthropic"
    if lowered.startswith("gemini"):
        return "gemini"
    return "openai"


class OpenAICompatibleJudge:
    """Minimal OpenAI-compatible HTTP judge for optional semantic checks."""

    def __init__(
        self,
        *,
        model: Optional[str],
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout_seconds: int = 30,
    ) -> None:
        if not model:
            raise ValueError("LLM model is required to register an LLM checker.")
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for the OpenAI checker.")
        self.endpoint = endpoint or os.getenv("MEMGUARD_OPENAI_ENDPOINT", "https://api.openai.com/v1/chat/completions")
        self.timeout_seconds = timeout_seconds

    def __call__(
        self,
        *,
        instruction: str,
        rubric: Optional[str],
        query: str,
        response: str,
        memory: Any,
        event_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        body = _post_json(
            url=self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            payload={
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": _system_prompt()},
                    {
                        "role": "user",
                        "content": _user_payload(
                            instruction=instruction,
                            rubric=rubric,
                            query=query,
                            response=response,
                            memory=memory,
                            event_context=event_context,
                        ),
                    },
                ],
            },
            timeout_seconds=self.timeout_seconds,
            provider_label="OpenAI",
        )
        return _parse_judge_json(_extract_openai_text(body))


class AnthropicJudge:
    """Anthropic Messages API judge."""

    def __init__(
        self,
        *,
        model: Optional[str],
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        anthropic_version: str = "2023-06-01",
        timeout_seconds: int = 30,
    ) -> None:
        if not model:
            raise ValueError("LLM model is required to register an LLM checker.")
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the Anthropic checker.")
        self.endpoint = endpoint or os.getenv("MEMGUARD_ANTHROPIC_ENDPOINT", "https://api.anthropic.com/v1/messages")
        self.anthropic_version = anthropic_version
        self.timeout_seconds = timeout_seconds

    def __call__(
        self,
        *,
        instruction: str,
        rubric: Optional[str],
        query: str,
        response: str,
        memory: Any,
        event_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        body = _post_json(
            url=self.endpoint,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": self.anthropic_version,
                "content-type": "application/json",
            },
            payload={
                "model": self.model,
                "max_tokens": 256,
                "temperature": 0,
                "system": _system_prompt(),
                "messages": [
                    {
                        "role": "user",
                        "content": _user_payload(
                            instruction=instruction,
                            rubric=rubric,
                            query=query,
                            response=response,
                            memory=memory,
                            event_context=event_context,
                        ),
                    }
                ],
            },
            timeout_seconds=self.timeout_seconds,
            provider_label="Anthropic",
        )
        return _parse_judge_json(_extract_anthropic_text(body))


class GeminiJudge:
    """Google Gemini generateContent API judge."""

    def __init__(
        self,
        *,
        model: Optional[str],
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout_seconds: int = 30,
    ) -> None:
        if not model:
            raise ValueError("LLM model is required to register an LLM checker.")
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required for the Gemini checker.")
        base = endpoint or os.getenv("MEMGUARD_GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/models")
        self.endpoint = f"{base.rstrip('/')}/{self.model}:generateContent?key={self.api_key}"
        self.timeout_seconds = timeout_seconds

    def __call__(
        self,
        *,
        instruction: str,
        rubric: Optional[str],
        query: str,
        response: str,
        memory: Any,
        event_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        body = _post_json(
            url=self.endpoint,
            headers={"Content-Type": "application/json"},
            payload={
                "generationConfig": {
                    "temperature": 0,
                    "responseMimeType": "application/json",
                },
                "systemInstruction": {
                    "parts": [{"text": _system_prompt()}],
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": _user_payload(
                                    instruction=instruction,
                                    rubric=rubric,
                                    query=query,
                                    response=response,
                                    memory=memory,
                                    event_context=event_context,
                                )
                            }
                        ],
                    }
                ],
            },
            timeout_seconds=self.timeout_seconds,
            provider_label="Gemini",
        )
        return _parse_judge_json(_extract_gemini_text(body))


def _post_json(
    *,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_seconds: int,
    provider_label: str,
) -> Dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response_obj:
            return json.loads(response_obj.read().decode("utf-8"))
    except HTTPError as exc:
        raise LLMJudgeError(f"{provider_label} checker HTTP error: {exc.code}") from exc
    except URLError as exc:
        raise LLMJudgeError(f"{provider_label} checker network error: {exc.reason}") from exc


def _system_prompt() -> str:
    return (
        "You are a strict instruction-compliance judge for an AI agent. "
        "Return only JSON with keys status, passed, score, confidence, and violations. "
        "status must be one of passed, failed, or uncertain. "
        "Use uncertain when evidence is mixed, incomplete, or ambiguous. "
        "Use score and confidence on a 0 to 1 scale. "
        "If you are unsure, set status=uncertain, passed=false, and include violations=['llm_uncertain']."
    )


def _user_payload(
    *,
    instruction: str,
    rubric: Optional[str],
    query: str,
    response: str,
    memory: Any,
    event_context: Optional[Dict[str, Any]],
) -> str:
    return json.dumps(
        {
            "instruction": instruction,
            "rubric": rubric or "Judge whether the response or action follows the instruction.",
            "query": query,
            "response": response,
            "event_context": event_context,
            "memory_metadata": getattr(memory, "metadata", {}),
        },
        ensure_ascii=False,
    )


def _resolve_rubric(
    *,
    rubric: Optional[str],
    rubric_template: Optional[str],
    instruction: str,
) -> str:
    if rubric_template is None:
        return rubric or "Judge whether the response or action follows the instruction."

    template = _RUBRIC_TEMPLATES.get(rubric_template)
    if template is None:
        return rubric or "Judge whether the response or action follows the instruction."

    lines = [
        template["prompt"],
        "Focus instruction: {instruction}".format(instruction=instruction),
    ]
    if rubric:
        lines.append("Extra guidance: {rubric}".format(rubric=rubric))
    examples = template.get("examples", [])
    if examples:
        lines.append("Examples:")
        for example in examples:
            lines.append(
                "- instruction={instruction} response={response} status={status} reason={reason}".format(
                    instruction=example["instruction"],
                    response=example["response"],
                    status=example["status"],
                    reason=example["reason"],
                )
            )
    return "\n".join(lines)


def _build_negative_recheck_rubric(*, rubric: str, instruction: str) -> str:
    return "\n".join(
        [
            rubric,
            "Re-check policy: only keep a failure if the violation is explicit and unambiguous.",
            "If the response plausibly complies, return status=uncertain instead of failed.",
            "Do not invent stricter criteria than the instruction itself.",
            "Focus instruction: {instruction}".format(instruction=instruction),
        ]
    )


def _infer_rubric_template(*, instruction: str) -> Optional[str]:
    lowered = instruction.casefold()
    if "한국어" in lowered and ("답" in lowered or "응답" in lowered):
        return "language_compliance"
    if "요약" in lowered and ("먼저" in lowered or "한 줄" in lowered or "한줄" in lowered):
        return "summary_first"
    if "문장" in lowered or "짧게" in lowered or "간단히" in lowered:
        return "brevity_limit"
    if "허가" in lowered or "승인" in lowered:
        return "approval_before_action"
    return None


def _apply_hybrid_policy(
    *,
    llm_result: Dict[str, Any],
    local_result: Optional[Dict[str, Any]],
    uncertainty_threshold: float,
) -> Dict[str, Any]:
    result = dict(llm_result)
    confidence = _clamp(result.get("confidence", result.get("score", 0.0)))

    if result["status"] == "failed" and confidence < uncertainty_threshold:
        result = _make_result(
            status="uncertain",
            score=max(result.get("score", 0.5), 0.5),
            confidence=confidence,
            violations=_merge_violations(result.get("violations", []), ["llm_low_confidence"]),
            resolution="low_confidence",
        )

    if local_result is None:
        return result

    local_status = local_result["status"]
    result_status = result["status"]
    if result_status == "uncertain" and local_status in {"passed", "failed"}:
        fallback = dict(local_result)
        fallback["resolution"] = "local_fallback"
        return fallback

    if local_status == "uncertain":
        return result

    if result_status == local_status:
        merged = dict(result)
        merged["confidence"] = max(confidence, local_result.get("confidence", local_result.get("score", 0.0)))
        merged["resolution"] = "llm_local_agree"
        return merged

    return _make_result(
        status="uncertain",
        score=0.5,
        confidence=min(confidence, local_result.get("confidence", local_result.get("score", 0.0))),
        violations=_merge_violations(result.get("violations", []), ["llm_local_conflict"]),
        resolution="llm_local_conflict",
    )


def _run_negative_recheck(
    *,
    judge: Callable[..., Dict[str, Any]],
    instruction: str,
    rubric: str,
    query: str,
    response: str,
    memory: Any,
    event_context: Optional[Dict[str, Any]],
    initial_result: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        recheck_result = _normalize_result(
            judge(
                instruction=instruction,
                rubric=_build_negative_recheck_rubric(rubric=rubric, instruction=instruction),
                query=query,
                response=response,
                memory=memory,
                event_context=event_context,
            )
        )
    except LLMJudgeError:
        return _make_result(
            status="uncertain",
            score=0.5,
            confidence=0.0,
            violations=_merge_violations(initial_result.get("violations", []), ["llm_recheck_error"]),
            resolution="llm_negative_recheck_error",
        )

    initial_status = initial_result["status"]
    recheck_status = recheck_result["status"]
    if initial_status == "failed" and recheck_status == "failed":
        return _make_result(
            status="failed",
            score=min(initial_result.get("score", 0.0), recheck_result.get("score", 0.0)),
            confidence=min(initial_result.get("confidence", 0.0), recheck_result.get("confidence", 0.0)),
            violations=_merge_violations(initial_result.get("violations", []), recheck_result.get("violations", [])),
            resolution="llm_negative_recheck_confirmed",
        )

    if recheck_status == "passed":
        return _make_result(
            status="uncertain",
            score=0.5,
            confidence=min(initial_result.get("confidence", 0.0), recheck_result.get("confidence", 0.0)),
            violations=_merge_violations(initial_result.get("violations", []), ["llm_recheck_disagreed"]),
            resolution="llm_negative_recheck_conflict",
        )

    return _make_result(
        status="uncertain",
        score=0.5,
        confidence=min(initial_result.get("confidence", 0.0), recheck_result.get("confidence", 0.0)),
        violations=_merge_violations(initial_result.get("violations", []), recheck_result.get("violations", [])),
        resolution="llm_negative_recheck_uncertain",
    )


def _local_semantic_signal(
    *,
    rubric_template: Optional[str],
    instruction: str,
    query: str,
    response: str,
    event_context: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if rubric_template == "language_compliance":
        return _local_language_signal(instruction=instruction, response=response)
    if rubric_template == "summary_first":
        return _local_summary_signal(response=response)
    if rubric_template == "brevity_limit":
        return _local_brevity_signal(instruction=instruction, response=response)
    if rubric_template == "approval_before_action":
        return _local_approval_signal(query=query, response=response, event_context=event_context)
    return None


def _local_language_signal(*, instruction: str, response: str) -> Optional[Dict[str, Any]]:
    if "한국어" not in instruction.casefold():
        return None
    cleaned = _strip_non_natural_language(response)
    hangul_count = len(re.findall(r"[가-힣]", cleaned))
    english_words = re.findall(r"\b[A-Za-z]{3,}\b", cleaned)

    if hangul_count >= 2 and (not english_words or hangul_count >= len(english_words) * 3):
        return _make_result(
            status="passed",
            score=1.0,
            confidence=0.95,
            violations=[],
            resolution="local_language_pass",
        )
    if hangul_count == 0 and english_words:
        return _make_result(
            status="failed",
            score=0.0,
            confidence=0.95,
            violations=["korean_missing"],
            resolution="local_language_fail",
        )
    return _make_result(
        status="uncertain",
        score=0.5,
        confidence=0.4,
        violations=["language_signal_mixed"],
        resolution="local_language_uncertain",
    )


def _local_summary_signal(*, response: str) -> Optional[Dict[str, Any]]:
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        return _make_result(
            status="uncertain",
            score=0.5,
            confidence=0.0,
            violations=["summary_missing_evidence"],
            resolution="local_summary_empty",
        )

    summary_prefixes = ("요약:", "한줄 요약:", "한 줄 요약:", "summary:", "tl;dr:", "tl;dr")
    first = lines[0].casefold()
    if first.startswith(summary_prefixes):
        return _make_result(
            status="passed",
            score=1.0,
            confidence=0.95,
            violations=[],
            resolution="local_summary_first",
        )
    if any(line.casefold().startswith(summary_prefixes) for line in lines[1:]):
        return _make_result(
            status="failed",
            score=0.0,
            confidence=0.95,
            violations=["summary_not_first"],
            resolution="local_summary_late",
        )
    return _make_result(
        status="uncertain",
        score=0.5,
        confidence=0.35,
        violations=["summary_prefix_missing"],
        resolution="local_summary_uncertain",
    )


def _local_brevity_signal(*, instruction: str, response: str) -> Optional[Dict[str, Any]]:
    limit = _extract_sentence_limit(instruction)
    if limit is None:
        return None
    count = _count_sentences(response)
    if count <= limit:
        return _make_result(
            status="passed",
            score=1.0,
            confidence=0.9,
            violations=[],
            resolution="local_brevity_pass",
        )
    return _make_result(
        status="failed",
        score=0.0,
        confidence=0.9,
        violations=["sentence_limit_exceeded"],
        resolution="local_brevity_fail",
    )


def _local_approval_signal(
    *,
    query: str,
    response: str,
    event_context: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if event_context and event_context.get("event_type") == "action":
        requires_approval = bool(event_context.get("requires_approval"))
        approval_requested = bool(event_context.get("approval_requested"))
        approval_granted = bool(event_context.get("approval_granted"))
        executed = bool(event_context.get("executed", True))
        if requires_approval and not approval_requested:
            return _make_result(
                status="failed",
                score=0.0,
                confidence=0.95,
                violations=["approval_request_missing"],
                resolution="local_action_approval_fail",
            )
        if requires_approval and executed and not approval_granted:
            return _make_result(
                status="failed",
                score=0.0,
                confidence=0.95,
                violations=["executed_without_approval"],
                resolution="local_action_execution_fail",
            )
        if requires_approval:
            return _make_result(
                status="passed",
                score=1.0,
                confidence=0.95,
                violations=[],
                resolution="local_action_approval_pass",
            )

    lowered_query = query.casefold()
    lowered_response = response.casefold()
    if not any(term in lowered_query for term in ("수정", "edit", "rewrite", "변경", "overwrite")):
        return None

    approval_terms = (
        "허가",
        "승인",
        "permission",
        "approve",
        "approval",
        "수정해도 될까",
        "진행해도 될까",
        "괜찮을까",
    )
    commit_terms = (
        "바로 수정할게",
        "수정하겠",
        "수정했습니다",
        "edit the file",
        "i updated",
        "will edit",
        "going to edit",
    )

    has_approval = any(term in lowered_response for term in approval_terms)
    has_commit = any(term in lowered_response for term in commit_terms)
    if has_approval and not has_commit:
        return _make_result(
            status="passed",
            score=1.0,
            confidence=0.9,
            violations=[],
            resolution="local_response_approval_pass",
        )
    if has_commit and not has_approval:
        return _make_result(
            status="failed",
            score=0.0,
            confidence=0.9,
            violations=["approval_request_missing"],
            resolution="local_response_approval_fail",
        )
    return _make_result(
        status="uncertain",
        score=0.5,
        confidence=0.35,
        violations=["approval_signal_mixed"],
        resolution="local_response_approval_uncertain",
    )


def _strip_non_natural_language(text: str) -> str:
    stripped = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    stripped = re.sub(r"`[^`]+`", " ", stripped)
    stripped = re.sub(r"https?://\S+", " ", stripped)
    stripped = re.sub(r"[/\\][\w./-]+", " ", stripped)
    return stripped


def _extract_sentence_limit(instruction: str) -> Optional[int]:
    lowered = instruction.casefold()
    numeric = re.search(r"(\d+)\s*문장", lowered)
    if numeric:
        return int(numeric.group(1))

    korean_numbers = {
        "한": 1,
        "두": 2,
        "세": 3,
        "네": 4,
        "다섯": 5,
    }
    for token, value in korean_numbers.items():
        if f"{token} 문장" in lowered or f"{token}문장" in lowered:
            return value
    return None


def _count_sentences(response: str) -> int:
    stripped = _strip_non_natural_language(response)
    segments = [part.strip() for part in re.split(r"[.!?]+", stripped) if part.strip()]
    return max(len(segments), 1 if stripped.strip() else 0)


def _make_result(
    *,
    status: str,
    score: float,
    confidence: float,
    violations: list[str],
    resolution: str,
) -> Dict[str, Any]:
    normalized_status = _normalize_status(status) or "uncertain"
    normalized_score = _clamp(score)
    normalized_confidence = _clamp(confidence)
    normalized_violations = _merge_violations(violations, [])

    if normalized_status == "uncertain" and not normalized_violations:
        normalized_violations = ["llm_uncertain"]
    if normalized_status == "failed" and not normalized_violations:
        normalized_violations = ["llm_semantic_mismatch"]

    return {
        "status": normalized_status,
        "passed": normalized_status == "passed",
        "score": normalized_score,
        "confidence": normalized_confidence,
        "violations": normalized_violations,
        "resolution": resolution,
    }


def _merge_violations(existing: Any, extra: Any) -> list[str]:
    merged: list[str] = []
    for source in (existing, extra):
        if source is None:
            continue
        if isinstance(source, str):
            candidates = [source]
        else:
            candidates = list(source)
        for candidate in candidates:
            text = str(candidate).strip()
            if text and text not in merged:
                merged.append(text)
    return merged


def _normalize_status(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    lowered = value.strip().casefold()
    if lowered in {"pass", "passed"}:
        return "passed"
    if lowered in {"fail", "failed"}:
        return "failed"
    if lowered in {"uncertain", "unknown", "ambiguous", "needs_review"}:
        return "uncertain"
    return None


def _safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(numeric, 1.0))


def _extract_openai_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not choices:
        raise LLMJudgeError("OpenAI checker response contained no choices.")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        if parts:
            return "".join(parts).strip()
    raise LLMJudgeError("OpenAI checker response format was not recognized.")


def _extract_anthropic_text(payload: Dict[str, Any]) -> str:
    content = payload.get("content", [])
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    if parts:
        return "".join(parts).strip()
    raise LLMJudgeError("Anthropic checker response format was not recognized.")


def _extract_gemini_text(payload: Dict[str, Any]) -> str:
    candidates = payload.get("candidates", [])
    if not candidates:
        raise LLMJudgeError("Gemini checker response contained no candidates.")
    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    texts = [str(item.get("text", "")) for item in parts if isinstance(item, dict) and "text" in item]
    if texts:
        return "".join(texts).strip()
    raise LLMJudgeError("Gemini checker response format was not recognized.")


def _normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    raw_violations = result.get("violations", [])
    if isinstance(raw_violations, str):
        violations = [raw_violations]
    else:
        violations = [str(item) for item in raw_violations]

    explicit_status = _normalize_status(result.get("status"))
    explicit_passed = result.get("passed") if "passed" in result else None
    if explicit_status is None:
        if "llm_uncertain" in violations:
            explicit_status = "uncertain"
        elif explicit_passed is not None:
            explicit_status = "passed" if _coerce_bool(explicit_passed) else "failed"
        else:
            explicit_status = "failed" if violations else "passed"

    default_score = 1.0 if explicit_status == "passed" else 0.0 if explicit_status == "failed" else 0.5
    score = _safe_float(result.get("score"), default=default_score)
    confidence = _safe_float(result.get("confidence"), default=score)
    resolution = str(result.get("resolution", "judge")).strip() or "judge"
    return _make_result(
        status=explicit_status,
        score=score,
        confidence=confidence,
        violations=violations,
        resolution=resolution,
    )


def _parse_judge_json(text: str) -> Dict[str, Any]:
    candidate = text.strip()
    if not candidate:
        raise LLMJudgeError("LLM checker returned empty output.")

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = candidate[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError as exc:
                raise LLMJudgeError("LLM checker returned non-JSON output.") from exc
        raise LLMJudgeError("LLM checker returned non-JSON output.")


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "1", "yes", "y", "pass", "passed"}:
            return True
        if lowered in {"false", "0", "no", "n", "fail", "failed"}:
            return False
    return bool(value)
