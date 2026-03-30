# Changelog

## v0.3 (alpha)
- optional `register_llm_checker(...)` 추가로 open-ended instruction verification 경로 도입
- OpenAI / Anthropic / Gemini provider-aware LLM judge 추가 (`src/memguard/llm.py`)
- LLM checker에 `rubric_template`, `uncertainty_threshold`, `local_fallback` 추가
- LLM checker에 optional `negative_recheck` confirmatory pass 추가
- `language_compliance`, `summary_first`, `brevity_limit`, `approval_before_action` rubric template 추가
- LLM 결과에 `uncertain` 상태와 hybrid fallback 추가
- `benchmarks/isb/run.py` 추가
- persistence/verification 2-track benchmark 구조 추가
- open-ended LLM benchmark dataset를 `benchmarks/isb/llm_cases.json`으로 분리하고 21개 케이스로 확장
- `uncertain_rate` 지표 추가
- `report()`와 benchmark 지표를 실제 observed compliance event 기준으로 정리
- `memorymesh` legacy shim 제거, `memguard` 단일 이름 체계로 정리
- provider-backed live LLM benchmark 경로 추가
- `python -m pytest tests -v` 기준 테스트 32개 통과

## v0.2
- 프로젝트 리네이밍: MemoryMesh → **MemGuard**
- 포지셔닝 전환: 통합 메모리 엔진 → **메모리 무결성 레이어**
- `MemoryGuard` 심플 API 신설 (protect/check/reminder/report)
- README 전면 개편
- 패키지명: `memorymesh` → `memguard`
- `check()` / `observe_response()`에 relevance gating 추가
- 무관한 protected rule은 `skip`, 룰 기반 체커가 없는 지시는 `unverified`로 분리
- custom checker registry 추가 (`register_checker`, `protect(..., checker=...)`)
- `observe_action()` 추가로 action-level verification 지원
- built-in approval-before-action / destructive git action checks 추가
