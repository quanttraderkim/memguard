# MemGuard — Product Requirements Document
**Version:** v0.2 Draft  
**Author:** Daehwan Kim  
**Date:** 2026-03-29

---

## 1. 문제 정의

### 실제로 유저가 겪는 문제
오늘의 AI 에이전트는 "기억을 저장"할 수는 있어도, 그 기억이 시간이 지나도 **행동으로 유지되는지**는 보장하지 못한다.

대표적인 실패 패턴:

```text
유저: 항상 반말로 대답해
→ 메모리에 저장됨
→ 대화가 길어짐
→ 세션 요약 / 컨텍스트 압축 / 리랭킹 발생
→ 해당 지시가 현재 컨텍스트에서 밀려남
→ 모델이 다시 존댓말로 응답
→ 유저: "이거 하지 말라고 했잖아"
```

이 문제는 단순 저장 문제가 아니다.  
핵심은 **instruction persistence**, 즉 "중요한 유저 지시와 핵심 선호가 시간이 지나도 계속 행동에 반영되는가"의 문제다.

### 기존 솔루션의 한계
- 대부분의 memory 솔루션은 저장과 검색에 집중한다.
- 그러나 유저가 실제로 원하는 것은 "기억이 남아 있다"가 아니라 "AI가 계속 기억하고 행동한다"이다.
- 메모리에 저장된 사실이 있어도, 컨텍스트 구성 단계에서 빠지면 사용자 경험은 실패로 인식된다.

### MemGuard가 해결하려는 것
MemGuard는 범용 "기억 데이터베이스"가 아니라, AI 에이전트가 **잊으면 안 되는 지시와 핵심 기억을 끝까지 붙잡아 두는 persistence layer**다.

---

## 2. 한 줄 정의

> **MemGuard는 AI가 진짜로 기억하고 있는지 검증하는 instruction persistence layer다.**

다른 솔루션이 저장과 검색을 제공한다면, MemGuard는 그 위에 다음을 추가한다:

- 어떤 기억이 절대 밀리면 안 되는지 분류
- 매 호출 전에 보호된 기억을 컨텍스트에 강제 주입
- 기억이 실제 응답 행동에 반영되는지 검증
- 시간이 지나며 지시 준수율이 떨어지는 drift를 감지

---

## 3. 제품 포지셔닝

### Before
AI agent memory = "기억을 저장하고 필요할 때 검색하는 시스템"

### After
AI agent memory = "핵심 지시와 정체성을 지속적으로 유지하고, 그 유지 상태를 검증하는 시스템"

### 포지셔닝 문구
- "Instruction persistence for AI agents"
- "Protected memory layer for long-running agents"
- "Store, protect, verify"

### 비포지셔닝
MemGuard는 초기 버전에서 아래를 목표로 하지 않는다.
- 범용 knowledge graph 플랫폼
- 대규모 엔터프라이즈 메모리 운영 플랫폼
- 모든 종류의 agent memory use case를 다 포괄하는 프레임워크

초기 집중 영역은 명확하다:

> **코딩 에이전트 / 개인 AI 비서 / 장시간 대화형 에이전트에서 유저 지시가 드리프트되는 문제 해결**

---

## 4. 핵심 인사이트

### 저장과 보존은 다르다
메모리에 들어갔다는 사실만으로는 충분하지 않다.

실패는 주로 아래 단계에서 발생한다:

1. 기억 저장은 됨
2. 관련도 기반 검색 또는 세션 압축 발생
3. 핵심 지시가 현재 prompt 조합에서 탈락
4. 모델 행동이 유저 의도와 다시 어긋남

따라서 제품의 중심은 `memory store`가 아니라 `context curation`이어야 한다.

### 인간 기억 비유의 올바른 사용
인지과학 비유는 유효하지만, 제품 설계는 검증 가능해야 한다.

| 인간 기억 시스템 | MemGuard |
|------------------|------------|
| 편도체: 강한 감정은 오래 남음 | 중요도 분류: 유저 교정, 금지, 선호는 상위 우선순위 |
| 해마: 반복은 장기기억화 | Reinforcement: 같은 지시가 반복되면 우선순위 상승 |
| 전전두엽: 작업기억 관리 | Context Curator: 현재 호출에 필요한 보호 기억 선별 |
| 수면: 불필요한 기억 정리 | Consolidation: 유휴 시 병합, 정리, 승격 |

핵심은 "뇌처럼 보이게"가 아니라, **유저가 체감하는 실패를 줄이는 방향으로 기억을 운영하는 것**이다.

---

## 5. 핵심 개념

### 5.1 Memory Priority

초기 버전의 핵심은 계층보다 **우선순위**다.

```python
class MemoryPriority:
    CORE = 1       # 절대 밀리면 안 되는 지시, 금지사항, 포맷 규칙
    IDENTITY = 2   # 이름, 관계, 역할, 말투, 장기 선호
    PROJECT = 3    # 현재 진행 중인 프로젝트 규칙과 의사결정
    EPISODE = 4    # 특정 시점의 사건, 작업 로그, 일회성 문맥
    TRIVIAL = 5    # 중요하지 않은 소소한 대화
```

### 5.2 Protected Memory
`CORE`와 일부 `IDENTITY` 기억은 일반 검색 결과와 별도로 관리한다.

이 기억은 매 LLM 호출 전에 별도 보호 구역으로 prompt에 포함된다.

중요한 점은 protected memory가 "무한 보존"을 뜻하지는 않는다는 것이다.

- protected 내부에서도 topic 단위 최신 우선 규칙이 적용된다.
- 상충하는 instruction은 `supersedes` 관계 또는 최신 지시 우선으로 정리된다.
- 보호 기억 수가 일정 예산을 넘으면, 세부 항목을 유지한 채 상위 summary memory로 압축할 수 있다.
- 즉 protected zone은 삭제 불가 영역이 아니라, **최우선 보존 + 내부 정리 규칙이 있는 영역**이다.

### 5.3 Instruction Persistence
중요한 유저 지시가 여러 턴과 여러 세션을 거쳐도 살아남는 상태를 의미한다.

예시:
- "항상 반말로 대답해"
- "코드 블록엔 항상 언어 태그를 붙여"
- "추측하지 말고 모르면 모른다고 말해"
- "내가 요청하지 않으면 git commit 하지 마"

### 5.4 Integrity Verification
기억이 저장돼 있는지 확인하는 것이 아니라, **실제로 동작하고 있는지**를 점검한다.

### 5.5 Drift Detection
시간이 지나며 특정 지시의 준수율이 서서히 떨어지는 현상을 탐지한다.

---

## 6. 사용자 시나리오

### 시나리오 1: 코딩 에이전트의 응답 스타일 유지

```python
mem.remember(
    "항상 코드 블록에 언어 태그를 붙여",
    priority="core",
    kind="instruction"
)

context = mem.build_context("FastAPI 예제 보여줘")
```

결과:
- 해당 지시는 일반 검색 결과가 아니라 protected zone에 들어간다.
- 세션 요약이나 오래된 대화 압축과 무관하게 유지된다.

### 시나리오 2: 금지사항 유지

```python
mem.remember(
    "사용자가 명시하지 않으면 destructive git command를 실행하지 마",
    priority="core",
    kind="guardrail"
)
```

결과:
- 이후 대화 길이와 무관하게 에이전트 실행 정책에 지속 반영된다.
- 동일 지시가 여러 번 등장하면 reinforcement 점수가 올라간다.

### 시나리오 3: 프로젝트 규칙 유지

```python
mem.remember(
    "이 저장소는 pnpm을 사용하고, 테스트는 vitest로 돌린다",
    priority="project",
    metadata={"repo": "acme/web-app"}
)
```

결과:
- 현재 저장소와 관련된 요청일 때 active zone에 우선 포함된다.
- 다른 프로젝트에서는 자동으로 약화되거나 제외된다.

### 시나리오 4: 준수율 검증

```python
report = mem.verify()
```

예상 결과:

```json
{
  "total_memories": 142,
  "protected_memories": 12,
  "integrity": {
    "protected_loaded": 12,
    "instruction_checks_passed": 9,
    "instruction_checks_failed": 1,
    "conflicts_detected": 1
  }
}
```

---

## 7. 제품 원칙

### 원칙 1: 중요한 기억은 검색에만 맡기지 않는다
핵심 지시를 vector similarity 결과에만 의존하면 언젠가 밀린다.  
핵심 기억은 별도 보호 정책이 필요하다.

### 원칙 2: 기억은 저장보다 행동 반영이 중요하다
메모리 DB에 존재하는 사실보다, 실제 응답에서 얼마나 준수되는지가 더 중요하다.

### 원칙 3: 기본값은 로컬 우선이어야 한다
기억 내용과 운영 상태는 로컬에 저장할 수 있어야 한다.  
외부 LLM은 선택 옵션이어야 하며, 완전 로컬 모드가 존재해야 한다.

### 원칙 4: 시스템은 설명 가능해야 한다
왜 어떤 기억이 보호되었는지, 왜 충돌이 났는지, 왜 드리프트로 판단했는지를 보고할 수 있어야 한다.

### 원칙 5: 제품 범위는 작게 시작한다
초기 버전은 "instruction persistence for agents"를 해결한다.  
그래프, 복잡한 관계 추론, 대규모 멀티 테넌시는 후순위다.

---

## 8. 핵심 기능

### 8.1 Memory Guardian

Memory Guardian은 매 LLM 호출 전에 컨텍스트를 세 구역으로 구성한다.

```text
Protected Zone
- CORE instruction
- guardrail
- identity

Active Zone
- 현재 요청과 관련된 project memory
- 최근 에피소드

Buffer Zone
- 오래된 대화 요약
- 중요도 낮은 에피소드
```

규칙:
- Protected Zone은 토큰 예산 내에서 항상 우선 포함된다.
- Protected Zone 내부 충돌 시 최신 지시 또는 명시적 override rule을 적용한다.
- Active Zone은 관련도 기반으로 채운다.
- Buffer Zone은 예산 부족 시 가장 먼저 압축 또는 제거한다.

추가 운영 규칙:
- Protected Zone은 전체 토큰 예산의 고정 비율 상한을 가진다. 초기 기본값은 40~50% 범위로 둔다.
- 동일 topic의 instruction이 여러 개 쌓이면 최신 지시를 effective memory로 두고, 이전 항목은 history 또는 superseded 상태로 내린다.
- IDENTITY와 PROJECT memory는 장문 원문을 그대로 싣기보다 summary card 형태로 압축될 수 있다.
- verify 단계에서는 "보호 기억이 로드되었는가"뿐 아니라 "보호 기억이 과도하게 커져 active zone을 잠식하고 있는가"도 점검한다.

### 8.2 Reinforcement Engine

동일하거나 유사한 지시가 반복되면 해당 기억의 persistence score를 올린다.

예시:
- "반말로 해"가 3회 이상 반복되면 `CORE` 승격 후보
- "코드 블록에 언어 태그"가 여러 세션에 걸쳐 반복되면 보호 기억으로 승격

### 8.3 Consolidation

유휴 시 아래 작업을 수행한다.
- 유사 instruction 병합
- 중복 project memory 정리
- 충돌 후보 탐지
- stale episode 약화

### 8.4 Integrity Test

`verify()`는 시스템 수준 무결성을 점검한다.

검증 항목:
- 보호 기억이 실제로 context builder에 포함되는가
- instruction 충돌이 탐지되는가
- 보호 기억이 너무 많아 토큰 예산을 잠식하고 있지는 않은가
- 특정 instruction의 상태가 active, stale, conflicted 중 무엇인가

초기 구현 원칙:
- `verify()`는 기본적으로 로컬 구조 검사로 동작한다.
- LLM judge는 기본 경로가 아니라 선택형 adapter다.
- 즉 무결성 검사의 1차 책임은 rule-based local checker가 진다.

### 8.5 Replay Test

`replay()`는 특정 기억이 행동에 영향을 주는지 시뮬레이션한다.

예시:

```python
mem.replay(
    memory_id="instr_123",
    test_input="파이썬 예제 보여줘",
    expected_check="contains fenced code block with language tag"
)
```

출력:
- pass / fail
- 실패 이유
- 관련 충돌 기억
- 현재 보호 상태

초기 구현 원칙:
- `replay()`는 먼저 deterministic rule-based checker로 시작한다.
- 예: 언어 태그 존재 여부, 금지 명령어 포함 여부, 반말/존댓말 marker 검사
- 더 애매한 자연어 규칙만 선택적으로 LLM judge를 붙인다.
- 따라서 replay는 기본적으로 local-first이며, 비용이 큰 모델 평가는 후순위다.

### 8.6 Drift Detection

`detect_drift()`는 instruction별 준수 추세를 추적한다.

예시 출력:

```json
{
  "drifting": [
    {
      "instruction": "항상 반말로 대답해",
      "compliance_trend": [1.0, 1.0, 0.75, 0.5],
      "status": "warning"
    }
  ]
}
```

---

## 9. 아키텍처

### 시스템 구조

```text
AI Agent / App
    |
    v
MemGuard API
    |
    +-- Ingestion
    |     - instruction detection
    |     - priority classification
    |     - reinforcement update
    |
    +-- Memory Store
    |     - SQLite metadata
    |     - vector store (optional)
    |     - local JSON export
    |
    +-- Context Curator
    |     - protected zone builder
    |     - active memory selection
    |     - token budgeting
    |
    +-- Integrity Engine
    |     - verify
    |     - replay
    |     - detect_drift
    |
    +-- Consolidation
          - merge
          - conflict detection
          - stale cleanup
```

### 저장 구조

초기 버전은 지나치게 많은 스토어 조합을 지원하지 않는다.

- 기본 메타 저장소: SQLite
- 기본 로컬 직렬화: JSON
- 선택 벡터 검색: Chroma 또는 in-process embedding index
- 외부 graph store는 v0.x 범위에서 필수 아님
- 기본 검증기: rule-based local checker
- 선택 검증기: LLM-based judge adapter

이 선택은 제품 초점을 "기억 보존"에 두기 위한 의도적인 축소다.

---

## 10. API 초안

```python
class Memory:
    def __init__(
        self,
        agent_id: str,
        storage_path: str | None = None,
        mode: str = "local",
        llm: str | None = None,
    ):
        ...

    def remember(
        self,
        content: str,
        *,
        priority: str | None = None,
        kind: str | None = None,
        source: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        ...

    def build_context(
        self,
        query: str,
        *,
        token_budget: int = 4000,
        include_buffer: bool = True,
    ) -> dict:
        ...

    def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        include_protected: bool = True,
    ) -> list:
        ...

    def verify(self) -> dict:
        ...

    def replay(
        self,
        *,
        memory_id: str,
        test_input: str,
        expected_check: str,
    ) -> dict:
        ...

    def observe_response(
        self,
        *,
        query: str,
        response: str,
    ) -> dict:
        ...

    def detect_drift(self) -> dict:
        ...

    def consolidate(self) -> dict:
        ...

    def forget(self, memory_id: str) -> None:
        ...

    def export(self, path: str) -> None:
        ...
```

### 핵심 동작 원칙
- `remember()`는 기본적으로 instruction/identity/project/episode를 분류한다.
- `build_context()`는 protected, active, buffer 영역을 분리해 반환한다.
- `verify()`는 LLM 호출 없이도 가능한 구조적 무결성 검사를 우선 수행한다.
- `replay()`는 초기 버전에서 deterministic rule-based checker 또는 테스트용 model adapter를 사용할 수 있다.
- `observe_response()`는 실제 agent 응답을 같은 compliance event 포맷으로 기록한다.
- `detect_drift()`는 replay/observe 이력을 기반으로 준수율 하락 추세를 계산한다.
- 기본 경로는 local-first이며, 외부 모델 호출은 선택 옵션이다.

---

## 11. 데이터 모델

```python
@dataclass
class MemoryItem:
    id: str
    content: str
    priority: str                # core | identity | project | episode | trivial
    kind: str                    # instruction | preference | guardrail | fact | summary
    status: str                  # active | stale | conflicted | archived
    source: str                  # conversation | manual | import | system
    reinforcement_score: float
    persistence_score: float
    last_used_at: datetime | None
    created_at: datetime
    updated_at: datetime
    metadata: dict
```

추가 인덱스:
- `conflicts_with`
- `supersedes`
- `scope` (global / user / project / repo / session)

---

## 12. MVP 범위

### v0.1 목표
MemGuard v0.1은 아래만 해결한다.

1. 유저 instruction을 기억으로 저장할 수 있다.
2. 핵심 instruction을 protected memory로 승격할 수 있다.
3. 매 호출 전에 protected + active + buffer 컨텍스트를 구성할 수 있다.
4. `verify()`로 보호 기억 포함 여부와 충돌 여부를 점검할 수 있다.
5. 로컬 파일 기반 또는 SQLite 기반으로 완전 로컬 동작이 가능하다.
6. deterministic rule-based `replay()`로 핵심 instruction 준수 여부를 재현 검증할 수 있다.

### v0.1에서 의도적으로 제외
- knowledge graph
- 복잡한 relation extraction
- hosted service
- multi-tenant shared memory
- advanced semantic conflict resolution
- UI dashboard

---

## 13. 로드맵

### Phase 1: Instruction Persistence MVP

**v0.1**
- remember
- build_context
- protected memory policy
- verify
- SQLite/JSON local store

**v0.2**
- reinforcement scoring
- consolidate
- protected zone budget management
- supersede / overwrite / summary policy
- import/export

**v0.3**
- drift detection
- simple compliance history
- prompt interceptor / middleware example
- framework adapters for popular agent stacks

### Phase 2: Broader Agent Memory

**v0.4+**
- project-scoped memory packs
- richer retrieval
- optional graph relationship layer
- dashboard and inspection tooling

---

## 14. 성공 지표

### 제품 지표
- 핵심 instruction이 보호 영역에 포함되는 비율: 99%+
- instruction conflict 탐지율: 초기 golden dataset 기준 90%+
- `build_context()` 로컬 응답 시간: 50ms 미만
- `verify()` 구조 검사 시간: 100ms 미만

### 사용자 경험 지표
- "이거 하지 말라고 했잖아" 유형 실패율 감소
- 동일 세션 100턴 이후 instruction persistence 유지
- 장기 사용 시 준수율 하락 구간 탐지 가능

### 오픈소스 지표
- 3개월 내 초기 사용자 20명
- 실제 사용 사례 기반 이슈/피드백 30건 이상
- 첫 번째 외부 integration example 3개 이상

---

## 15. 경쟁 구도

| 솔루션 | 저장 | 검색 | 보호 기억 | 무결성 검증 | 드리프트 감지 |
|--------|------|------|-----------|-------------|---------------|
| mem0 | ✅ | ✅ | △ | ❌ | ❌ |
| Zep / Graph memory 계열 | ✅ | ✅ | △ | ❌ | ❌ |
| Letta | ✅ | ✅ | △ pinned memory | ❌ | ❌ |
| MemGuard | ✅ | ✅ | ✅ | ✅ | ✅ |

주의:
- 경쟁사도 일부 유사 개념을 추가할 수 있다.
- 따라서 차별화는 개념 이름이 아니라, 실제로 동작하는 `verify/replay/drift` 경험에서 나와야 한다.

---

## 16. 리스크

### 리스크 1: "보장"이라는 표현의 과장
LLM은 비결정적이므로 엄밀한 의미의 absolute guarantee는 어렵다.

대응:
- 외부 메시지는 "guarantee"보다 "protected memory"와 "verification" 중심으로 표현
- 제품 문서에 보장 범위와 한계 명시

### 리스크 2: instruction 판별 오탐
모든 문장을 instruction으로 오인하면 보호 기억이 비대해진다.

대응:
- rule-based + LLM hybrid 분류
- 수동 pin/unpin 제공
- 보호 구역 토큰 예산 상한 제공

### 리스크 2-1: Protected Zone 팽창
핵심 지시가 계속 누적되면 정작 현재 작업에 필요한 active context가 줄어들 수 있다.

대응:
- protected zone 비율 상한 적용
- topic 단위 latest-wins / supersedes 정책 도입
- 장문 identity/project memory는 summary card로 압축
- verify에서 protected budget overrun 경고 제공

### 리스크 3: drift 측정의 애매함
준수 여부 평가가 모호한 instruction이 존재한다.

대응:
- 초기에는 명확히 판정 가능한 포맷/스타일/금지 규칙 중심
- 자유형 취향 평가는 후순위

### 리스크 4: 제품 범위 확대
graph, RAG, multi-agent 등으로 쉽게 범위가 커질 수 있다.

대응:
- v0.x 동안은 instruction persistence KPI에 직접 연결되는 기능만 포함

---

## 17. 프로젝트 정보

| 항목 | 내용 |
|------|------|
| 이름 | MemGuard |
| 태그라인 | "Store, protect, verify." |
| 핵심 정의 | Instruction persistence for AI agents |
| 언어 | Python 3.9+ |
| 라이선스 | Apache 2.0 |
| 배포 형태 | OSS local-first library |

---

## 18. 최종 요약

MemGuard의 핵심은 "기억을 더 많이 저장하는 것"이 아니다.

핵심은 아래 세 가지다:
- 잊으면 안 되는 기억을 분류한다.
- 그 기억을 실제 컨텍스트에서 보호한다.
- 그 보호가 실제 행동으로 이어지는지 검증한다.

즉, MemGuard는 단순한 memory layer가 아니라:

> **AI가 중요한 지시를 실제로 계속 지키고 있는지 확인하는 persistence and verification layer다.**
