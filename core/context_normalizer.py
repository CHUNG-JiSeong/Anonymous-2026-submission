# context_normalizer_v2.py
# -*- coding: utf-8 -*-
"""
컨텍스트 전용 정규화 파이프라인 (사전/정규식 최소화, LLM 주도)
요구사항:
  1) context에서 명사/고유명사 추출 (LLM)
  2) 해당 mention의 로컬 문맥(문장) + mention을 쿼리로 top-k policy fragments 검색(임베딩)
  3) mention + 로컬문맥 + fragments를 LLM에 제공 → 상위어(hypernym/상위카테고리) 후보 제안
  4) fragment.type이 'premise'면 강(STRONG) 확신, 아니면 약(WEAK)
  5) 사전(dictionary)와 하드코딩 정규식 최소화

환경:
  pip install openai
  export OPENAI_API_KEY=...
"""

from __future__ import annotations
import json, os, time, math, hashlib, random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

# ===== OpenAI SDK =====
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("OpenAI SDK가 필요합니다. `pip install openai`") from e

DEBUG = False
def dbg(*a): 
    if DEBUG: print(*a)

# ====== 데이터 구조 ======

@dataclass
class PolicyFragment:
    frag_id: str
    frag_type: str     # 'premise' | 'actor_cu' | 'meta_cu' | 'unknown'
    text: str
    meta: Dict[str, Any]
    embedding: Optional[List[float]] = None

@dataclass
class Mention:
    text: str
    span: Optional[Tuple[int, int]]
    local_context: str      # mention이 포함된 문장/절 (LLM이 추출)
    mention_type: str       # data_item/purpose/activity/recipient/transfer/basis/role/jurisdiction/safeguard
    entity_id: Optional[str] = None   # ← 추가

@dataclass
class AMRRecord:
    context_id: str
    direction: str
    message_hash: str
    mention_id: str
    mention_text: str
    span: Optional[Tuple[int, int]]
    mention_type: str

    hypernym: str                   # 제안된 상위어
    strength: str                   # 'STRONG' | 'WEAK'
    confidence: float               # 0~1 (결정 모드에선 1.0/0.0)

    local_context: str
    retrieval: List[Dict[str, Any]] # [{"frag_id","frag_type","score","snippet"} ...]
    supporting_fragment_ids: List[str]
    rationale: str                  # 간단 근거

    status: str = "proposed"
    role: Optional[str] = None            # "strict" | "exploratory"
    kind: Optional[str] = None            # "data_category" | "recipient_type" | ...
    lexical_match: Optional[str] = None   # "exact" | "none"
    generalization: Optional[float] = None
    entity_id: Optional[str] = None   # ← 추가

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.span is not None:
            d["span"] = [self.span[0], self.span[1]]
        return d
    

ALLOWED_KINDS = {
    "data_item": {"data_category", "identifier", "special_category"},
    "purpose": {"purpose_category"},
    "activity": {"system", "processing_operation", "filing_system"},
    "system": {"system", "processing_operation", "filing_system"},
    "recipient": {"recipient_type", "controller", "processor", "role", "recipient"},
    "role": {"controller", "processor", "role"},
    "jurisdiction": {"jurisdiction", "region"},
    "basis": {"legal_basis", "transfer_basis"},
    "safeguard": {"safeguard"},
}

# ====== OpenAI Client 래퍼 ======

class OpenAIClient:
    def __init__(self, model: Optional[str] = None, embed_model: Optional[str] = None, max_retries: int = 1):
        self.client = OpenAI()
        self.chat_model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.embed_model = embed_model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        self.max_retries = max_retries

    def chat_json(self, system: str, user: str, temperature: float = 0.0) -> Dict[str, Any]:
        """JSON 강제 응답."""
        backoff = 1.0
        last_err: Optional[Exception] = None
        for _ in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=[{"role":"system","content":system},
                            {"role":"user","content":user}],
                    # temperature=0.0,
                    # top_p=0.0,
                    # presence_penalty=0.0,
                    # frequency_penalty=0.0,
                    # response_format={"type":"json_object"},
                )
                return json.loads(resp.choices[0].message.content)
            except Exception as e:
                last_err = e
                time.sleep(backoff); backoff = min(backoff*2, 16.0)
        raise RuntimeError(f"OpenAI chat_json 실패: {last_err}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        backoff = 1.0
        last_err: Optional[Exception] = None
        for _ in range(self.max_retries):
            try:
                resp = self.client.embeddings.create(model=self.embed_model, input=texts)
                return [d.embedding for d in resp.data]
            except Exception as e:
                last_err = e
                time.sleep(backoff); backoff = min(backoff*2, 16.0)
        raise RuntimeError(f"OpenAI embed 실패: {last_err}")


# ====== 유틸 ======

def sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) or 1e-9
    nb = math.sqrt(sum(y*y for y in b)) or 1e-9
    return dot / (na * nb)

def truncate(txt: str, max_chars: int = 600) -> str:
    if len(txt) <= max_chars:
        return txt
    return txt[:max_chars-3] + "..."

def _lexical_match(hypernym: str, texts: List[str]) -> str:
    h = (hypernym or "").strip().lower()
    if not h:
        return "none"
    for t in texts or []:
        if h in (t or "").lower():
            return "exact"
    return "none"

def _decide_role(strength: str, lexical_match: str, supporting_frag_types: List[str]) -> str:
    has_premise = any((ft or "").lower() == "premise" for ft in (supporting_frag_types or []))
    if has_premise and lexical_match == "exact":
        return "strict"
    return "exploratory"


# ====== Policy Fragment 인덱스 ======

class PolicyFragmentIndex:
    """
    gdpr_snapshot.json 포맷을 로드해 fragment 목록을 만들고 임베딩 인덱스를 구성.
    - 정규식/사전 없이: fragment 텍스트는 노드의 label/attrs를 직조해 구성
    - compliance_unit의 type은 있으면 사용, 없으면 'unknown'
    - attrs.section_role == 'premise'인 경우도 premise로 간주(있다면)
    """
    def __init__(self, graph_path: str, oai: OpenAIClient):
        self.graph_path = graph_path
        self.oai = oai
        self.fragments: List[PolicyFragment] = []

    def _node_to_text(self, n: Dict[str, Any]) -> str:
        label = (n.get("label") or "").strip()
        attrs = n.get("attrs") or {}
        text = (attrs.get("text") or "").strip()

        if n.get("kind") == "compliance_unit":
            # subject/condition/constraint/context를 자연어로 이어붙임(사전X, 포맷X)
            subj = attrs.get("subject")
            cond = attrs.get("condition")
            cons = attrs.get("constraint")
            ctx = attrs.get("context")
            parts = [label]
            if subj: parts.append(f"subject: {json.dumps(subj, ensure_ascii=False)}")
            if cond: parts.append(f"condition: {json.dumps(cond, ensure_ascii=False)}")
            if cons: parts.append(f"constraint: {json.dumps(cons, ensure_ascii=False)}")
            if ctx:  parts.append(f"context: {json.dumps(ctx, ensure_ascii=False)}")
            return " | ".join(p for p in parts if p)
        else:
            return text or label

    def _node_type(self, n: Dict[str, Any]) -> str:
        """
        정책 스냅샷 노드에서 fragment type을 안전하게 판별:
        - compliance_unit: 최상위 node["type"] 우선 → 구버전 attrs.type 보조 → section_role=="premise" 보정
        - 그 외(point/text): section_role=="premise"면 premise, 아니면 unknown
        """
        kind = (n.get("kind") or "").strip().lower()

        if kind == "compliance_unit":
            # 1) 최상위 type(현행 스냅샷)
            t_top = (n.get("type") or "").strip().lower()
            if t_top in {"actor_cu", "meta_cu", "premise"}:
                return t_top
            # 2) 구버전 호환: attrs.type
            t_attr = ((n.get("attrs") or {}).get("type") or "").strip().lower()
            if t_attr in {"actor_cu", "meta_cu", "premise"}:
                return t_attr
            # 3) 섹션 롤로 보정
            sr = ((n.get("attrs") or {}).get("section_role") or "").strip().lower()
            return "premise" if sr == "premise" else "unknown"

        # point/text 등
        sr = ((n.get("attrs") or {}).get("section_role") or "").strip().lower()
        return "premise" if sr == "premise" else "unknown"

    def build(self) -> None:
        with open(self.graph_path, "r", encoding="utf-8") as f:
            g = json.load(f)

        nodes: List[Dict[str, Any]] = g.get("nodes", [])
        dbg(f"[IDX] nodes={len(nodes)}")  # ← 추가
        frags: List[PolicyFragment] = []
        for n in nodes:
            if n.get("kind") not in {"compliance_unit", "point", "text"}:
                continue
            txt = self._node_to_text(n)
            if not txt:
                continue
            ftype = self._node_type(n)
            frags.append(PolicyFragment(
                frag_id=n.get("id") or "",
                frag_type=ftype,
                text=txt,
                meta={"kind": n.get("kind")}
            ))
        dbg(f"[IDX] frags(before-emb)={len(frags)}")  # ← 추가

        # 임베딩 생성
        embeddings = self.oai.embed([f.text for f in frags])
        for f, vec in zip(frags, embeddings):
            f.embedding = vec
        self.fragments = frags
        dbg(f"[IDX] frags(embedded)={len(self.fragments)} dim={len(self.fragments[0].embedding) if self.fragments else 0}")  # ← 추가

    def search(self, query_text: str, top_k: int = 6) -> List[Tuple[PolicyFragment, float]]:
        qvec = self.oai.embed([query_text])[0]
        scored = [(f, cosine(qvec, f.embedding)) for f in self.fragments if f.embedding]
        scored.sort(key=lambda x: (-x[1], x[0].frag_id))  # ← tie-breaker
        out = scored[:top_k]
        dbg(f"[SRCH] top={len(out)} sample={[ (f.frag_type, round(s,3)) for f,s in out[:3] ]}")  # ← 추가
        return out
    
    def search_batch(self, query_texts: List[str], top_k: int = 6) -> List[List[Tuple[PolicyFragment, float]]]:
        """여러 쿼리를 임베딩 1회로 처리."""
        if not query_texts:
            return []
        qvecs = self.oai.embed(query_texts)  # 1회 배치 호출
        results: List[List[Tuple[PolicyFragment, float]]] = []
        for qv in qvecs:
            scored = [(f, cosine(qv, f.embedding)) for f in self.fragments if f.embedding]
            scored.sort(key=lambda x: (-x[1], x[0].frag_id))
            # 보너스 재가중(옵션)
            def bonus(f: PolicyFragment) -> float:
                b = 0.0
                if (f.frag_type or "").lower() == "premise":
                    b += 0.05
                t = (f.text or "").lower()
                if " means " in t or " shall mean " in t or "definition" in t:
                    b += 0.03
                return b
            rescored = [(f, s + bonus(f)) for (f, s) in scored]
            rescored.sort(key=lambda x: (-x[1], x[0].frag_id))
            results.append(rescored[:top_k])
        return results
    


# ====== LLM: 명사/고유명사 + 로컬 문맥 추출 ======

MENTION_SYS = """\
You extract domain-relevant mentions from CONTEXT.
Return JSON:
{ "mentions": [ { "text": "...", "span":[start,end], "local_context":"sentence/phrase containing the text", "type":"data_item|purpose|activity|recipient|transfer|basis|role|jurisdiction|safeguard" }, ... ] }
Guidelines:
- Use only the given context; do not invent facts.
- Prefer NOUN/PROPER NOUN phrases important to compliance (data items, actors, purposes, transfers, destinations, bases).
- 2~12 mentions.
- Offsets are 0-based, end-exclusive, on the given CONTEXT string.
"""

def build_mention_user(context_text: str) -> str:
    return json.dumps({"CONTEXT": context_text}, ensure_ascii=False)

class NounMentionExtractor:
    def __init__(self, oai: OpenAIClient):
        self.oai = oai

    def extract(self, context_text: str) -> List[Mention]:
        obj = self.oai.chat_json(MENTION_SYS, build_mention_user(context_text), temperature=0.0)
        out = obj.get("mentions") or []
        dbg(f"[MENT] raw={len(out)}")  # ← 추가
        mentions: List[Mention] = []
        for m in out[:12]:
            txt = (m.get("text") or "").strip()
            span = m.get("span")
            if not txt:
                continue
            if not (isinstance(span, list) and len(span) == 2):
                span = None
            lc = (m.get("local_context") or "").strip() or context_text
            t = (m.get("type") or "data_item").strip()
            mentions.append(Mention(
                text=txt,
                span=tuple(span) if span else None,
                local_context=lc,
                mention_type=t if t in {
                    "data_item","purpose","activity","recipient",
                    "transfer","basis","role","jurisdiction","safeguard"
                } else "data_item"
            ))
        dbg(f"[MENT] kept={len(mentions)} ex={[ (m.text, m.mention_type) for m in mentions[:5] ]}")  # ← 추가
        return mentions


# ====== LLM: 상위어 후보 판단(문장 기반, dictionary 없음) ======

HYPERNYM_SYS = """\
You normalize a SOURCE mention into up to 3 hypernyms using ONLY LOCAL_CONTEXT and POLICY_FRAGMENTS.

ENTITY_TYPE ∈ {data_item, purpose, activity, recipient, transfer, basis, role, jurisdiction, safeguard, system}.
Treat activity and system as the same “processing/system” family.

INPUT STRUCTURE:
{
  "GLOBAL_CONTEXT": "<full original text>",
  "ITEMS": [
    {
      "idx": 0,
      "entity_type": "data_item|purpose|activity|recipient|transfer|basis|role|jurisdiction|safeguard|system",
      "source": {"text": "<mention>", "local_context": "<optional short snippet>"},
      "policy_fragments": [
        {"frag_id":"...", "frag_type":"premise|actor_cu|meta_cu|unknown", "text":"..."},
        ...
      ]
    },
    ...
  ]
}

Task:
1) Always return 0–3 proposals. If evidence is weak (no clear premise), you MUST still return proposals but mark them as "role":"exploratory".
2) Kinds allowed per ENTITY_TYPE.
   - Examples:
     - ENTITY_TYPE=data_item → {data_category, identifier, special_category} only
     - ENTITY_TYPE=system|activity → {system, processing_operation, filing_system}
     - ENTITY_TYPE=role|recipient → {role, recipient, controller, processor}
     - ENTITY_TYPE=jurisdiction → {jurisdiction, region}
     - ENTITY_TYPE=basis → {legal_basis, transfer_basis}
     - ENTITY_TYPE=safeguard → {safeguard}
3) Evidence requirement:
   - If kind is data_category|identifier|special_category, at least ONE supporting fragment MUST be type "premise".
   - Otherwise prefer premise, but actor_cu/meta_cu allowed.
4) Base decisions ONLY on LOCAL_CONTEXT and POLICY_FRAGMENTS. Do not invent labels not supported by fragments.

OUTPUT (JSON only):
{
  "items": [
    {
      "idx": 0,
      "proposals": [
        {
          "hypernym": "...",
          "kind": "...",
          "supporting_fragment_ids": ["..."],
          "role": "strict" | "exploratory",
          "lexical_match": "exact" | "none",
          "generalization": 0.2 | 0.6,
          "short_rationale": "one short sentence"
        }
      ]
    },
    ...
  ]
}

FEW-SHOT EXAMPLES:

[EX1: data_item → data_category]
SOURCE.text = "ICD-10 diagnosis codes"
LOCAL_CONTEXT = "We export from the EHR: ICD-10 diagnosis codes, lab flags, year of birth..."
POLICY_FRAGMENTS includes a premise defining "data concerning health".
Output:
{
  "proposals": [
    {
      "hypernym": "data concerning health",
      "kind": "data_category",
      "supporting_fragment_ids": ["<frag:health_def>"],
      "role": "strict",
      "lexical_match": "exact",
      "generalization": 0.2,
      "short_rationale": "The codes directly indicate a person's health status."
    },
    {
      "hypernym": "codes of conduct",
      "kind": "other",
      "supporting_fragment_ids": ["<frag:conduct_generic>"],
      "role": "exploratory",
      "lexical_match": "none",
      "generalization": 0.6,
      "short_rationale": "This label does not semantically fit a data item."
    }
  ]
}

[EX2: data_item → location]
SOURCE.text = "5-digit postcode"
Output:
{
  "proposals": [
    {
      "hypernym": "location data",
      "kind": "data_category",
      "supporting_fragment_ids": ["<frag:location_def>"],
      "role": "strict",
      "lexical_match": "exact",
      "generalization": 0.2,
      "short_rationale": "A postcode is a direct indicator of location."
    },
    {
      "hypernym": "structured data",
      "kind": "other",
      "supporting_fragment_ids": ["<frag:generic_structure>"],
      "role": "exploratory",
      "lexical_match": "none",
      "generalization": 0.6,
      "short_rationale": "This is a broad category not specific to the item."
    }
  ]
}

[EX3: data_item (identifier)]
SOURCE.text = "hashed patient ID"
Output:
{
  "proposals": [
    {
      "hypernym": "identifier",
      "kind": "data_category",
      "supporting_fragment_ids": ["<frag:identifier_def>"],
      "role": "strict",
      "lexical_match": "exact",
      "generalization": 0.2,
      "short_rationale": "A hashed ID is used to identify a person consistently."
    },
    {
      "hypernym": "data concerning health",
      "kind": "data_category",
      "supporting_fragment_ids": ["<frag:health_def>"],
      "role": "exploratory",
      "lexical_match": "none",
      "generalization": 0.6,
      "short_rationale": "Link to health is indirect and weak without context."
    }
  ]
}

[EX4: recipient]
SOURCE.text = "US-based ad vendors"
Output:
{
  "proposals": [
    {
      "hypernym": "recipients in a third country",
      "kind": "recipient_type",
      "supporting_fragment_ids": ["<frag:third_country_recipient>"],
      "role": "strict",
      "lexical_match": "exact",
      "generalization": 0.2,
      "short_rationale": "US-based advertising vendors are recipients outside the EEA."
    }
  ]
}

Return ONLY the JSON.
"""

def build_hypernym_user(mention: Mention, fragments: List[Tuple[PolicyFragment, float]]) -> str:
    payload = {
        "ENTITY_TYPE": mention.mention_type,  # <= 추가
        "SOURCE": {
            "text": mention.text,
            "local_context": mention.local_context
        },
        "POLICY_FRAGMENTS": [
            {
                "frag_id": f.frag_id,
                "frag_type": f.frag_type,   # 'premise' | 'actor_cu' | 'meta_cu' | 'unknown'
                "text": truncate(f.text, 700)
            }
            for (f, _score) in fragments
        ]
    }
    return json.dumps(payload, ensure_ascii=False)

class HypernymInferencer:
    def __init__(self, oai: OpenAIClient):
        self.oai = oai

    def infer(self, mention, frags, n_attempts: int = 1):
        obj = self.oai.chat_json(
            HYPERNYM_SYS,
            build_hypernym_user(mention, frags),  # 기존 build 함수 재사용 (필드 그대로)
            temperature=0.0
        )
        props = []
        for p in (obj.get("proposals") or [])[:3]:
            h = (p.get("hypernym") or "").strip()
            if not h: continue
            props.append({
                "hypernym": h,
                "supporting_fragment_ids": p.get("supporting_fragment_ids") or []
            })
        return props, (1.0 if props else 0.0)
    
    # === In HypernymInferencer (context_normalizer.py) ===
    def infer_batch(self,
                    mentions: List[Mention],
                    frags_per_mention: List[List[Tuple[PolicyFragment, float]]],
                    global_context: str,
                    temperature: float = 0.0
                ) -> List[List[Dict[str, Any]]]:
        """
        mentions[i] 에 대해 frags_per_mention[i] 정렬 리스트를 주고,
        HYPERNYM_SYS_BATCH 1회 호출로 각 mention별 proposals 배열을 돌려받는다.
        return: proposals_by_idx: List[List[proposal_dict]]
        """
        items = []
        for i, m in enumerate(mentions):
            frs = frags_per_mention[i] if i < len(frags_per_mention) else []
            items.append({
                "idx": i,
                "entity_type": m.mention_type,
                "source": {"text": m.text, "local_context": m.local_context or ""},
                "policy_fragments": [
                    {"frag_id": f.frag_id, "frag_type": f.frag_type, "text": truncate(f.text, 350)}
                    for (f, _s) in frs
                ],
            })
        payload = {"GLOBAL_CONTEXT": global_context, "ITEMS": items}

        obj = self.oai.chat_json(HYPERNYM_SYS, json.dumps(payload, ensure_ascii=False), temperature=temperature)
        by_idx: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(mentions))}
        for it in obj.get("items", []):
            idx = int(it.get("idx", -1))
            props = it.get("proposals") or []
            if 0 <= idx < len(mentions):
                # 상한 3개, 안전 필드만 유지
                cleaned = []
                for p in props[:3]:
                    cleaned.append({
                        "hypernym": (p.get("hypernym") or "").strip(),
                        "kind": (p.get("kind") or None),
                        "role": (p.get("role") or None),
                        "lexical_match": (p.get("lexical_match") or None),
                        "generalization": p.get("generalization"),
                        "supporting_fragment_ids": list(p.get("supporting_fragment_ids") or []),
                        "short_rationale": (p.get("short_rationale") or "").strip(),
                    })
                by_idx[idx] = cleaned
        # 인덱스 순서로 반환
        return [by_idx[i] for i in range(len(mentions))]


# (1) 엔티티 타입 → mention_type 간단 매핑 (사전 최소화. 필요시 조정)
ENTITY2MENTION = {
    "data_item": "data_item",
    "actor": "role",
    "recipient": "recipient",
    "purpose": "purpose",
    "basis": "basis",
    "jurisdiction": "jurisdiction",
    "system": "activity",   # 시스템은 활동/처리 맥락으로 태깅
    "other": "data_item",
}

def _slice_local_context(text: str, span: Optional[Tuple[int,int]], window: int = 160) -> str:
    if not text:
        return ""
    if not span:
        # span이 없으면 전체에서 앞부분만 사용
        return text[:min(len(text), 400)].strip()

    # span 클램핑
    s, e = span
    n = len(text)
    s = max(0, min(int(s), n))
    e = max(0, min(int(e), n))
    if e < s:
        s, e = e, s  # 잘못된 순서 방어

    # 윈도우 자르기
    left = max(0, s - window)
    right = min(n, e + window)
    snippet = text[left:right]

    # 문장 경계(느슨) 찾기
    punct = ".!?;\n"
    anchor_in_snip = s - left
    # 왼쪽 경계: anchor 이전에서 가장 오른쪽 문장부호(없으면 0)
    left_candidates = []
    for p in punct:
        idx = snippet.rfind(p, 0, anchor_in_snip)
        if idx != -1:
            left_candidates.append(idx + 1)  # 문장부호 다음부터
    cut_l = max(left_candidates) if left_candidates else 0

    # 오른쪽 경계: e 이후에서 가장 이른 문장부호(없으면 끝)
    right_candidates = []
    for p in punct:
        idx = snippet.find(p, e - left)
        if idx != -1:
            right_candidates.append(idx)
    cut_r = min(right_candidates) if right_candidates else len(snippet)

    piece = snippet[cut_l:cut_r].strip()
    return piece or snippet.strip()

# ====== 오케스트레이터 ======

class ContextNormalizerV2:
    """
    호출 시 AMR 리스트를 즉시 반환.
    """
    def __init__(self, policy_graph_path: str,
                 chat_model: Optional[str] = None,
                 embed_model: Optional[str] = None,
                 seed: int = 0,
                 soft_guard: bool = False,     # ← 추가 (기본 OFF: 기존과 동일)
                 strict_mode: bool = False):   # ← 추가 (기본 OFF)
        random.seed(seed)
        self.oai = OpenAIClient(chat_model, embed_model)
        self.index = PolicyFragmentIndex(policy_graph_path, self.oai)
        self.index.build()
        self.mentioner = NounMentionExtractor(self.oai)
        self.inferencer = HypernymInferencer(self.oai)

        self.soft_guard = soft_guard
        self.strict_mode = strict_mode

    @staticmethod
    def _hash(text: str) -> str:
        return sha256(text)

    @staticmethod
    def _reweight_scores(scored: List[Tuple[PolicyFragment, float]]) -> List[Tuple[PolicyFragment, float]]:
        def bonus(f: PolicyFragment) -> float:
            b = 0.0
            if (f.frag_type or "").lower() == "premise":
                b += 0.05
            t = (f.text or "").lower()
            # 사전 아님: 일반적 정의 표식만 사용
            if " means " in t or " shall mean " in t or "definition" in t:
                b += 0.03
            return b
        return [(f, s + bonus(f)) for (f, s) in scored]
    
    # --- soft-guard helpers ---
    @staticmethod
    def _evidence_tier(f: PolicyFragment) -> int:
        """
        0(낮음) / 1(보통) / 2(강함)
        - premise면 2
        - 정의형 문구(shall mean / means / definition)가 있으면 2
        - 강한 의무/요구(must/shall/require 등) 문구는 1
        - 그 외 0
        """
        if (f.frag_type or "").lower() == "premise":
            return 2
        t = (f.text or "").lower()
        if (" shall mean " in t) or (" means " in t) or ("definition" in t):
            return 2
        if any(k in t for k in (" must ", " shall ", " require", " obligation", " required ")):
            return 1
        return 0

    @staticmethod
    def _compatibility_weight(entity_type: str, kind: Optional[str]) -> float:
        """
        엔티티 타입 ↔ hypernym kind 정합성 가중치.
        - 정합: 1.0
        - 불명/비정합: 0.5 (드롭하지 않고 강등만; 문서 일반성 보존)
        """
        if not kind:
            return 0.6  # kind 미지정이면 중간값
        et = (entity_type or "").lower()
        k = (kind or "").lower()

        allowed = {
            "data_item": {"data_category", "identifier", "special_category"},
            "activity": {"system", "processing_operation", "filing_system"},
            "recipient": {"recipient", "controller", "processor", "role", "recipient_type"},
            "role": {"role", "controller", "processor"},
            "jurisdiction": {"jurisdiction", "region"},
            "basis": {"legal_basis", "transfer_basis"},
            "transfer": {"transfer_basis", "recipient_type"},
            "safeguard": {"safeguard"},
            "system": {"system", "processing_operation", "filing_system"},
        }
        # 맵에 없으면 data_item로 완화
        allowed_set = allowed.get(et, {"data_category", "identifier", "special_category"})
        return 1.0 if k in allowed_set else 0.5

    def _retrieval_pack(self, scored: List[Tuple[PolicyFragment, float]], max_items: int = 6) -> List[Dict[str, Any]]:
        out = []
        for f, s in scored[:max_items]:
            out.append({
                "frag_id": f.frag_id,
                "frag_type": f.frag_type,
                "score": round(float(s), 6),
                "snippet": truncate(f.text, 200)
            })
        return out

    def normalize(self, contexts: List[Dict[str, Any]],
                  top_k: int = 6) -> List[Dict[str, Any]]:
        # amrs: List[AMRRecord] = []

        # for c in contexts:
        #     ctx_id = c.get("id") or c.get("context_id") or f"ctx_{len(amrs)}"
        #     direction = c.get("direction", "Inbound")
        #     text = c.get("text", "")
        #     msg_hash = self._hash(text)

        #     mentions = self.mentioner.extract(text)
        #     if not mentions:
        #         # abstain 1줄 남겨 누락 추적
        #         amrs.append(AMRRecord(
        #             context_id=ctx_id, direction=direction, message_hash=msg_hash,
        #             mention_id=f"{ctx_id}_m00", mention_text="", span=None, mention_type="data_item",
        #             hypernym="", strength="WEAK", confidence=0.0,
        #             local_context="", retrieval=[], supporting_fragment_ids=[], rationale="",
        #             status="abstain"
        #         ))
        #         continue

        #     for m in mentions:
        #         amrs.extend(self._amr_records_for_mention(
        #             ctx_id=ctx_id,
        #             direction=direction,
        #             msg_hash=msg_hash,
        #             m=m,
        #             top_k=top_k
        #         ))

        # return [a.to_dict() for a in amrs]
        raise NotImplementedError("Use normalize_entities(...) only.")
    
    # def _process_mentions(self, ctx_id: str, direction: str, text: str,
    #                       mentions: List[Mention], top_k: int) -> List[AMRRecord]:
    #     amrs: List[AMRRecord] = []
    #     msg_hash = self._hash(text)
    #     for i, m in enumerate(mentions):
    #         query = f"SOURCE: {m.text}\nCONTEXT: {m.local_context}"
    #         scored = self.index.search(query, top_k=top_k)
    #         ret_pack = self._retrieval_pack(scored)

    #         # LLM 하이퍼님 판단
    #         props, conf = self.inferencer.infer(m, scored, n_attempts=1)
    #         if not props:
    #             amrs.append(AMRRecord(
    #                 context_id=ctx_id, direction=direction, message_hash=msg_hash,
    #                 mention_id=f"{ctx_id}_m{i:02d}", mention_text=m.text, span=m.span,
    #                 mention_type=m.mention_type, hypernym="", strength="WEAK", confidence=0.0,
    #                 local_context=m.local_context, retrieval=ret_pack,
    #                 supporting_fragment_ids=[], rationale="", status="abstain"
    #             ))
    #             continue

    #         # 제안들 → AMR 레코드
    #         frag_map = {f.frag_id: f for f, _ in scored}
    #         for p in props:
    #             sup_ids = list(p.get("supporting_fragment_ids") or [])
    #             strength = "WEAK"
    #             for sid in sup_ids:
    #                 f = frag_map.get(sid)
    #                 if f and f.frag_type.lower() == "premise":
    #                     strength = "STRONG"; break

    #             amrs.append(AMRRecord(
    #                 context_id=ctx_id, direction=direction, message_hash=msg_hash,
    #                 mention_id=f"{ctx_id}_m{i:02d}", mention_text=m.text, span=m.span,
    #                 mention_type=m.mention_type, hypernym=p["hypernym"],
    #                 strength=strength, confidence=float(conf),
    #                 local_context=m.local_context, retrieval=ret_pack,
    #                 supporting_fragment_ids=sup_ids, rationale=p.get("rationale",""),
    #                 status="proposed"
    #             ))
    #     return amrs

    def normalize_entities(self,
                        context_id: str,
                        direction: str,
                        context_text: str,
                        entities: List[Dict[str, Any]],
                        top_k: int = 6) -> List[Dict[str, Any]]:
        """
        ER 엔티티 입력 → (배치 임베딩 검색 + 배치 하이퍼님 추론) → AMR 리스트 반환
        """
        print(f"[NORM-ENT] entities={len(entities)}")
        mentions: List[Mention] = []
        for e in entities:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            # span: 첫 mention만 사용
            span = None
            for m in (e.get("mentions") or []):
                sp = m.get("span")
                if isinstance(sp, (list, tuple)) and len(sp) == 2:
                    span = (int(sp[0]), int(sp[1])); break
            etype = (e.get("type") or "other").lower()
            mtype = ENTITY2MENTION.get(etype, "data_item")
            local_ctx = ""  # 배치에서는 전문을 LLM에 주므로, 여기선 비워도 OK (원하면 _slice_local_context 사용)
            mentions.append(Mention(text=name, span=span, local_context=local_ctx, mention_type=mtype))

        if not mentions:
            return []

        # (1) 배치 임베딩 검색
        queries = [f"SOURCE: {m.text}\nCONTEXT: {m.local_context or ''}" for m in mentions]
        frags_lists = self.index.search_batch(queries, top_k=top_k)
        print(f"[SRCH.B] items={len(frags_lists)} top_k={top_k} sample={[ (frags_lists[0][0][0].frag_type, round(frags_lists[0][0][1],3)) ] if frags_lists and frags_lists[0] else []}")

        # (2) 배치 하이퍼님 LLM
        proposals_by_idx = self.inferencer.infer_batch(mentions, frags_lists, global_context=context_text, temperature=0.0)
        print(f"[HYP.B] items={len(proposals_by_idx)}")

        # (3) AMR 레코드 변환
        msg_hash = self._hash(context_text)
        amrs_out: List[AMRRecord] = []
        for i, m in enumerate(mentions):
            scored = frags_lists[i]
            ret_pack = self._retrieval_pack(scored)
            frag_map = {f.frag_id: f for f, _ in scored}
            props = proposals_by_idx[i] or []

            if not props:
                amrs_out.append(AMRRecord(
                    context_id=context_id, direction=direction, message_hash=msg_hash,
                    mention_id=f"{context_id}_m{i:02d}", mention_text=m.text, span=m.span,
                    mention_type=m.mention_type, hypernym="", strength="WEAK", confidence=0.0,
                    local_context=m.local_context, retrieval=ret_pack,
                    supporting_fragment_ids=[], rationale="", status="abstain"
                ))
                continue

            for p in props:
                sup_ids = [sid for sid in (p.get("supporting_fragment_ids") or []) if sid in frag_map]
                support_types = [(frag_map[sid].frag_type or "").lower() for sid in sup_ids]
                strength = "STRONG" if any(t == "premise" for t in support_types) else "WEAK"
                lex = _lexical_match(p.get("hypernym",""), [r["snippet"] for r in ret_pack if r["frag_id"] in sup_ids] + [m.local_context])

                amrs_out.append(AMRRecord(
                    context_id=context_id, direction=direction, message_hash=msg_hash,
                    mention_id=f"{context_id}_m{i:02d}", mention_text=m.text, span=m.span,
                    mention_type=m.mention_type, hypernym=p.get("hypernym",""),
                    strength=strength, confidence=1.0,   # 배치 1회 → 1.0로 고정(원하면 이후 self-consistency 추가)
                    local_context=m.local_context, retrieval=ret_pack,
                    supporting_fragment_ids=sup_ids, rationale=p.get("short_rationale",""),
                    status="proposed",
                    role=(p.get("role") or None),
                    kind=(p.get("kind") or None),
                    lexical_match=lex,
                    generalization=float(p.get("generalization") or 0.6)
                ))

        print(f"[AMR.B] out={len(amrs_out)}")
        return [a.to_dict() for a in amrs_out]
    
    def _amr_records_for_mention(
        self,
        ctx_id: str,
        direction: str,
        msg_hash: str,
        m: Mention,
        top_k: int,
        idx: Optional[int] = None,
    ) -> List[AMRRecord]:
        # 1) 검색
        query = f"SOURCE: {m.text}\nCONTEXT: {m.local_context}"
        scored = self.index.search(query, top_k=top_k)  # ← _reweight_scores 제거
        ret_pack = self._retrieval_pack(scored)
        frag_map = {f.frag_id: f for f,_ in scored}

        props, conf = self.inferencer.infer(m, scored, n_attempts=1)
        if not props:
            return [AMRRecord(
                context_id=ctx_id, direction=direction, message_hash=msg_hash,
                mention_id=f"{ctx_id}_m{(idx if idx is not None else 0):02d}",
                mention_type=m.mention_type, hypernym="", strength="WEAK", confidence=0.0,
                local_context=m.local_context, retrieval=ret_pack,
                supporting_fragment_ids=[], rationale="", status="abstain",
                role=None, kind=None, lexical_match=None, generalization=None,
                entity_id=m.entity_id  # ← 추가
            )]

        out = []
        for p in props:
            sup_ids = [sid for sid in (p.get("supporting_fragment_ids") or []) if sid in frag_map]
            strength = "STRONG" if any(frag_map[s].frag_type.lower()=="premise" for s in sup_ids) else "WEAK"
            out.append(AMRRecord(
                context_id=ctx_id, direction=direction, message_hash=msg_hash,
                mention_id=f"{ctx_id}_m{(idx or 0):02d}",
                mention_text=m.text, span=m.span, mention_type=m.mention_type,
                hypernym=p["hypernym"], strength=strength, confidence=conf,
                local_context=m.local_context, retrieval=ret_pack,
                supporting_fragment_ids=sup_ids, rationale="",
                status="proposed",
                role=None, kind=None, lexical_match=None, generalization=None,
                entity_id=m.entity_id  # ← 추가
            ))
        return out
    
    def _eval_proposal(self, mention_type, p, frag_map):
        kind = (p.get("kind") or "").strip().lower()
        allowed = ALLOWED_KINDS.get(mention_type, set())
        kind_ok = (kind in allowed) if allowed else True

        sup_ids = [sid for sid in (p.get("supporting_fragment_ids") or []) if sid in frag_map]
        has_premise = any((frag_map[sid].frag_type or "").lower()=="premise" for sid in sup_ids)
        requires_premise = kind in {"data_category", "identifier", "special_category"}
        evidence_ok = has_premise if requires_premise else True

        lex_ok = (p.get("lexical_match") == "exact")
        role_strict = (p.get("role") == "strict")
        return kind_ok, evidence_ok, lex_ok, role_strict, sup_ids

    def _guard_and_rank(self, mention, props, frag_map, soft_guard=True, strict_mode=False):
        ranked = []
        for p in props:
            kind_ok, evidence_ok, lex_ok, role_strict, sup_ids = self._eval_proposal(mention.mention_type, p, frag_map)

            # strict: 불일치/증거부족이면 과감히 제거
            if strict_mode and (not kind_ok or not evidence_ok):
                continue

            score = 1.0
            if soft_guard:
                if kind_ok:      score += 0.30
                if evidence_ok:  score += 0.30
                if lex_ok:       score += 0.10
                if role_strict:  score += 0.10

            p["_score"] = score
            p["_sup_ids"] = sup_ids
            ranked.append(p)

        ranked.sort(key=lambda x: x["_score"], reverse=True)
        return ranked[:3]





# ====== 외부 진입점(간단 호출) ======

def normalize_contexts(contexts: List[Dict[str, Any]],
                       policy_graph_path: str,
                       chat_model: Optional[str] = None,
                       embed_model: Optional[str] = None,
                       seed: int = 0,
                       top_k: int = 6) -> List[Dict[str, Any]]:
    """
    즉시 AMR 리스트 반환.
    """
    normalizer = ContextNormalizerV2(policy_graph_path, chat_model, embed_model, seed)
    return normalizer.normalize(contexts, top_k=top_k)

def normalize_for_entities(context_id: str,
                           direction: str,
                           context_text: str,
                           entities: List[Dict[str, Any]],
                           policy_graph_path: str,
                           chat_model: Optional[str] = None,
                           embed_model: Optional[str] = None,
                           seed: int = 0,
                           top_k: int = 6) -> List[Dict[str, Any]]:
    normalizer = ContextNormalizerV2(policy_graph_path, chat_model, embed_model, seed)
    return normalizer.normalize_entities(context_id, direction, context_text, entities, top_k=top_k)