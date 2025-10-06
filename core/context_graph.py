# er_triple_extractor.py
# -*- coding: utf-8 -*-
"""
컨텍스트를 LLM 프롬프트(그래프RAG 유사)로 파싱해
  - 엔티티들(노드)
  - 관계 트리플들(엣지: subject, predicate, object)
을 JSON으로 추출.
이미 계산된 AMR(상위어 제안)이 있으면 해당 엔티티 feature.hypernyms로 흡수.

요구 사항
- 사전(dictionary) / 정규식 규칙에 의존하지 않음(최소화)
- 상위어는 있으면 feature로만 보강 (없어도 동작)
- 결과는 즉시 반환
- OpenAI SDK 사용: pip install openai; export OPENAI_API_KEY=...

출력 스키마(간결):
{
  "context_id": "...",
  "entities": [
    {"id":"e1","name":"...", "type":"data_item|actor|purpose|basis|recipient|jurisdiction|system|other",
     "aliases":["..."], "features":{"hypernyms":["...","..."]}, "mentions":[{"text":"...","span":[s,e]}]}
  ],
  "relations": [
    {"subj":"e1","pred":"uses|transfers_to|relies_on|purpose|contains|identifies|acts_as|located_in|targets|shares_with|processed_under|... (open)",
     "obj":"e2", "evidence":{"quote":"...", "span":[s,e]}, "confidence":0.0~1.0}
  ]
}
"""

from __future__ import annotations
import os, json, time, hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

# ===== OpenAI SDK 래퍼 =====
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("OpenAI SDK가 필요합니다. `pip install openai`") from e


def sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


class OpenAIClient:
    def __init__(self, model: Optional[str] = None, max_retries: int = 5):
        self.client = OpenAI()
        self.chat_model = model or os.getenv("OPENAI_MODEL", "gpt-5")
        self.max_retries = max_retries

    def chat_json(self, system: str, user: str, temperature: float = 0.0) -> Dict[str, Any]:
        backoff = 1.0
        last_err: Optional[Exception] = None
        for _ in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.chat_model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    # temperature=temperature,
                    # response_format={"type": "json_object"},
                )
                return json.loads(resp.choices[0].message.content)
            except Exception as e:
                last_err = e
                time.sleep(backoff); backoff = min(backoff*2, 16.0)
        raise RuntimeError(f"OpenAI chat_json 실패: {last_err}")


# ===== 데이터 구조 =====

@dataclass
class ERMention:
    text: str
    span: Optional[Tuple[int, int]] = None

@dataclass
class EREntity:
    id: str
    name: str
    etype: str  # data_item|actor|purpose|basis|recipient|jurisdiction|system|other
    aliases: List[str]
    features: Dict[str, Any]     # {"hypernyms":[...], ...}
    mentions: List[ERMention]

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id, "name": self.name, "type": self.etype,
            "aliases": self.aliases, "features": self.features,
            "mentions": [{"text": m.text, "span": list(m.span) if m.span else None} for m in self.mentions],
        }
        return d

@dataclass
class ERRelation:
    subj: str
    pred: str
    obj: str
    evidence_quote: Optional[str] = None
    evidence_span: Optional[Tuple[int, int]] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subj": self.subj, "pred": self.pred, "obj": self.obj,
            "evidence": {"quote": self.evidence_quote,
                         "span": list(self.evidence_span) if self.evidence_span else None},
            "confidence": float(self.confidence),
        }


# ===== 프롬프트 (그래프RAG 유사 스타일) =====

ER_SYS = """\
You are an information extraction model that builds an entity–relation graph for compliance analysis.
Follow ONLY the provided text and hints.
Do not invent facts.
Be conservative, but DO extract explicitly stated incidents, time expressions, counts, and intentions (plans/considerations).

Output JSON: { "entities":[ {"id":"e1","name":"...","type":"data_item|actor|purpose|basis|recipient|jurisdiction|system|other|event|timepoint|duration|number", "aliases":["..."], "features":{"hypernyms":["..."]}, "mentions":[{"text":"...","span":[start,end]}]}, ... ], "relations":[ {"subj":"eX","pred":"...", "obj":"eY", "evidence":{"quote":"...", "span":[start,end]}, "confidence":0.0} ] }
Guidelines (additions):
- In addition to generic entities, ALWAYS extract:
    • **event** for security incidents (e.g., “publicly accessible”, “exposed”, “leak”, “breach”, “unauthorised access”).
    • **timepoint** for discovery/notification times (e.g., “T-1d”, dates, “within 72 hours”).
    • **duration** for periods (e.g., “unknown period”, “for a week”, ISO-like “P7D” if obvious).
    • **number** for record counts (e.g., “45,000”). • “intent/plan” cues (consider/plan/intend) as relations; do NOT assert actual performance.
- Keep predicates short and consistent. In addition to your current set, you MAY use: breach_of_security, publicly_accessible, involves, discovered_at, exposure_duration, approx_count_of_records, includes_field, security_control, intends_to_delay_notification, plans_monitoring, notify_authority, notify_individuals.
- If something is **unknown/undecided**, you may still extract it as an entity (e.g., duration="unknown") and connect with a predicate (e.g., exposure_duration → unknown).
- Evidence spans: use conservative 0-based [start,end). Quote the smallest span that supports the relation.
- Avoid self-loops and tautologies. 10–30 relations are typical if the paragraph is rich. Quality rules (unchanged + clarifications):
- Prefer relations that connect actors/systems/recipients/purposes to data items or events.
- One event node can link multiple signals (system/data/time/duration/count).
- Do not derive legal conclusions; only extract stated facts and intentions.

-- Few-shot example 1 --
Context: “An S3 bucket with around 45,000 customer PDFs—containing names, addresses, and partial card numbers—was publicly accessible for an unknown period. We discovered this yesterday (T-1d). We are considering monitoring for a week before deciding whether to notify the authority and individuals. Jurisdiction: UK. Controls noted: encryption at rest, logging.”
Output: {
  "entities": [
    {"id":"e_sys","name":"S3 bucket","type":"system","aliases":[],"features":{"hypernyms":["storage"]},"mentions":[{"text":"S3 bucket","span":[3,12]}]},
    {"id":"e_docs","name":"customer PDFs","type":"data_item","aliases":[],"features":{"hypernyms":["documents"]},"mentions":[{"text":"customer PDFs","span":[24,37]}]},
    {"id":"e_name","name":"names","type":"data_item","aliases":[],"features":{},"mentions":[{"text":"names","span":[50,55]}]},
    {"id":"e_addr","name":"addresses","type":"data_item","aliases":[],"features":{},"mentions":[{"text":"addresses","span":[57,66]}]},
    {"id":"e_card","name":"partial card numbers","type":"data_item","aliases":[],"features":{"hypernyms":["financial data","identifiers"]},"mentions":[{"text":"partial card numbers","span":[72,93]}]},
    {"id":"e_evt","name":"security incident","type":"event","aliases":["public exposure"],"features":{},"mentions":[{"text":"publicly accessible","span":[100,119]}]},
    {"id":"e_cnt","name":"45000","type":"number","aliases":["45,000"],"features":{},"mentions":[{"text":"45,000","span":[17,23]}]},
    {"id":"e_dur","name":"unknown period","type":"duration","aliases":["unknown"],"features":{},"mentions":[{"text":"unknown period","span":[128,142]}]},
    {"id":"e_t","name":"T-1d","type":"timepoint","aliases":["yesterday"],"features":{},"mentions":[{"text":"yesterday (T-1d)","span":[168,185]}]},
    {"id":"e_actor","name":"controller","type":"actor","aliases":[],"features":{},"mentions":[{"text":"We","span":[187,189]}]},
    {"id":"e_week","name":"P7D","type":"duration","aliases":["for a week"],"features":{},"mentions":[{"text":"for a week","span":[205,215]}]},
    {"id":"e_auth","name":"authority","type":"recipient","aliases":["supervisory authority"],"features":{},"mentions":[{"text":"authority","span":[249,258]}]},
    {"id":"e_inds","name":"individuals","type":"recipient","aliases":["data subjects"],"features":{},"mentions":[{"text":"individuals","span":[263,274]}]},
    {"id":"e_uk","name":"UK","type":"jurisdiction","aliases":["United Kingdom"],"features":{},"mentions":[{"text":"UK","span":[290,292]}]},
    {"id":"e_enc","name":"encryption at rest","type":"other","aliases":[],"features":{"hypernyms":["security control"]},"mentions":[{"text":"encryption at rest","span":[305,323]}]},
    {"id":"e_log","name":"logging","type":"other","aliases":[],"features":{"hypernyms":["security control"]},"mentions":[{"text":"logging","span":[325,332]}]}
  ],
  "relations": [
    {"subj":"e_sys","pred":"contains","obj":"e_docs","evidence":{"quote":"S3 bucket ... customer PDFs","span":[3,37]},"confidence":0.9},
    {"subj":"e_docs","pred":"includes_field","obj":"e_name","evidence":{"quote":"containing names","span":[46,55]},"confidence":0.9},
    {"subj":"e_docs","pred":"includes_field","obj":"e_addr","evidence":{"quote":"addresses","span":[57,66]},"confidence":0.9},
    {"subj":"e_docs","pred":"includes_field","obj":"e_card","evidence":{"quote":"partial card numbers","span":[72,93]},"confidence":0.9},

    {"subj":"e_evt","pred":"involves","obj":"e_sys","evidence":{"quote":"publicly accessible","span":[100,119]},"confidence":0.9},
    {"subj":"e_evt","pred":"involves","obj":"e_docs","evidence":{"quote":"publicly accessible","span":[100,119]},"confidence":0.9},
    {"subj":"e_sys","pred":"publicly_accessible","obj":"e_evt","evidence":{"quote":"publicly accessible","span":[100,119]},"confidence":0.9},

    {"subj":"e_evt","pred":"approx_count_of_records","obj":"e_cnt","evidence":{"quote":"around 45,000","span":[10,23]},"confidence":0.9},
    {"subj":"e_evt","pred":"exposure_duration","obj":"e_dur","evidence":{"quote":"unknown period","span":[128,142]},"confidence":0.8},
    {"subj":"e_evt","pred":"discovered_at","obj":"e_t","evidence":{"quote":"discovered ... yesterday (T-1d)","span":[160,185]},"confidence":0.9},

    {"subj":"e_actor","pred":"plans_monitoring","obj":"e_week","evidence":{"quote":"considering monitoring for a week","span":[193,215]},"confidence":0.8},
    {"subj":"e_actor","pred":"intends_to_delay_notification","obj":"e_week","evidence":{"quote":"before deciding whether to notify","span":[216,248]},"confidence":0.8},
    {"subj":"e_actor","pred":"notify_authority","obj":"e_auth","evidence":{"quote":"whether to notify the authority","span":[228,258]},"confidence":0.5},
    {"subj":"e_actor","pred":"notify_individuals","obj":"e_inds","evidence":{"quote":"and individuals","span":[260,274]},"confidence":0.5},

    {"subj":"e_actor","pred":"located_in","obj":"e_uk","evidence":{"quote":"Jurisdiction: UK","span":[286,292]},"confidence":0.9},
    {"subj":"e_actor","pred":"security_control","obj":"e_enc","evidence":{"quote":"encryption at rest","span":[305,323]},"confidence":0.8},
    {"subj":"e_actor","pred":"security_control","obj":"e_log","evidence":{"quote":"logging","span":[325,332]},"confidence":0.8}
  ]
}


-- Few-shot example 2 --
CONTEXT:
“We transfer customer emails to a US vendor for support. SCCs are in place.”
Output: {
  "entities":[
    {"id":"e_act","name":"controller","type":"actor","aliases":[],"features":{},"mentions":[{"text":"We","span":[0,2]}]},
    {"id":"e_email","name":"customer emails","type":"data_item","aliases":[],"features":{"hypernyms":["personal data"]},"mentions":[{"text":"customer emails","span":[10,25]}]},
    {"id":"e_us","name":"US","type":"jurisdiction","aliases":["United States"],"features":{},"mentions":[{"text":"US","span":[40,42]}]},
    {"id":"e_vendor","name":"vendor","type":"recipient","aliases":["processor"],"features":{},"mentions":[{"text":"vendor","span":[44,50]}]},
    {"id":"e_purpose","name":"support","type":"purpose","aliases":[],"features":{},"mentions":[{"text":"support","span":[55,62]}]},
    {"id":"e_scc","name":"SCCs","type":"other","aliases":["standard contractual clauses"],"features":{"hypernyms":["safeguard"]},"mentions":[{"text":"SCCs","span":[64,68]}]}
  ],
  "relations":[
    {"subj":"e_act","pred":"transfers_to","obj":"e_vendor","evidence":{"quote":"transfer ... to a US vendor","span":[3,50]},"confidence":0.9},
    {"subj":"e_vendor","pred":"located_in","obj":"e_us","evidence":{"quote":"US vendor","span":[40,50]},"confidence":0.9},
    {"subj":"e_act","pred":"uses","obj":"e_email","evidence":{"quote":"transfer customer emails","span":[3,25]},"confidence":0.9},
    {"subj":"e_act","pred":"purpose","obj":"e_purpose","evidence":{"quote":"for support","span":[53,62]},"confidence":0.9},
    {"subj":"e_act","pred":"security_control","obj":"e_scc","evidence":{"quote":"SCCs are in place","span":[64,82]},"confidence":0.8}
  ]
}

Return only the JSON.
"""

REF_ER_SYS = """\
You are the SECOND-PASS ER extractor reviewing a previous ER extraction for compliance analysis.
Another ER extractor already produced JSON_1 from the same text, but some labels are still missing.
Your job: RE-EXTRACT the full ER graph (same schema as the first pass), focusing on recall of items that are
explicitly present in the CONTEXT. Do NOT invent facts. If the text does not support an item, simply omit it.

Output JSON (same as first pass):
{
  "entities":[
    {"id":"e1","name":"...","type":"data_item|actor|purpose|basis|recipient|jurisdiction|system|other|event|timepoint|duration|number",
     "aliases":["..."], "features":{"hypernyms":["..."]}, "mentions":[{"text":"...","span":[start,end]}]}
  ],
  "relations":[
    {"subj":"eX","pred":"...", "obj":"eY",
     "evidence":{"quote":"...", "span":[start,end]}, "confidence":0.0}
  ]
}

Guidance (recall targets; omit if not in text):
- event (breach/leak/publicly accessible/unauthorised access)
- timepoint (discovered/notification times; T-1d, dates, “within 72 hours”)
- duration (e.g., “unknown period”, “for a week”, ISO-like P7D)
- number (record counts; e.g., “45,000”)
- recipients (+ shares_with/transfers_to links)
- data_subject (patients/customers/users)
- purposes (profiling/lookalike, direct marketing, service improvement)
- cross border pair (recipient located_in third country + transfers_to)
- retention (duration + retention_period relation)
- security_controls (hashing/access/encryption as relations)
- Keep predicates/types consistent with the first pass; you MAY also use:
  breach_of_security, publicly_accessible, involves, discovered_at, exposure_duration,
  approx_count_of_records, includes_field, security_control,
  plans_monitoring, intends_to_delay_notification, shares_with, transfers_to, retention_period.
- Evidence spans are 0-based [start,end). Quote minimally. Avoid self-loops/duplicates.

CONTEXT:
<<< {TEXT} >>>

PREVIOUS ER (JSON_1):
<<< {JSON_1} >>>
"""

from typing import Dict, Any, List, Tuple

def _key_ent(e: Dict[str, Any]) -> Tuple[str, str, Tuple[int,int] | None]:
    name = (e.get("name") or "").strip().lower()
    etyp = (e.get("type") or "").strip().lower()
    span0 = None
    m = (e.get("mentions") or [])
    if m and isinstance(m[0].get("span"), list) and len(m[0]["span"]) == 2:
        span0 = tuple(m[0]["span"])
    return (name, etyp, span0)

def _key_rel(r: Dict[str, Any]) -> Tuple[str, str, str, Tuple[int,int] | None]:
    ev = r.get("evidence") or {}
    sp = ev.get("span")
    span = tuple(sp) if isinstance(sp, list) and len(sp) == 2 else None
    return (str(r.get("subj")), str(r.get("pred")), str(r.get("obj")), span)

def merge_er_graphs(base: Dict[str, Any], add: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "context_id": base.get("context_id") or add.get("context_id"),
        "hash": base.get("hash") or add.get("hash"),
        "entities": list(base.get("entities") or []),
        "relations": list(base.get("relations") or []),
    }
    seen_e = { _key_ent(e) for e in out["entities"] }
    for e in (add.get("entities") or []):
        k = _key_ent(e)
        if k in seen_e:
            # mentions만 병합
            for be in out["entities"]:
                if _key_ent(be) == k:
                    be_m = be.get("mentions") or []
                    for m in (e.get("mentions") or []):
                        if m not in be_m:
                            be_m.append(m)
                    be["mentions"] = be_m
                    break
        else:
            out["entities"].append(e); seen_e.add(k)

    seen_r = { _key_rel(r) for r in out["relations"] }
    for r in (add.get("relations") or []):
        k = _key_rel(r)
        if k not in seen_r:
            out["relations"].append(r); seen_r.add(k)
    return out

def graph_skeleton(g: Dict[str, Any]) -> Dict[str, Any]:
    """리플렉션 프롬프트에 넣을 때 토큰 절약 스켈레톤."""
    ents = []
    for e in (g.get("entities") or []):
        ents.append({
            "id": e.get("id"),
            "name": e.get("name"),
            "type": e.get("type"),
            "mentions": [{"text": (m.get("text") or e.get("name")), "span": m.get("span")}
                         for m in (e.get("mentions") or [])]
        })
    rels = []
    for r in (g.get("relations") or []):
        ev = r.get("evidence") or {}
        rels.append({"subj": r.get("subj"), "pred": r.get("pred"), "obj": r.get("obj"),
                     "evidence": {"quote": ev.get("quote"), "span": ev.get("span")}})
    return {"context_id": g.get("context_id"), "entities": ents, "relations": rels}


def build_er_user_payload(context_id: str,
                          context_text: str) -> str:
    """
    hypernym_hints: [{"mention":"ICD-10 diagnosis codes","span":[s,e],"hypernyms":["data concerning health","personal data"]}, ...]
    """
    # return json.dumps({"CONTEXT_ID": context_id, "CONTEXT": context_text, "HYPERNYM_HINTS": hypernym_hints or []}, ensure_ascii=False)
    return json.dumps({"CONTEXT_ID": context_id, "CONTEXT": context_text}, ensure_ascii=False)


# ===== 유틸: AMR → 하이퍼님 힌트 변환 =====

def amrs_to_hint_map(amrs: List[Dict[str, Any]], context_id: str) -> Dict[str, Any]:
    """
    return:
      {
        <mention_lower>: {
          "mention": "...",
          "span": [s,e] | None,
          "strong": [{"label": "...", "role": "...", "kind": "...", "confidence": 1.0, "frag_ids": [...]}, ...],
          "weak":   [{"label": "...", "role": "...", "kind": "...", "confidence": 0.66, "frag_ids": [...]}, ...]
        },
        ...
      }
    """
    out: Dict[str, Any] = {}
    for a in amrs or []:
        if (a.get("context_id") or a.get("contextId")) != context_id:
            continue
        mtxt = (a.get("mention_text") or "").strip()
        if not mtxt:
            continue
        key = mtxt.casefold()
        bucket = out.get(key) or {"mention": mtxt, "span": a.get("span"), "strong": [], "weak": []}
        # span은 최초 1회 채택
        if bucket.get("span") is None and a.get("span") is not None:
            bucket["span"] = a.get("span")
        # 제안 1건
        hyper = (a.get("hypernym") or "").strip()
        if not hyper:
            out[key] = bucket
            continue
        item = {
            "label": hyper,
            "role": (a.get("role") or None),
            "kind": (a.get("kind") or None),
            "confidence": float(a.get("confidence") or 0.0),
            "frag_ids": list(a.get("supporting_fragment_ids") or []),
        }
        if (a.get("strength") or "WEAK").upper() == "STRONG":
            # 중복 제거
            if not any(h["label"].casefold() == hyper.casefold() for h in bucket["strong"]):
                bucket["strong"].append(item)
        else:
            if not any(h["label"].casefold() == hyper.casefold() for h in bucket["weak"]):
                bucket["weak"].append(item)
        out[key] = bucket
    return out

def _flatten_hints_for_llm(hmap: Dict[str, Any], cap: int = 4) -> List[Dict[str, Any]]:
    simple = []
    for rec in hmap.values():
        labels = [h["label"] for h in rec["strong"]] + [h["label"] for h in rec["weak"]]
        # 순서 보존 dedup
        seen, uniq = set(), []
        for x in labels:
            k = x.casefold()
            if k not in seen:
                seen.add(k); uniq.append(x)
        simple.append({"mention": rec["mention"], "span": rec.get("span"), "hypernyms": uniq[:cap]})
    return simple

def print_graph_with_hypernyms(graph: Dict[str, Any], strong_only: bool = False) -> None:
    eid = graph.get("context_id")
    ents = graph.get("entities", [])
    rels = graph.get("relations", [])
    print(f"\n[Graph] context_id={eid}  entities={len(ents)}  relations={len(rels)}")

    # 엔티티 + 하이퍼님(메타) 보기
    for e in ents:
        name = e.get("name", "UNKNOWN")
        hlist = e.get("features", {}).get("hypernyms", [])
        # 표준화: 문자열이면 메타로 승격해서 보여주기
        rows = []
        for h in hlist:
            if isinstance(h, str):
                rows.append({"label": h, "strength": "WEAK", "confidence": 0.0,
                             "role": None, "lexical_match": None, "kind": None,
                             "source_mentions": [], "amr_ids": []})
            else:
                rows.append(h)
        if strong_only:
            rows = [r for r in rows if (r.get("strength") == "STRONG")]
        if not rows:
            print(f"[E] {name}  → hypernyms=[]")
            continue
        # 상위 3개만 보기 좋게
        rows = sorted(rows, key=lambda r: r.get("confidence", 0.0), reverse=True)[:3]
        desc = []
        for r in rows:
            bits = [r.get("label","?")]
            bits.append(f"{r.get('strength','WEAK')}")
            conf = r.get("confidence")
            if conf is not None:
                bits.append(f"c={conf:.2f}")
            role = r.get("role"); lm = r.get("lexical_match")
            if role: bits.append(f"role={role}")
            if lm: bits.append(f"lex={lm}")
            kind = r.get("kind")
            if kind: bits.append(f"kind={kind}")
            desc.append(" | ".join(bits))
        print(f"[E] {name}  → " + ";  ".join(desc))

    # 릴레이션 보기
    if rels:
        print("\n[Relations]")
        for r in rels[:30]:  # 너무 길면 30개 제한
            sj, pj, ob = r.get("subj"), r.get("pred"), r.get("obj")
            conf = r.get("confidence", 0.0)
            ev = r.get("evidence", {}) or {}
            quote = ev.get("quote")
            print(f" - ({sj}) -[{pj}]-> ({ob})  c={conf:.2f}" + (f"  q='{quote[:60]}...'" if quote else ""))

from typing import Dict, Any, List, Tuple, Optional

def _graph_skeleton(graph: Dict[str, Any]) -> Dict[str, Any]:
    """리플렉션 프롬프트에 넣을 때 토큰 절약: id/name/type/mentions.span만."""
    ents = []
    for e in (graph.get("entities") or []):
        ents.append({
            "id": e.get("id"),
            "name": e.get("name"),
            "type": e.get("type"),
            "mentions": [{"text": (m.get("text") or e.get("name")), "span": m.get("span")}
                         for m in (e.get("mentions") or [])]
        })
    rels = []
    for r in (graph.get("relations") or []):
        ev = r.get("evidence") or {}
        rels.append({
            "subj": r.get("subj"), "pred": r.get("pred"), "obj": r.get("obj"),
            "evidence": {"quote": ev.get("quote"), "span": ev.get("span")}
        })
    return {"context_id": graph.get("context_id"), "entities": ents, "relations": rels}

def _ekey(e: Dict[str, Any]) -> Tuple[str, str, Optional[Tuple[int,int]]]:
    name = (e.get("name") or "").strip().lower()
    etyp = (e.get("type") or "").strip().lower()
    span0 = None
    m = (e.get("mentions") or [])
    if m and isinstance(m[0].get("span"), list) and len(m[0]["span"]) == 2:
        span0 = tuple(m[0]["span"])
    return (name, etyp, span0)

def _rkey(r: Dict[str, Any]) -> Tuple[str, str, str, Optional[Tuple[int,int]]]:
    ev = r.get("evidence") or {}
    sp = ev.get("span")
    span = tuple(sp) if isinstance(sp, list) and len(sp) == 2 else None
    return (str(r.get("subj")), str(r.get("pred")), str(r.get("obj")), span)

def _merge_full_graphs(base: Dict[str, Any], add: Dict[str, Any]) -> Dict[str, Any]:
    """
    풀 ER JSON끼리 병합.
    - 엔티티는 (name,type,첫스팬) 키로 중복 판정 → 기존 id 유지, 신규는 id 충돌 없게 재부여
    - 리レー션은 (subj,pred,obj,스팬) 키로 중복 제거
    - add 쪽 엔티티가 중복일 경우, 해당 id를 base의 id로 **리맵**해서 리レー션에 반영
    """
    out = {
        "context_id": base.get("context_id") or add.get("context_id"),
        "hash": base.get("hash") or add.get("hash"),
        "entities": list(base.get("entities") or []),
        "relations": list(base.get("relations") or []),
    }

    # base 인덱스
    base_idx_by_key = { _ekey(e): e for e in out["entities"] }
    base_ids = { e.get("id") for e in out["entities"] }

    # add → base id 매핑
    id_map = {}

    # 1) 엔티티 병합
    for e in (add.get("entities") or []):
        k = _ekey(e)
        if k in base_idx_by_key:
            base_e = base_idx_by_key[k]
            id_map[e.get("id")] = base_e.get("id")
            # mentions merge
            be_m = base_e.get("mentions") or []
            for m in (e.get("mentions") or []):
                if m not in be_m:
                    be_m.append(m)
            base_e["mentions"] = be_m
        else:
            # 새 id 충돌 방지
            old_id = e.get("id")  
            new_id = e.get("id") or f"e{len(out['entities'])+1}"
            if new_id in base_ids:
                new_id = f"e{len(out['entities'])+1}"
            e = {**e, "id": new_id}
            out["entities"].append(e)
            base_idx_by_key[k] = e
            base_ids.add(new_id)
            if old_id:                                    # ← old_id가 있으면 old→new로 리맵
                id_map[old_id] = new_id

    # 2) 관계 병합 (id 리맵 반영)
    seen_r = { _rkey(r) for r in out["relations"] }
    for r in (add.get("relations") or []):
        sj = id_map.get(r.get("subj"), r.get("subj"))
        ob = id_map.get(r.get("obj"),  r.get("obj"))
        rr = {**r, "subj": sj, "obj": ob}
        k = _rkey(rr)
        if k not in seen_r:
            out["relations"].append(rr)
            seen_r.add(k)

    return out



# ===== 메인 클래스 =====

class ERTripleExtractor:
    """
    - 입력: context (id, text), 선택적으로 AMR(hypernym) 레코드 리스트
    - 처리: LLM 프롬프트로 엔티티/트리플 추출, (있다면) AMR을 엔티티 feature로 주입
    - 출력: 간단 JSON 그래프(dict)
    """
    def __init__(self,
                 chat_model: Optional[str] = None,
                 hypernym_mode: str = "prefer_strong",
                 include_hypernym_meta: bool = True):
        """
        hypernym_mode:
          - "prefer_strong" (기본): strong 있으면 strong만, 없으면 weak 사용
          - "strong_only": strong만, 없으면 비움
          - "all": strong+weak 모두
        include_hypernym_meta:
          - True: features.hypernyms 를 메타 객체 리스트로 저장
          - False: 라벨 문자열 리스트로만 저장
        """
        self.oai = OpenAIClient(model=chat_model)
        assert hypernym_mode in {"prefer_strong", "strong_only", "all"}
        self.hypernym_mode = hypernym_mode
        self.include_hypernym_meta = include_hypernym_meta

    def extract(self,
                context_id: str,
                context_text: str,
                amr_records: Optional[List[Dict[str, Any]]] = None,
                temperature: float = 0.0) -> Dict[str, Any]:

        # (지금 설계에서는 ER 단계에 힌트를 쓰지 않습니다)
        # 그래도 API 시그니처 유지 차원에서 빈 hints 전달
        user = build_er_user_payload(context_id, context_text)
        obj = self.oai.chat_json(ER_SYS, user, temperature=temperature)

        # --- 엔티티 정리 ---
        ents_in = obj.get("entities") or []
        entities: List[EREntity] = []
        id_set = set()
        for e in ents_in:
            eid = (e.get("id") or "").strip() or f"e{len(entities)+1}"
            if eid in id_set:
                eid = f"e{len(entities)+1}"
            id_set.add(eid)
            name = (e.get("name") or "").strip() or "UNKNOWN"
            etype = (e.get("type") or "other").strip().lower()
            aliases = [a for a in (e.get("aliases") or []) if isinstance(a, str)]
            feats = e.get("features") or {}
            # LLM이 넣어 줬을 수도 있는 문자열 리스트 hypernyms 정규화(중복 제거)
            hypers_in = [h for h in (feats.get("hypernyms") or []) if isinstance(h, str)]
            feats["hypernyms"] = sorted(list({(h or "").strip() for h in hypers_in if h}))

            # mentions
            mouts: List[ERMention] = []
            for m in (e.get("mentions") or []):
                txt = (m.get("text") or "").strip()
                span = m.get("span")
                span = tuple(span) if isinstance(span, list) and len(span) == 2 else None
                mouts.append(ERMention(text=txt or name, span=span))
            if not mouts:
                mouts.append(ERMention(text=name, span=None))

            entities.append(EREntity(
                id=eid, name=name, etype=etype, aliases=aliases, features=feats, mentions=mouts
            ))

        # # --- (선택) AMR 하이퍼님 주입 ---
        # if amr_records:
        #     self._inject_hypernyms_into_entity_list(
        #         context_id=context_id,
        #         entities=entities,
        #         amrs=amr_records,
        #         hypernym_mode=self.hypernym_mode,
        #         include_meta=self.include_hypernym_meta
        #     )

        # --- 관계 정리 ---
        rels_in = obj.get("relations") or []
        ent_ids = {e.id for e in entities}
        relations: List[ERRelation] = []
        for r in rels_in:
            sj = (r.get("subj") or "").strip()
            ob = (r.get("obj") or "").strip()
            if not sj or not ob or sj not in ent_ids or ob not in ent_ids:
                continue
            pred = (r.get("pred") or "").strip()
            ev = r.get("evidence") or {}
            quote = ev.get("quote")
            span = ev.get("span")
            span = tuple(span) if isinstance(span, list) and len(span) == 2 else None
            conf = float(r.get("confidence") or 0.0)
            relations.append(ERRelation(
                subj=sj, pred=pred, obj=ob,
                evidence_quote=quote, evidence_span=span,
                confidence=conf
            ))

        return {
            "context_id": context_id,
            "hash": sha256(context_text),
            "entities": [e.to_dict() for e in entities],
            "relations": [r.to_dict() for r in relations],
        }
    
    
    def extract_with_reflection(self,
                                context_id: str,
                                context_text: str,
                                amr_records: Optional[List[Dict[str, Any]]] = None,
                                temperature: float = 0.0,
                                n_reflect: int = 2,
                                reflect_temperature: float = 0.3) -> Dict[str, Any]:
        """
        1) ER_SYS로 1차 추출
        2) REFLECT_PROMPT로 n_reflect(기본 2회) 반복 재추출 → 매회 병합
        """
        # pass-0
        base = self.extract(context_id, context_text, amr_records, temperature=temperature)

        # pass-1..n (reflection)
        total_new_e, total_new_r = 0, 0
        for i in range(max(0, n_reflect)):
            # 0) 현재 그래프의 키 스냅샷(중복 방지용)
            before_e = { _ekey(e) for e in (base.get("entities") or []) }
            before_r = { _rkey(r) for r in (base.get("relations") or []) }

            # 1) 리플렉션 호출 (payload는 문자열 JSON으로)
            payload = {"TEXT": context_text, "JSON_1": _graph_skeleton(base)}
            obj_reflect = self.oai.chat_json(
                REF_ER_SYS,
                json.dumps(payload, ensure_ascii=False),
                temperature=reflect_temperature
            )
            if not isinstance(obj_reflect, dict):
                print(f"[reflect] ctx={context_id} pass={i+1} (skip: non-dict response)")
                continue

            # 2) 정규화 후 병합
            g_reflect = self._normalize_er_object(context_id, context_text, obj_reflect)
            merged = _merge_full_graphs(base, g_reflect)

            # 3) 병합 후 증분 계산
            after_e = { _ekey(e) for e in (merged.get("entities") or []) }
            after_r = { _rkey(r) for r in (merged.get("relations") or []) }
            delta_e = len(after_e - before_e)
            delta_r = len(after_r - before_r)
            total_new_e += delta_e
            total_new_r += delta_r

            # 4) 라벨 증분 출력
            if delta_e or delta_r:
                print(f"[reflect] ctx={context_id} pass={i+1} "
                      f"new_entities={delta_e} new_relations={delta_r} "
                      f"(tot_e={len(after_e)}, tot_r={len(after_r)})")
            else:
                print(f"[reflect] ctx={context_id} pass={i+1} no new labels")

            # 5) 증분 없으면 조기 종료
            if delta_e == 0 and delta_r == 0:
                break
            base = merged

        # 전체 리플렉션 요약
        if total_new_e or total_new_r:
            print(f"[reflect] ctx={context_id} total_new={total_new_e + total_new_r} "
                  f"(entities={total_new_e}, relations={total_new_r})")
        else:
            print(f"[reflect] ctx={context_id} no missing labels found in reflection")

        return base


    # (ERTripleExtractor 클래스 내부에 추가)
    def _normalize_er_object(self,
                             context_id: str,
                             context_text: str,
                             obj: Dict[str, Any]) -> Dict[str, Any]:
        # --- 엔티티 정리 (extract()와 동일 로직) ---
        ents_in = obj.get("entities") or []
        entities: List[EREntity] = []
        id_set = set()
        for e in ents_in:
            eid = (e.get("id") or "").strip() or f"e{len(entities)+1}"
            if eid in id_set:
                eid = f"e{len(entities)+1}"
            id_set.add(eid)
            name = (e.get("name") or "").strip() or "UNKNOWN"
            etype = (e.get("type") or "other").strip().lower()
            aliases = [a for a in (e.get("aliases") or []) if isinstance(a, str)]
            feats = e.get("features") or {}
            hypers_in = [h for h in (feats.get("hypernyms") or []) if isinstance(h, str)]
            feats["hypernyms"] = sorted(list({(h or "").strip() for h in hypers_in if h}))

            mouts: List[ERMention] = []
            for m in (e.get("mentions") or []):
                txt = (m.get("text") or "").strip()
                span = m.get("span")
                span = tuple(span) if isinstance(span, list) and len(span) == 2 else None
                mouts.append(ERMention(text=txt or name, span=span))
            if not mouts:
                mouts.append(ERMention(text=name, span=None))

            entities.append(EREntity(
                id=eid, name=name, etype=etype, aliases=aliases, features=feats, mentions=mouts
            ))

        # --- 관계 정리 ---
        rels_in = obj.get("relations") or []
        ent_ids = {e.id for e in entities}
        relations: List[ERRelation] = []
        for r in rels_in:
            sj = (r.get("subj") or "").strip()
            ob = (r.get("obj") or "").strip()
            if not sj or not ob or sj not in ent_ids or ob not in ent_ids:
                continue
            pred = (r.get("pred") or "").strip()
            ev = r.get("evidence") or {}
            quote = ev.get("quote")
            span = ev.get("span")
            span = tuple(span) if isinstance(span, list) and len(span) == 2 else None
            conf = float(r.get("confidence") or 0.0)
            relations.append(ERRelation(
                subj=sj, pred=pred, obj=ob,
                evidence_quote=quote, evidence_span=span,
                confidence=conf
            ))
        return {
            "context_id": context_id,
            "hash": sha256(context_text),
            "entities": [e.to_dict() for e in entities],
            "relations": [r.to_dict() for r in relations],
        }


    # -------------------- AMR → 엔티티 주입 헬퍼들 --------------------

    @staticmethod
    def _norm_name(s: str) -> str:
        import re
        s = (s or "").lower()
        s = re.sub(r"[-_/]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _inject_hypernyms_into_entity_list(self,
                                           context_id: str,
                                           entities: List[EREntity],
                                           amrs: List[Dict[str, Any]],
                                           hypernym_mode: str,
                                           include_meta: bool) -> None:
        """
        동일 context_id AMR에서 하이퍼님을 엔티티 feature로 주입.
        1) entity_id 직결 우선
        2) 없으면 이름 느슨 매칭(부분 포함 양방향)
        hypernym_mode: "strong_only" | "prefer_strong" | "all"
        """
        # 1) preprocess AMR rows
        cand = []
        for a in amrs:
            if (a.get("context_id") or a.get("contextId")) != context_id:
                continue
            hyper = (a.get("hypernym") or "").strip()
            if not hyper:
                continue
            strength = (a.get("strength") or "WEAK").upper()
            if hypernym_mode == "strong_only" and strength != "STRONG":
                continue
            cand.append(a)

        # 2) entity_id → entity 매핑
        ent_by_id = {e.id: e for e in entities}

        # 3) 먼저 entity_id가 있는 AMR을 주입
        for a in cand:
            eid = (a.get("entity_id") or "").strip()
            if eid and eid in ent_by_id:
                self._merge_hypernym_feature(ent_by_id[eid], a, include_meta)

        # 4) entity_id 없는 AMR은 이름 느슨 매칭
        for a in cand:
            eid = (a.get("entity_id") or "").strip()
            if eid and eid in ent_by_id:
                continue  # 이미 처리
            mkey = self._norm_name(a.get("mention_text"))
            if not mkey:
                continue
            for e in entities:
                name_n = self._norm_name(e.name)
                if not name_n:
                    continue
                if (mkey in name_n) or (name_n in mkey):
                    # prefer_strong 모드: 동일 mention에 strong이 하나라도 있으면 weak는 스킵
                    if hypernym_mode == "prefer_strong":
                        # 같은 mention_text에 대해 STRONG 존재 여부 확인
                        has_strong = any(
                            (x.get("mention_text") == a.get("mention_text")) and
                            (x.get("strength", "").upper() == "STRONG")
                            for x in cand
                        )
                        if has_strong and (a.get("strength","WEAK").upper() != "STRONG"):
                            continue
                    self._merge_hypernym_feature(e, a, include_meta)

    def _merge_hypernym_feature(self,
                                ent: EREntity,
                                amr: Dict[str, Any],
                                include_meta: bool) -> None:
        """
        같은 label은 confidence 큰 쪽을 유지. STRONG은 confidence에 +0.1 보정.
        include_meta=False면 라벨 문자열만.
        """
        # 준비
        ent.features = ent.features or {}
        ent.features.setdefault("hypernyms", [])

        label = (amr.get("hypernym") or "").strip()
        if not label:
            return

        strength = (amr.get("strength") or "WEAK").upper()
        base_conf = float(amr.get("confidence") or 0.0)
        conf_adj = base_conf + (0.1 if strength == "STRONG" else 0.0)

        if not include_meta:
            # 문자열 리스트 모드
            labels = set([x for x in ent.features["hypernyms"] if isinstance(x, str)])
            labels.add(label)
            ent.features["hypernyms"] = sorted(list(labels))
            return

        # 메타 객체 모드
        items: List[Dict[str, Any]] = []
        for it in ent.features["hypernyms"]:
            if isinstance(it, dict):
                items.append(it)
            elif isinstance(it, str):
                # 기존 문자열을 메타로 승격
                items.append({
                    "label": it, "strength": "WEAK", "confidence": 0.0,
                    "role": None, "lexical_match": None, "kind": None,
                    "source_mentions": [], "amr_ids": [], "supporting_fragment_ids": []
                })

        new_rec = {
            "label": label,
            "strength": strength,
            "confidence": conf_adj,
            "role": amr.get("role"),
            "lexical_match": amr.get("lexical_match"),
            "kind": amr.get("kind"),
            "source_mentions": [amr.get("mention_text")] if amr.get("mention_text") else [],
            "amr_ids": [amr.get("mention_id")] if amr.get("mention_id") else [],
            "supporting_fragment_ids": list(amr.get("supporting_fragment_ids") or []),
        }

        # 같은 라벨이면 confidence 큰 쪽으로 교체 + 필드 병합
        best: Dict[str, Dict[str, Any]] = {}
        for it in items + [new_rec]:
            lab = it.get("label")
            if not lab:
                continue
            cur = best.get(lab)
            if (not cur) or (it.get("confidence", 0.0) > cur.get("confidence", 0.0)):
                best[lab] = {
                    **it,
                    "source_mentions": list(set(it.get("source_mentions") or [])),
                    "amr_ids": list(set(it.get("amr_ids") or [])),
                    "supporting_fragment_ids": list(set(it.get("supporting_fragment_ids") or [])),
                }
            else:
                # 병합(출처/근거 set union)
                cur["source_mentions"] = list(set((cur.get("source_mentions") or []) + (it.get("source_mentions") or [])))
                cur["amr_ids"] = list(set((cur.get("amr_ids") or []) + (it.get("amr_ids") or [])))
                cur["supporting_fragment_ids"] = list(set((cur.get("supporting_fragment_ids") or []) + (it.get("supporting_fragment_ids") or [])))
                best[lab] = cur

        ent.features["hypernyms"] = list(best.values())




# ===== 사용 예시 =====
if __name__ == "__main__":
    sample = (
        "I'm the IT operations manager at a private hospital group in Lyon. "
        "We plan to export from the EHR a weekly file containing: patient discharge date, ICD-10 diagnosis codes, "
        "lab result flags (e.g., HbA1c>7), year of birth, sex, and 5-digit postcode, plus a stable hashed patient ID. "
        "The file will be ingested into our customer data platform to build lookalike audiences and to retarget discharged patients "
        "on social media via server-to-server integrations. We have not collected explicit, separate consent for using health data for marketing. "
        "Marketing proposes to rely on legitimate interests and to continue sending events to US-based ad vendors. "
        "We have not completed an updated SCC/TIA package for these transfers. "
        "Context details: Acting as a controller. Purposes include marketing and retargeting. Data involved includes health-related data, identifiers and contact details. "
        "Potential special categories: health. Data subjects: patients. Recipients include advertising vendor and social media platform. "
        "Cross-border transfer: US. Retention: 365d. Security controls: hashing and access controls. Sector: healthcare. Jurisdiction: EU, FR."
    )

    # (선택) 기존 AMR 결과가 있다면 여기에 넣으세요.
    amr_records = []  # [{"context_id":"ex_ctx_001","mention_text":"ICD-10 diagnosis codes","span":[...], "hypernym":"data concerning health","strength":"STRONG"}, ...]

    extractor = ERTripleExtractor(chat_model="gpt-4o")
    graph = extractor.extract(context_id="ex_ctx_001", context_text=sample, amr_records=amr_records, temperature=0.0)
    print(json.dumps(graph, ensure_ascii=False, indent=2))
