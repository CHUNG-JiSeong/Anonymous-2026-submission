# ctx_to_graph_wrapper.py
# -*- coding: utf-8 -*-
"""
외부 호출용 초경량 래퍼:
  run(contexts) -> (amrs, graphs)

의존:
  - context_normalizer.normalize_contexts
  - er_triple_extractor.ERTripleExtractor
"""

from __future__ import annotations
import re
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Tuple, Optional
from copy import deepcopy

from context_normalizer import ContextNormalizerV2
from context_graph import ERTripleExtractor


ContextsInput = Union[str, List[str], List[Dict[str, Any]]]

def _coerce_contexts(ctxs: ContextsInput, default_id_prefix: str = "ctx") -> List[Dict[str, Any]]:
    """str / List[str] / List[dict] 를 파이프라인 표준 입력으로 정규화."""
    out: List[Dict[str, Any]] = []
    if isinstance(ctxs, str):
        return [{"id": f"{default_id_prefix}_001", "direction": "Inbound", "text": ctxs}]
    if isinstance(ctxs, list):
        if not ctxs:
            return out
        if isinstance(ctxs[0], str):
            for i, t in enumerate(ctxs, start=1):
                out.append({"id": f"{default_id_prefix}_{i:03d}", "direction": "Inbound", "text": t})
            return out
        if isinstance(ctxs[0], dict):
            for i, d in enumerate(ctxs, start=1):
                out.append({
                    "id": d.get("id") or f"{default_id_prefix}_{i:03d}",
                    "direction": d.get("direction", "Inbound"),
                    "text": d.get("text", "")
                })
            return out
    raise TypeError("contexts 입력은 str, List[str], 또는 List[dict] 여야 합니다.")

def _norm(s: str) -> str:
    # 매우 가벼운 정규화: 소문자 + 연속 공백 축소 + 기호 제거(하이픈/슬래시 등은 공백화)
    s = (s or "").lower()
    s = re.sub(r"[-_/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _attach_hypernyms_to_graph(graph: dict, amrs: list) -> None:
    """
    같은 context_id의 AMR 중 status=proposed && hypernym 이 있는 것만 대상으로
    엔티티명과 AMR mention_text를 느슨히 매칭하여 entity.features.hypernyms 를 채운다.
    """
    ctx_id = graph.get("context_id")
    cand = [
        a for a in amrs
        if a.get("context_id") == ctx_id
        and a.get("status") == "proposed"
        and (a.get("mention_text") or "").strip()
        and (a.get("hypernym") or "").strip()
    ]

    # AMR 인덱스(정규화된 mention_text → [(hypernym, strength, confidence), ...])
    idx = {}
    for a in cand:
        m = _norm(a["mention_text"])
        idx.setdefault(m, []).append((
            a["hypernym"].strip(),
            a.get("strength", "WEAK"),
            float(a.get("confidence", 0.0))
        ))

    for ent in graph.get("entities", []):
        name_n = _norm(ent.get("name", ""))
        hyps = []

        # 1) 정확/부분 일치(양방향 포함)
        for m, items in idx.items():
            if not m: 
                continue
            if (m in name_n) or (name_n in m):
                for (h, strength, conf) in items:
                    hyps.append((h, strength, conf))

        # 2) 멘션별 top-1 가중(동일 hypernym이 다수면 max(conf) 채택)
        by_h = {}
        for h, strength, conf in hyps:
            # premise 기반 STRONG이면 +0.1 보너스(최종 정렬용, 임의 가중)
            adj = conf + (0.1 if strength == "STRONG" else 0.0)
            by_h[h] = max(by_h.get(h, 0.0), adj)

        # 3) 정렬 후 상위만 부여(너무 길어지지 않게 N 제한)
        top = [h for h, _ in sorted(by_h.items(), key=lambda kv: kv[1], reverse=True)[:5]]

        # 4) 엔티티에 병합
        ent.setdefault("features", {}).setdefault("hypernyms", [])
        before = set(ent["features"]["hypernyms"])
        ent["features"]["hypernyms"] = list(before.union(top))

def _inject_hypernyms_into_entities(graph: Dict[str, Any], amrs: List[Dict[str, Any]]) -> None:
    """
    동일 context_id의 AMR에서 하이퍼님을 읽어, 엔티티 features.hypernyms에 주입.
    - 엔티티명 ↔ AMR.mention_text를 느슨히 매칭(부분 포함 양방향)
    - 같은 hypernym이 여러 번 나오면 confidence 최대값으로 집계
    - 엔티티 features.hypernyms 스키마: [{"label","source_mentions","strength","confidence","amr_ids"} ...]
    """
    ctx_id = graph.get("context_id")
    amrs = [a for a in amrs if a.get("context_id") == ctx_id and a.get("status") == "proposed" and a.get("hypernym")]

    # AMR 인덱스(mention_text 정규화 → 레코드들)
    idx: Dict[str, List[Dict[str, Any]]] = {}
    for a in amrs:
        key = _norm(a.get("mention_text"))
        idx.setdefault(key, []).append(a)

    for ent in graph.get("entities", []):
        name_n = _norm(ent.get("name"))
        collected: Dict[str, Dict[str, Any]] = {}  # hypernym → aggregator
        for mkey, rows in idx.items():
            if not mkey:
                continue
            if (mkey in name_n) or (name_n in mkey):
                for r in rows:
                    h = r["hypernym"].strip()
                    strength = r.get("strength", "WEAK")
                    conf = float(r.get("confidence", 0.0))
                    mid = r.get("mention_id")
                    agg = collected.setdefault(h, {"label": h, "source_mentions": set(), "strength": strength,
                                                   "confidence": 0.0, "amr_ids": set()})
                    # premise 가중(약간)
                    if strength == "STRONG":
                        conf_adj = conf + 0.1
                    else:
                        conf_adj = conf
                    agg["confidence"] = max(agg["confidence"], conf_adj)
                    if r.get("mention_text"):
                        agg["source_mentions"].add(r["mention_text"])
                    if mid:
                        agg["amr_ids"].add(mid)

        # 정렬, top-N 제한
        items = sorted(collected.values(), key=lambda x: x["confidence"], reverse=True)[:5]
        # set → list
        for it in items:
            it["source_mentions"] = sorted(list(it["source_mentions"]))
            it["amr_ids"] = sorted(list(it["amr_ids"]))
        # 주입
        ent.setdefault("features", {}).setdefault("hypernyms", [])
        ent["features"]["hypernyms"] = items

@dataclass
class PipelineConfig:
    policy_graph_path: str
    chat_model: str = "gpt-5"
    embed_model: str = "text-embedding-3-small"
    seed: int = 42
    top_k: int = 8
    er_temperature: float = 0.0   # ER 추출 LLM 온도
    soft_guard=True          # ← 켠다
    strict_mode=True         # 필요 시 True


class ContextToGraph:
    """외부에 노출되는 단 하나의 클래스(파사드)."""
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self._triple_extractor = ERTripleExtractor(chat_model=self.cfg.chat_model)
        self._normalizer = ContextNormalizerV2(self.cfg.policy_graph_path,
                                              chat_model=self.cfg.chat_model,
                                              embed_model=self.cfg.embed_model,
                                              seed=self.cfg.seed,
                                              soft_guard=self.cfg.soft_guard,
                                              strict_mode=self.cfg.strict_mode
                                              )

    def run(self, contexts: ContextsInput) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        ctx_list = _coerce_contexts(contexts)

        graphs: List[Dict[str, Any]] = []
        all_amrs: List[Dict[str, Any]] = []

        for ctx in ctx_list:
            # 1) ER 추출
            g = self._triple_extractor.extract_with_reflection(
                context_id=ctx["id"],
                context_text=ctx["text"],
                amr_records=[],                      # ← (이제 필요 없음: ER에서 AMR 사용 안 함)
                temperature=self.cfg.er_temperature
            )

            # 2) 엔티티 기반 하이퍼님 추론(AMR)
            amrs_ent = self._normalizer.normalize_entities(
                context_id=ctx["id"],
                direction=ctx.get("direction","Inbound"),
                context_text=ctx["text"],
                entities=g.get("entities", []),
                top_k=self.cfg.top_k
            )
            all_amrs.extend(amrs_ent)

            # 3) 그래프에 하이퍼님 주입
            _inject_hypernyms_into_entities(g, amrs_ent)
            graphs.append(g)

        return all_amrs, graphs


# 함수형 원샷 API도 제공(선호 시 사용)
def run_contexts_to_graphs(
    contexts: ContextsInput,
    policy_graph_path: str,
    chat_model: str = "gpt-5",
    embed_model: str = "text-embedding-3-small",
    seed: int = 42,
    top_k: int = 8,
    er_temperature: float = 0.0
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cfg = PipelineConfig(
        policy_graph_path=policy_graph_path,
        chat_model=chat_model,
        embed_model=embed_model,
        seed=seed,
        top_k=top_k,
        er_temperature=er_temperature,
    )
    return ContextToGraph(cfg).run(contexts)

# --- add: pretty printer for graphs -----------------------------------------
def pretty_print_graph(graph: Dict[str, Any],
                       show_relations: bool = False,
                       only_strong: bool = False) -> None:
    """
    엔티티 노드의 feature들을 메타까지 보기 좋게 출력.
    - only_strong=True면 STRONG 하이퍼님만 보여줌
    - show_relations=True면 관계도 함께 출력
    """
    cid = graph.get("context_id")
    ents = graph.get("entities", [])
    rels = graph.get("relations", [])
    print(f"[Graph] context_id={cid} | entities={len(ents)} | relations={len(rels)}")

    for e in ents:
        name = e.get("name", "")
        etype = e.get("type", "other")
        print(f"[E] {name} (type={etype})")

        feats = e.get("features", {})
        hypers = feats.get("hypernyms", [])

        if not hypers:
            print("  └─ hypernyms: (none)")
        else:
            print("  └─ hypernyms:")
            # hypernyms가 문자열 리스트일 수도, 메타 객체 리스트일 수도 있음
            for h in hypers:
                if isinstance(h, str):
                    # include_hypernym_meta=False인 경우
                    print(f"     - {h}")
                    continue

                # 메타 객체 모드
                if only_strong and (h.get("strength","WEAK").upper() != "STRONG"):
                    continue
                label = h.get("label","")
                strength = h.get("strength")
                role = h.get("role")
                kind = h.get("kind")
                conf = h.get("confidence")
                lex = h.get("lexical_match")
                srcs = h.get("source_mentions") or []
                amr_ids = h.get("amr_ids") or []
                frags = h.get("supporting_fragment_ids") or []

                print(f"     - {label} "
                      f"[strength={strength}, role={role}, kind={kind}, "
                      f"conf={conf:.2f}]" if isinstance(conf, (int,float)) else
                      f"     - {label} [strength={strength}, role={role}, kind={kind}]")
                if lex:
                    print(f"         lexical_match={lex}")
                if srcs:
                    print(f"         source_mentions={srcs}")
                if amr_ids:
                    print(f"         amr_ids={amr_ids}")
                if frags:
                    print(f"         supporting_fragment_ids={frags}")

        # 다른 feature들도 함께 보여주고 싶으면 여기서 확장
        for k, v in feats.items():
            if k == "hypernyms":
                continue
            print(f"  └─ feature.{k} = {v}")

    if show_relations and rels:
        print("\n[Relations]")
        for r in rels:
            sj, pred, ob = r.get("subj"), r.get("pred"), r.get("obj")
            conf = r.get("confidence", 0.0)
            ev = r.get("evidence") or {}
            quote = ev.get("quote")
            span = ev.get("span")
            line = f"  {sj} -[{pred}]-> {ob} (conf={conf:.2f})"
            print(line)
            if quote:
                print(f"    • evidence: {quote}")
            if span:
                print(f"    • span: {span}")


def _normalize_hypernyms(hlist: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """
    엔티티 features.hypernyms가 문자열/딕셔너리 섞여 있어도
    표준 스키마로 정규화해 반환.
    """
    if not hlist:
        return []
    normed = []
    for h in hlist:
        if isinstance(h, str):
            normed.append({
                "label": h.strip(),
                "strength": None,
                "confidence": 0.0,
                "role": None,
                "lexical_match": None,
                "kind": None,
                "source_mentions": [],
                "amr_ids": [],
                "supporting_fragment_ids": [],
            })
        elif isinstance(h, dict):
            base = {
                "label": None,
                "strength": None,
                "confidence": 0.0,
                "role": None,
                "lexical_match": None,
                "kind": None,
                "source_mentions": [],
                "amr_ids": [],
                "supporting_fragment_ids": [],
            }
            # 사용자가 넣은 키가 우선
            base.update(h)
            # 타입 보정
            base["label"] = (base.get("label") or "").strip() or None
            base["strength"] = (base.get("strength") or None)
            base["role"] = (base.get("role") or None)
            base["lexical_match"] = (base.get("lexical_match") or None)
            base["kind"] = (base.get("kind") or None)
            base["confidence"] = float(base.get("confidence") or 0.0)
            base["source_mentions"] = list(base.get("source_mentions") or [])
            base["amr_ids"] = list(base.get("amr_ids") or [])
            base["supporting_fragment_ids"] = list(base.get("supporting_fragment_ids") or [])
            normed.append(base)
    # label 없는 항목 제거 + 라벨 중복은 confidence 큰 쪽만 유지
    best = {}
    for it in normed:
        lab = it.get("label")
        if not lab:
            continue
        if (lab not in best) or (it.get("confidence", 0.0) > best[lab].get("confidence", 0.0)):
            best[lab] = it
    return list(best.values())

def export_graph_json(graph: Dict[str, Any],
                      strong_only: bool = False,
                      compact: bool = False) -> str:
    """
    ERTripleExtractor/ContextToGraph가 만든 단일 그래프 dict를
    하이퍼님 메타까지 포함해 JSON 문자열로 직렬화.
    strong_only=True면 STRONG만 남김.
    compact=True면 들여쓰기 없이 압축.
    """
    g = deepcopy(graph)

    # 엔티티 feature 정리
    for ent in g.get("entities", []):
        feats = ent.setdefault("features", {})
        feats["hypernyms"] = _normalize_hypernyms(feats.get("hypernyms"))
        if strong_only:
            feats["hypernyms"] = [
                h for h in feats["hypernyms"]
                if (h.get("strength") or "").upper() == "STRONG"
            ]

    # 관계(evidence) 필드는 그대로 보존 (이미 직렬화 가능한 형태)
    return json.dumps(
        g,
        ensure_ascii=False,
        indent=None if compact else 2,
        sort_keys=False
    )

def save_graph_json(graph: Dict[str, Any],
                    path: str,
                    strong_only: bool = False,
                    compact: bool = False) -> None:
    """
    위 export_graph_json을 파일 저장 버전으로 래핑.
    """
    s = export_graph_json(graph, strong_only=strong_only, compact=compact)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)
