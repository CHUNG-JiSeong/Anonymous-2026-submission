# -*- coding: utf-8 -*-
"""
Compliance Gate (Graph-of-Thought, bidirectional references, unlimited depth)
Default LLM backend: OpenAI gpt-4o via Responses API.

- Anchors(UoE) & CU ranking: policy-agnostic, deterministic
- Step 2: LLM GoT over constraint → condition (direct exceptions) → context alignment
- Step 3: Bidirectional REFERS/DERIVES reference closure (no hop limit) → LLM indirect exception check → override

Env:
  pip install openai
  export OPENAI_API_KEY=sk-...

CLI:
  python compliance_gate_got_openai.py --policy gdpr_snapshot.json --context context_graphs.json

"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Iterable, Set
import numpy as np
import json, re, collections, uuid, argparse, os, sys, textwrap
from pathlib import Path
import hashlib, tempfile
import time as _t, datetime, pathlib, uuid as _uuid
from collections import Counter
from sentence_transformers import CrossEncoder
import torch

TRIAGE_BATCH = 24
CHECKLIST_BATCH = 16
TAU_SEM = 0.55

RAW_DUMP = True                              # ← 필요 없으면 False
RAW_DIR  = Path("monitor/raw_llm"); RAW_DIR.mkdir(parents=True, exist_ok=True)

def _raw_dump(tag: str, anchor_id: str, sys_msg: str, payload: dict, resp):
    """tag in {'listwise','refs'}"""
    try:
        ts = int(_t.time()*1000)
        fn = RAW_DIR / f"{ts}_{anchor_id}_{tag}.txt"
        with open(fn, "w", encoding="utf-8") as f:
            f.write("=== SYS ===\n"); f.write(sys_msg); f.write("\n\n")
            f.write("=== PAYLOAD (truncated) ===\n")
            s_payload = json.dumps(payload, ensure_ascii=False)[:12000]  # 과도 방지
            f.write(s_payload); f.write("\n\n")
            f.write("=== RESPONSE RAW ===\n")
            if isinstance(resp, (dict, list)):
                f.write(json.dumps(resp, ensure_ascii=False, indent=2))
            else:
                f.write(str(resp))
        print(f"[raw] dump -> {fn}")
    except Exception as e:
        print(f"[raw][err] {e}")
# -------------------------------------------------------------------

def _as_list(x):
    if isinstance(x, list): return x
    if x is None: return []
    return [x]

# ------------------------------
# Data structures
# ------------------------------
@dataclass
class Entity:
    id: str
    name: str
    type: str
    features: Dict[str, Any] = field(default_factory=dict)
    mentions: List[Dict[str, Any]] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    def hypernyms(self) -> List[Dict[str, Any]]:
        return self.features.get("hypernyms", []) if self.features else []
    def hypernym_labels(self, strength: Optional[str] = None) -> List[str]:
        labs = []
        for h in self.hypernyms():
            if strength and h.get("strength") != strength: continue
            labs.append(str(h.get("label","")).lower())
        return labs

@dataclass
class Relation:
    subj: str
    pred: str
    obj: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

@dataclass
class Anchor:  # Unit of Evaluation (행위 중심 단위)
    id: str
    actor: Optional[str]
    pred: str
    obj: Optional[str]
    pred_frame: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)
    evidence: Optional[Dict[str, Any]] = None
    def as_tuple(self) -> Tuple[Optional[str], str, Optional[str]]:
        return (self.actor, self.pred, self.obj)

@dataclass
class Plan:
    cu_id: str
    cu_type: str  # actor_cu/meta_cu
    subject_text: str
    constraint_texts: List[str]
    condition_text: Optional[Any]
    context_text: Optional[str]
    intent: str               # PROHIBIT | ALLOW_IF | APPLIES | UNSPEC
    action_hint: Optional[str]
    patient_hints: List[str]

@dataclass
class Decision:
    cu_id: str
    article: str
    cu_type: str
    verdict: str              # NON_COMPLIANT | COMPLIANT | NOT_APPLICABLE | INSUFFICIENT
    score: float
    why: str
    evidence: List[str] = field(default_factory=list)
    anchor: Optional[Anchor] = None

class _RunReport:
    def __init__(self, max_show=8, to_file=True, file_path="monitor/compliance_run_report.json"):
        self.max_show = max_show
        self.to_file = to_file
        self.file_path = file_path
        self.meta = {}
        # 앵커별: {anc_id: {"anchor": {...}, "preselect":[{cu_id,score,subject}], "rerank":[...],
        #                   "listwise":[{cu_id, verdict, score, why}], "refs":[{base_cu_id, overridden, final_verdict, why}]}}
        self.anc = {}
        # 최종 결정(기사 단위)
        self.final = []

    def set_meta(self, **kw):
        self.meta.update(kw)

    def _get(self, anc_id):
        if anc_id not in self.anc:
            self.anc[anc_id] = {"anchor":{}, "preselect":[], "rerank":[], "listwise":[], "refs":[]}
        return self.anc[anc_id]

    def set_anchor(self, anc_id, anchor_blob):
        self._get(anc_id)["anchor"] = anchor_blob

    def add_preselect(self, anc_id, rows):
        self._get(anc_id)["preselect"] = rows

    def add_rerank(self, anc_id, rows):
        self._get(anc_id)["rerank"] = rows

    def add_listwise(self, anc_id, rows):
        self._get(anc_id)["listwise"] = rows

    def add_refs(self, anc_id, rows):
        if rows:
            self._get(anc_id)["refs"] = rows

    def set_final(self, decisions):
        # decisions: List[Decision]
        self.final = [{
            "article": d.article, "cu_id": d.cu_id, "verdict": d.verdict,
            "score": float(d.score), "why": d.why, "anchor_id": getattr(d.anchor, "id", None)
        } for d in decisions]

    @staticmethod
    def _short(s, n=90):
        s = "" if s is None else str(s)
        return s if len(s) <= n else s[:n] + "…"

    def print_human(self, pol, max_show=None):
        ms = max_show or self.max_show
        print("\n========== COMPLIANCE RUN REPORT ==========")
        if self.meta:
            print("[meta]", {k: v for k, v in self.meta.items()})
        print(f"[anchors] {len(self.anc)} anchors")
        # 앵커별 요약
        for anc_id, rec in self.anc.items():
            a = rec.get("anchor", {})
            head = f"{anc_id} | pred={a.get('predicate','')} actor={a.get('actor_type','')} object={a.get('object_type','')}"
            print(f"\n-- Anchor: {head}")
            # 1) 프리셀렉트
            pre = rec.get("preselect", [])[:ms]
            if pre:
                print(f"  [preselect x{len(rec.get('preselect',[]))}] top{len(pre)}:")
                for row in pre:
                    print(f"    - {row['cu_id']}  s={row['score']:.3f}  subj='{self._short(row.get('subject',''))}'")
            else:
                print("  [preselect] (none)")
            # 2) 리랭크
            rr = rec.get("rerank", [])[:ms]
            if rr:
                print(f"  [rerank    x{len(rec.get('rerank',[]))}] top{len(rr)}:")
                for row in rr:
                    print(f"    - {row['cu_id']}  s={row['score']:.3f}  subj='{self._short(row.get('subject',''))}'")
            else:
                print("  [rerank] (none)")
            # 3) 컴플라이언스 판단(listwise)
            lw = rec.get("listwise", [])[:ms]
            if lw:
                print(f"  [listwise  x{len(rec.get('listwise',[]))}] top{len(lw)}:")
                for r in lw:
                    print(f"    - {r['cu_id']}  verdict={r['verdict']} score={r.get('score',0):.2f} why='{self._short(r.get('why',''))}'")
            else:
                print("  [listwise] (none)")
            # 4) 참조 재검사
            rf = rec.get("refs", [])[:ms]
            if rf:
                print(f"  [refs      x{len(rec.get('refs',[]))}] top{len(rf)}:")
                for r in rf:
                    print(f"    - base={r['base_cu_id']}  overridden={r['overridden']}  final={r['final_verdict']} why='{self._short(r.get('why',''))}'")
        # 최종 결과
        print("\n== Final Decisions (by article) ==")
        if not self.final:
            print("  (empty)")
        else:
            for d in self.final:
                print(f"  - {d['article']}: {d['verdict']} (score={d['score']:.2f}) cu={d['cu_id']}  why='{self._short(d['why'])}'")

    def dump_json(self):
        if not self.to_file:
            return
        Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": self.meta,
            "anchors": list(self.anc.values()),
            "final": self.final,
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        }
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[report] written to {self.file_path}")
# ============================================================================

@dataclass
class LegalCrossReranker:
    """
    Cross-encoder 기반 리랭커.
    - model_name: CE 헤드가 달린 체크포인트(권장: 'BAAI/bge-reranker-v2-m3').
      * 법률 도메인에서 미세학습한 Legal-BERT CE가 있다면 그 경로로 교체.
    """
    model_name: str = "BAAI/bge-reranker-v2-m3"
    max_length: int = 512
    batch_size: int = 32

    def __post_init__(self):
        assert CrossEncoder is not None, "pip install sentence-transformers 필요"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(self.model_name, max_length=self.max_length, device=self.device)
        if self.device == "cuda":
            # 메모리 여유 있으면 반정밀도 권장(지원 모델 한정)
            try: 
                self.model.model.half()
                print("[diag] reranker: fp16 enabled")
            except Exception:
                pass
        print(f"[diag] reranker.device={self.device}, batch_size={self.batch_size}")

    @staticmethod
    def _anchor_text(anc, ctx) -> str:
        act = ctx.get_entity(anc.actor) if anc.actor else None
        obj = ctx.get_entity(anc.obj) if anc.obj else None
        q = (anc.evidence or {}).get("quote", "")
        return f"{anc.pred} ; actor:{(act.type if act else '')} ; object:{(obj.type if obj else '')} ; evidence:{q}"

    @staticmethod
    def _cu_text_from_attrs(attrs: Dict[str, Any], cand_retriever) -> str:
        subj = attrs.get("subject") or ""
        cons = attrs.get("constraint")
        cons = " ; ".join(cons) if isinstance(cons, list) else (cons or "")
        cond = cand_retriever._cond_to_text(attrs.get("condition"))
        ctx  = cand_retriever._cond_to_text(attrs.get("context"))
        # 길이 과도 방지
        if len(cond) > 300: cond = cond[:300]
        if len(ctx)  > 300: ctx  = ctx[:300]
        return f"{subj} || {cons} || {cond} || {ctx}"

    def rerank_one_anchor(
        self,
        anchor, ctx, policy,
        cand_list: List[Tuple[str, float]],   # [(cu_id, pre_score)]
        cand_retriever,
        final_k: int = 8
    ) -> List[Tuple[str, float]]:
        """cand_list를 CE 점수로 재정렬하고 상위 final_k 반환."""
        if not cand_list:
            return []

        query = self._anchor_text(anchor, ctx)
        cu_ids, docs = [], []
        for cu_id, _ in cand_list:
            node  = policy.get_cu(cu_id)
            attrs = (node.get("attrs") or {})
            cu_ids.append(cu_id)
            docs.append(self._cu_text_from_attrs(attrs, cand_retriever))

        pairs = [(query, d) for d in docs]
        scores = self.model.predict(pairs, batch_size=self.batch_size)  # numpy array
        # 0..1 정규화
        scores = (scores - float(np.min(scores))) / (float(np.ptp(scores)) + 1e-6)
        reranked = sorted(zip(cu_ids, scores), key=lambda x: x[1], reverse=True)
        return reranked[:final_k]

# === DIAG LIGHT ===
class _Diag:
    n = {}      # counters
    v = {}      # sums (for averages)
    t = {}      # times (seconds)
    t0 = None
    @classmethod
    def begin(cls): cls.t0 = _t.perf_counter()
    @classmethod
    def inc(cls, k, by=1): cls.n[k] = cls.n.get(k, 0) + by
    @classmethod
    def add(cls, k, val): cls.v[k] = cls.v.get(k, 0.0) + float(val)
    @classmethod
    def time_call(cls, key, fn, *a, **kw):
        s = _t.perf_counter(); r = fn(*a, **kw); e = _t.perf_counter()
        cls.t[key] = cls.t.get(key, 0.0) + (e - s); cls.inc(f"calls.{key}")
        return r
    @classmethod
    def report(cls, enable_meta=None):
        run = (_t.perf_counter() - (cls.t0 or _t.perf_counter()))
        A = cls.n.get("anchors", 0)
        L1 = cls.n.get("calls.llm_listwise", 0)
        L2 = cls.n.get("calls.llm_refs", 0)
        meta = 1 if enable_meta else 0
        total_llm = L1 + L2 + meta
        k_pre_avg = (cls.v.get("K_pre_sum", 0.0) / A) if A else 0.0
        k_fin_avg = (cls.v.get("K_final_sum", 0.0) / A) if A else 0.0
        p_v = (L2 / A) if A else 0.0
        print(f"[diag] anchors={A}  preselect_K_avg={k_pre_avg:.2f}  final_K_avg={k_fin_avg:.2f}  p_v(violation_rate)={p_v:.2f}")
        print(f"[diag] llm_calls: listwise={L1}  refs={L2}  meta={(meta or 0)}  total={total_llm}")
        for k, sec in sorted(cls.t.items()):
            print(f"[diag] time.{k}={sec:.3f}s")
        print(f"[diag] run_time.total={run:.3f}s")
def diag_time_call(key, fn, *a, **kw):  # one-liner용 래퍼
    return _Diag.time_call(key, fn, *a, **kw)
# === /DIAG LIGHT ===

class DiskVecCache:
    def __init__(self, root: str = None):
        self.root = Path(root or "cache").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def cu_subj_paths(self, policy_name: str):
        base = self.root / f"{policy_name}_cache"
        return base.with_suffix(".npy"), base.with_suffix(".meta.json"), base.with_suffix(".ids.json")

    def load_cu_subj(self, policy_name: str):
        npy, meta_j, ids_j = self.cu_subj_paths(policy_name)
        try:
            # 최소 크기/동반 파일 확인
            if npy.exists() and npy.stat().st_size > 64 and meta_j.exists() and ids_j.exists():
                import numpy as np, json
                vecs = np.load(npy, mmap_mode="r")      # 손상 시 EOFError/ValueError
                meta  = json.loads(meta_j.read_text("utf-8"))
                cu_ids= json.loads(ids_j.read_text("utf-8"))
                # 간단한 무결성 점검
                if len(cu_ids) != getattr(vecs, "shape", (0,))[0]:
                    raise ValueError(f"ids({len(cu_ids)}) != vecs({getattr(vecs,'shape',('?','?'))})")
                return cu_ids, vecs, meta
        except Exception as e:
            print(f"[diag] cache corrupt for {npy}: {e} → rebuilding…")
        # 손상/불일치면 깨끗이 지우고 재빌드 유도
        for p in (npy, meta_j, ids_j):
            try: p.unlink()
            except Exception: pass
        return None

    def save_cu_subj(self, policy_name: str, cu_ids, vecs: "np.ndarray", meta: dict):
        npy, meta_j, ids_j = self.cu_subj_paths(policy_name)
        import numpy as np, tempfile, os, json
        # 임시 파일에 먼저 기록
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tmp = tf.name
        try:
            np.save(tmp, vecs)
            os.replace(tmp, npy)  # 원자적 교체
        finally:
            try: os.remove(tmp)
            except Exception: pass
        meta_j.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
        ids_j .write_text(json.dumps(list(cu_ids), ensure_ascii=False), "utf-8")



def _l2norm(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / n

def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


# --- [NEW] PredicateClusterer (얇은 래퍼: 명시적 이름만 제공) ---
@dataclass
class PredicateClusterer:
    """
    relation 술어 + 주변 맥락(증거 문장, actor/object 타입)을 임베딩해 KMeans로 군집,
    최근접 중심으로 프레임 라벨을 할당한다.
    - k=None이면 k_range에서 자동 선택(평균 최근접 중심 코사인 최대화).
    - sklearn 미설치 시 간단 폴백(lexical label).
    """
    def __init__(self,
                 embedder,              # SemanticEmbedder
                 k: int | None = None,
                 k_range: tuple[int,int] = (6, 12),
                 random_state: int = 42,
                 n_init: int = 10,
                 cache_path: str | None = None):
        self.E = embedder
        self.k = k
        self.k_range = k_range
        self.random_state = random_state
        self.n_init = n_init
        self.cache_path = cache_path
        self.centroids: np.ndarray | None = None
        self.labels: list[str] | None = None

    def _feat(self, pred: str, ev: str, a_type: str, o_type: str) -> str:
        # 컨텍스트를 간단히 합성(과도한 토큰 방지 위해 증거는 앞부분만)
        ev = (ev or "")
        if len(ev) > 160:
            ev = ev[:160]
        return f"{pred} || {a_type}->{o_type} || {ev}"

    def fit(self, ctx):
        # 1) 피처 텍스트 구성
        feats, metas = [], []
        for r in ctx.relations:
            a = ctx.entities.get(r.subj); o = ctx.entities.get(r.obj)
            ev = (r.evidence or {}).get("quote", "")
            feats.append(self._feat(r.pred, ev, getattr(a, "type", ""), getattr(o, "type", "")))
            metas.append({"pred": r.pred})

        if not feats:
            # 데이터가 없으면 빈 상태
            self.centroids, self.labels = None, None
            return

        # 2) 임베딩 + 정규화
        X = np.vstack(self.E.encode(feats))
        Xn = _l2norm(X)

        # 3) KMeans 군집 (자동 k 선택 또는 고정 k)
        n = Xn.shape[0]
        k_min, k_max = self.k_range
        k_min = max(2, min(k_min, n))
        k_max = max(k_min, min(k_max, n))
        candidate_ks = [self.k] if self.k else list(range(k_min, k_max + 1))

        best = (None, -1.0, None)  # (km, score, centroids)
        try:
            from sklearn.cluster import KMeans, MiniBatchKMeans
            for k in candidate_ks:
                # n이 큰 경우 MiniBatchKMeans 사용
                algo = MiniBatchKMeans if n > 2000 else KMeans
                km = algo(n_clusters=k, n_init=self.n_init, random_state=self.random_state)
                labels = km.fit_predict(Xn)
                C = km.cluster_centers_
                Cn = _l2norm(C)
                # 평균 최근접 중심 코사인 점수(간단/빠름)
                sc = float(np.mean(np.max(Xn @ Cn.T, axis=1)))
                if sc > best[1]:
                    best = (km, sc, Cn)
            km, _, centroids = best
            self.centroids = centroids
            # 4) 클러스터 이름: 각 클러스터의 상위 술어로 요약
            names = []
            for i in range(self.centroids.shape[0]):
                preds = [metas[j]["pred"] for j in range(n) if km.labels_[j] == i]
                top = Counter(preds).most_common(1)[0][0] if preds else f"cluster_{i}"
                names.append(top)
            self.labels = names
        except Exception:
            # sklearn 없는 환경: 폴백(프레임=대표 술어 텍스트)
            uniq = [m["pred"] for m in metas]
            uniq = [p for p, _ in Counter(uniq).most_common()]
            self.centroids = None
            self.labels = uniq[: min(len(uniq), (self.k or k_max))]

        # (선택) 캐시 저장
        if self.cache_path and self.centroids is not None and self.labels is not None:
            try:
                import json, os
                np.save(self.cache_path + ".centroids.npy", self.centroids)
                with open(self.cache_path + ".labels.json", "w", encoding="utf-8") as f:
                    json.dump(self.labels, f, ensure_ascii=False)
            except Exception:
                pass

    def label(self, pred: str, ev: str = "", a_type: str = "", o_type: str = "") -> str:
        # 학습 안 됐거나 폴백이면 술어 텍스트를 그대로 라벨로 사용
        if self.centroids is None or self.labels is None:
            return _norm(pred) or "other"
        v = self.E.encode([self._feat(pred, ev, a_type, o_type)])[0]
        v = v / (np.linalg.norm(v) + 1e-9)
        sims = self.centroids @ v  # (k,)
        idx = int(np.argmax(sims))
        return f"{idx}:{self.labels[idx]}"

    def similarity(self, pred_a: str, pred_b: str) -> float:
        va = self.E.encode([pred_a])[0]; vb = self.E.encode([pred_b])[0]
        va = va / (np.linalg.norm(va) + 1e-9)
        vb = vb / (np.linalg.norm(vb) + 1e-9)
        return float(np.dot(va, vb))

class AnchorExtractor:
    def __init__(self, clusterer: PredicateClusterer):
        self.clusterer = clusterer
    def __call__(self, ctx):
        anchors = []
        for r in ctx.iter_relations():
            a = ctx.get_entity(r.subj); o = ctx.get_entity(r.obj)
            ev = (r.evidence or {}).get("quote", "")
            frame = self.clusterer.label(r.pred, ev, getattr(a, "type", ""), getattr(o, "type", ""))
            anchors.append(Anchor(
                id=f"anc_{uuid.uuid4().hex[:8]}",
                actor=r.subj, pred=r.pred, obj=r.obj,
                pred_frame=frame, evidence=r.evidence
            ))
        # dedup by (actor, frame, obj)
        uniq = {}
        for a in anchors:
            key = (a.actor, a.pred_frame or a.extras.get("pred_cluster"), a.obj)
            uniq[key] = a
        return list(uniq.values())


# ------------------------------
# Utils
# ------------------------------
def _normalize_text(t: Optional[str]) -> str:
    if isinstance(t, (dict, list)):
        t = json.dumps(t, ensure_ascii=False)
    return re.sub(r"\s+"," ", (t or "")).strip().lower()

def _tokenize(t: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", _normalize_text(t))

def _contains_neg(text: str) -> bool:
    t = _normalize_text(text)
    cues = [" not "," no ","without "," absent "," missing "," lack "," lack of "]
    return any(c in t for c in cues) or t.startswith("no ")

def _article_from_cu_id(cu_id: str) -> str:
    m = re.search(r"/ARTICLE:([0-9]+)", cu_id)
    return f"Article {m.group(1)}" if m else "Unknown Article"

def _split_subject_action(text: str) -> Tuple[Optional[str], Optional[str]]:
    s = _normalize_text(text)
    if "processing of" in s: return "process", s.split("processing of",1)[1].strip()
    if s == "processing":    return "process", None
    if "transfer of" in s:   return "transfer", s.split("transfer of",1)[1].strip()
    if s == "transfer":      return "transfer", None
    return None, None

def _ensure_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(e) for e in x if e]
    return [str(x)]

# ------------------------------
# Graph wrappers
# ------------------------------
class ContextGraph:
    def __init__(self, data: Dict[str, Any]):
        self.entities: Dict[str, Entity] = {e["id"]: Entity(**{**e, "features": e.get("features", {}), "mentions": e.get("mentions", []), "aliases": e.get("aliases", [])}) for e in data.get("entities", [])}
        self.relations: List[Relation] = [Relation(**r) for r in data.get("relations", [])]
    def get_entity(self, eid: str) -> Optional[Entity]:
        return self.entities.get(eid)
    def iter_relations(self) -> Iterable[Relation]:
        return iter(self.relations)

class PolicyGraph:
    def __init__(self, data: Dict[str, Any]):
        self.nodes = data.get("nodes", [])
        self.edges = data.get("edges", [])
        self.cus: Dict[str, Dict[str, Any]] = {n["id"]: n for n in self.nodes if n.get("kind")=="compliance_unit"}
        self.by_type: Dict[str, List[str]] = collections.defaultdict(list)
        for nid, n in self.cus.items(): self.by_type[n.get("type","actor_cu")].append(nid)
        # adjacency (bidirectional)
        self._outs: Dict[str, List[Dict[str,str]]] = collections.defaultdict(list)
        self._ins : Dict[str, List[Dict[str,str]]] = collections.defaultdict(list)
        for e in self.edges:
            if e.get("etype") in ("REFERS_TO","DERIVES"):
                self._outs[e["src"]].append(e)
                self._ins [e["dst"]].append(e)
    def get_cu(self, cu_id: str) -> Dict[str, Any]:
        return self.cus[cu_id]
    def iter_cus(self, cu_type: Optional[str]=None) -> Iterable[Tuple[str, Dict[str, Any]]]:
        ids = self.by_type.get(cu_type, list(self.cus.keys())) if cu_type else self.cus.keys()
        for i in ids:
            yield i, self.cus[i]
    def neighbors_bidir(self, cu_id: str, etypes=("REFERS_TO","DERIVES")) -> List[str]:
        out = [e["dst"] for e in self._outs.get(cu_id, []) if e.get("etype") in etypes]
        inn = [e["src"] for e in self._ins .get(cu_id, []) if e.get("etype") in etypes]
        return [x for x in (out + inn) if x in self.cus]
    def reference_closure(self, start_id: str, etypes=("REFERS_TO","DERIVES"), max_nodes: int=5000) -> List[str]:
        """Unlimited-hop closure (until fixpoint). Cycles handled with visited set."""
        visited: Set[str] = {start_id}
        queue: collections.deque[str] = collections.deque([start_id])
        order: List[str] = []
        while queue and len(visited) < max_nodes:
            nid = queue.popleft()
            for nb in self.neighbors_bidir(nid, etypes=etypes):
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
                    order.append(nb)
        return order

class SemanticEmbedder:
    def __init__(self, backend="openai",
                 openai_model="text-embedding-3-large",
                 local_model="sentence-transformers/all-mpnet-base-v2",
                 batch_size=64):
        self.backend = backend; self.openai_model = openai_model
        self.local_model = local_model; self.batch_size = batch_size
        self._local = None
    def encode(self, texts: list[str]) -> list[np.ndarray]:
        try:
            if self.backend == "openai":
                from openai import OpenAI
                out = OpenAI().embeddings.create(model=self.openai_model, input=texts)
                return [np.array(d.embedding, dtype=np.float32) for d in out.data]
            if self.backend == "local":
                from sentence_transformers import SentenceTransformer
                if self._local is None: self._local = SentenceTransformer(self.local_model)
                vecs = self._local.encode(texts, batch_size=self.batch_size, normalize_embeddings=False)
                return [np.array(v, dtype=np.float32) for v in vecs]
        except Exception:
            pass
        # 안전 폴백
        def bow(t): 
            h = hashlib.md5(t.encode("utf-8")).digest()*8
            v = np.frombuffer(h[:128], dtype=np.uint8).astype(np.float32)
            return (v - v.mean())/(v.std()+1e-6)
        return [bow(x) for x in texts]
    @staticmethod
    def cos(a,b):
        na,nb = np.linalg.norm(a), np.linalg.norm(b)
        if na==0 or nb==0: return 0.0
        return float(np.dot(a,b)/(na*nb))

# ------------------------------
# Similarity (pluggable)
# ------------------------------
class SimilarityModel:
    def sim(self,a:str,b:str)->float: raise NotImplementedError
    def pred_cluster(self,p:str)->str:
        p=_normalize_text(p)
        if any(x in p for x in ["transfer","share","send","export"]): return "transfer"
        if any(x in p for x in ["process","use","collect","store","handle","retain"]): return "process"
        if any(x in p for x in ["purpose","profil","target"]): return "purpose"
        if any(x in p for x in ["relies_on","basis","lawful","consent","contract","legitimate","legal"]): return "legal_basis"
        if any(x in p for x in ["contains","include","has"]): return "contains"
        if any(x in p for x in ["located_in","store_in","hosted_in"]): return "location"
        return p

class SimpleSimilarity(SimilarityModel):
    def sim(self,a:str,b:str)->float:
        ta,tb=set(_tokenize(a)),set(_tokenize(b))
        if not ta or not tb: return 0.0
        return len(ta & tb)/float(len(ta|tb))
    def sim_pred(self,pred:str, action_hint:Optional[str])->float:
        if not action_hint: return 0.0
        return 1.0 if self.pred_cluster(pred)==self.pred_cluster(action_hint) else self.sim(pred, action_hint)

@dataclass
class CandidateRetriever:
    sim: SimilarityModel
    top_k:int=8; w_h:float=0.2; w_sem:float=0.5; w_frame:float=0.2; w_intent:float=0.1
    # ↓ 새로 주입: 임베더/클러스터러(없으면 생성자에서 set 후 사용)
    embedder: Optional[Any] = None
    clusterer: Optional[Any] = None
    use_subject_only: bool = False
    tau_subj: float = 0.4

    @staticmethod
    def _short(x, n=120):
        s = x if isinstance(x, str) else (json.dumps(x, ensure_ascii=False) if x is not None else "")
        return s[:n] + ("…" if len(s) > n else "")

    def _hypernym_score(self, cu_subject:str, ent:Optional[Entity])->float:
        if not ent: return 0.0
        subj=_normalize_text(cu_subject); labels=ent.hypernym_labels()+[ent.name.lower()]
        return max((self.sim.sim(subj,l) for l in labels), default=0.0)

    def _derive_action(self, attrs:Dict[str,Any])->Optional[str]:
        subj=attrs.get("subject") or ""; cond=attrs.get("condition")
        a,_=_split_subject_action(subj)
        if not a:
            if isinstance(cond,str): a,_=_split_subject_action(cond)
            elif isinstance(cond,dict):
                for k in ("all","any"):
                    for it in cond.get(k,[]):
                        a,_=_split_subject_action(it)
                        if a: return a
        return a

    def _cond_to_text(self, x) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, list):
            return " ; ".join(self._cond_to_text(e) for e in x)
        if isinstance(x, dict):
            # DSL 스키마 힌트 처리
            if "exists" in x and isinstance(x["exists"], dict):
                term = x["exists"].get("term", "")
                return f"exists({term})"
            if "rel" in x and isinstance(x["rel"], dict):
                fr  = x["rel"].get("frame","")
                obj = x["rel"].get("obj_term","")
                return f"rel({fr}->{obj})"
            parts = []
            if "all" in x: parts.append("ALL(" + self._cond_to_text(x["all"]) + ")")
            if "any" in x: parts.append("ANY(" + self._cond_to_text(x["any"]) + ")")
            if "not" in x: parts.append("NOT(" + self._cond_to_text(x["not"]) + ")")
            if parts:
                return " & ".join(p for p in parts if p)
            # 기타 dict는 JSON으로 안전 변환
            try:
                return json.dumps(x, ensure_ascii=False, sort_keys=True)
            except Exception:
                return str(x)
        # 그 외 타입은 문자열화
        return str(x)

    # [REPLACE] cue 텍스트 생성: dict/list 안전 처리
    def _cue_text(self, attrs: Dict[str, Any]) -> str:
        subj = attrs.get("subject") or ""
        cstr = attrs.get("constraint")
        cstr_txt = " ; ".join(str(c) for c in cstr) if isinstance(cstr, list) else str(cstr or "")
        cond_txt = self._cond_to_text(attrs.get("condition"))
        ctx_txt  = self._cond_to_text(attrs.get("context"))
        # 임베딩 입력 길이 과도 방지(옵션)
        if len(cond_txt) > 300: cond_txt = cond_txt[:300]
        if len(ctx_txt)  > 300: ctx_txt  = ctx_txt[:300]
        return f"{subj} || {cstr_txt} || {cond_txt} || {ctx_txt}"

    # [KEEP/UPDATED] intent 보너스(프레임 조화 가중)
    def _intent_bonus(self, attrs: Dict[str,Any], anc_frame: str) -> float:
        lab = (attrs.get("intent") or "").upper().replace("-","_")
        af = (anc_frame or "").split(":",1)[-1].lower()
        if "PERMIT_COND" in lab and ("transfer" in af or "share" in af):
            return 1.0
        if "PROHIBIT" in lab and ("process" in af or "purpose" in af):
            return 0.8
        if "OBLIGE" in lab and ("process" in af or "legal" in af):
            return 0.6
        if "PERMIT" in lab:
            return 0.4
        return 0.0
    
    # CandidateRetriever 안/밖 유틸
    @staticmethod
    def _norm_tok(x) -> str:
        return str(x or "").lower().replace("_", " ").strip()

    def _entity_terms(self, ctx, anc):
        def _hset(e):
            if not e: return set()
            out = set()
            if getattr(e, "type", None): out.add(self._norm_tok(e.type))
            if hasattr(e, "hypernym_labels"):
                for h in (e.hypernym_labels() or []): out.add(self._norm_tok(h))
            # 이름/별칭도 살짝
            if getattr(e, "name", None): out.add(self._norm_tok(e.name))
            return out
        actor = ctx.get_entity(anc.actor) if anc.actor else None
        obj   = ctx.get_entity(anc.obj)   if anc.obj   else None
        return _hset(actor), _hset(obj)
    
    def __call__(self, anchors, ctx, policy):
        if getattr(self, "use_subject_only", True) and getattr(self, "cu_subj_vec", None):
            w_ent, w_hyp = 0.65, 0.35     # ← 권장 가중치
            tau = getattr(self, "tau_subj", 0.40)

            out = {}
            for anc in anchors:
                # 1) 앵커 텍스트 구성
                H_act, H_obj = self._entity_terms(ctx, anc)  # 하이퍼님/타입/이름 집합
                ent_terms = " ".join(sorted((H_act | H_obj) - set())) or "entity"
                hyp_terms = " ".join(sorted(H_act | H_obj)) or ent_terms  # 하이퍼님 묶음

                # 2) 앵커당 1회 임베딩
                v_ent, v_hyp = self.embedder.encode([ent_terms, hyp_terms])
                v_ent = v_ent/(np.linalg.norm(v_ent)+1e-9)
                v_hyp = v_hyp/(np.linalg.norm(v_hyp)+1e-9)

                scores = []
                H_anchor = H_act | H_obj  # 문자열 교집합 보너스용

                for cu_id, node in policy.iter_cus("actor_cu"):
                    attrs = (node.get("attrs") or {})
                    if (attrs.get("role","").lower() == "exception"):
                        continue
                    v_subj = self.cu_subj_vec.get(cu_id)
                    if v_subj is None: 
                        continue

                    # (A) subject ⟷ entity 코사인
                    s_ent = float(np.dot(v_ent, v_subj))
                    if s_ent < tau: 
                        continue  # 약한 컷으로 리콜 유지

                    # (B) subject ⟷ hypernym 코사인
                    s_hyp = float(np.dot(v_hyp, v_subj))

                    # (C) 하이퍼님 라벨 교집합 보너스(문자열 매칭, 감점 없음)
                    cu_hset = set()
                    for k in ("subject_hypernym","subject_hypernyms","subject","data_category","keywords"):
                        v = attrs.get(k)
                        if isinstance(v, str) and v: cu_hset.add(self._norm_tok(v))
                        elif isinstance(v, (list, tuple)):
                            for t in v[:8]:
                                if t: cu_hset.add(self._norm_tok(t))
                    bonus = 1.0 if (H_anchor & cu_hset) else 0.0  # 이진 보너스

                    total = w_ent*s_ent + w_hyp*s_hyp + 0.05*bonus
                    scores.append((cu_id, total))

                scores.sort(key=lambda x: x[1], reverse=True)
                # top-M (preselect_k) + 마진 컷
                keep = scores[: self.top_k]  # ← getattr(self,"preselect_k",120) 대신
                if keep:
                    smax = keep[0][1]
                    keep = [c for c in keep if c[1] >= 0.80*smax]
                out[anc.id] = keep

                # debug
                head = keep[:5]
                print(f"[preselect][anc={anc.id}] ent='{self._short(ent_terms,80)}'  K={len(keep)}  head={[ (cid, round(sc,3)) for cid, sc in head ]}")
                # for cid, sc in head:
                #     a = (policy.get_cu(cid).get('attrs') or {})
                #     subj = a.get("subject") or a.get("data_category") or a.get("title")
                #     print(f"  - cu={cid} subj='{self._short(subj,80)}'  total={sc:.3f}")
            return out


_INTENT_CANON = {
    "PROHIBIT": "prohibit",
    "FORBID": "prohibit",
    "OBLIGE": "oblige",
    "OBLIGATION": "oblige",
    "MUST": "oblige",
    "PERMIT": "permit",
    "ALLOW": "permit",
    "ALLOWED": "permit",
    "PERMITTED": "permit",
    "PERMIT_COND": "permit_cond",
    "CONDITIONAL": "permit_cond",
    "ONLY_IF": "permit_cond",
}

# ------------------------------
# CU → Plan
# ------------------------------
@dataclass
class CUCompiler:
    sim: SimilarityModel
    def _intent(self, attrs: dict,
                *,
                trust_conf: float = 0.0,   # 스냅샷 intent_conf가 이 값 이상이면 그대로 채택
                use_fallback: bool = True  # 스냅샷이 없거나 신뢰도 낮을 때만 최소 폴백 사용
    ) -> str:
        """
        Snapshot 우선: attrs.intent / attrs.intent_conf 사용.
        - PROHIBIT / OBLIGE / PERMIT / PERMIT_COND만 정상화해서 반환 (소문자)
        - 신뢰도(intent_conf) < trust_conf면 폴백(선택)
        - 둘 다 없으면 'unspec'
        """
        # 1) snapshot 우선
        snap = (attrs or {}).get("intent")
        conf = float((attrs or {}).get("intent_conf") or 0.0)
        if snap:
            key = str(snap).strip().upper().replace("-", "_").replace(" ", "_")
            if key in _INTENT_CANON and conf >= trust_conf:
                return _INTENT_CANON[key]

        if not use_fallback:
            return "unspec"

        # 2) 최소 폴백 (보편 deontic cue만)
        c = attrs.get("constraint")
        text = " ".join(c) if isinstance(c, list) else (c or "")
        s = _norm(text)

        if "shall not" in s or "prohibited" in s or "forbidden" in s:
            return "prohibit"
        if "only if" in s or "provided that" in s or "subject to" in s or "on condition that" in s:
            return "permit_cond"
        if re.search(r"\b(shall|must)\b", s):
            return "oblige"
        if re.search(r"\b(may|allowed|permitted)\b", s):
            return "permit"

        return "unspec"
    
    def _patient_hints(self, subject:str, context:Optional[str], condition:Optional[Any])->List[str]:
        hints=[]; s=_normalize_text(subject); c=_normalize_text(context or "")
        if isinstance(condition,str): cond=_normalize_text(condition)
        elif isinstance(condition,dict): cond=" ".join(_normalize_text(x) for k in ("all","any") for x in condition.get(k,[]))
        else: cond=""
        for src in (s,c,cond):
            for m in re.findall(r"(data concerning [a-z ]+|personal data|third country|international organisation|genetic data|biometric data|health|health data)", src):
                hints.append(m.strip())
        return list(dict.fromkeys(hints))

    def __call__(self, cu_id:str, node:Dict[str,Any])->Plan:
        attrs=node.get("attrs",{}); subj=attrs.get("subject") or ""
        cond=attrs.get("condition"); cons=attrs.get("constraint") or []; ctx=attrs.get("context")
        act,_=_split_subject_action(subj)
        if not act:
            if isinstance(cond,str): act,_=_split_subject_action(cond)
            elif isinstance(cond,dict):
                for k in ("all","any"):
                    for it in cond.get(k,[]):
                        a,_=_split_subject_action(it)
                        if a: act=a; break
        snap_intent = self._intent(attrs, trust_conf=0.0, use_fallback=False)  # 스냅샷 우선
        plan = Plan(
            cu_id=cu_id,
            cu_type=node.get("type", "actor_cu"),
            subject_text=subj,
            constraint_texts=cons if isinstance(cons, list) else [str(cons)],
            condition_text=cond,
            context_text=ctx,
            intent=snap_intent,           # ← 여기!
            action_hint=act,
            patient_hints=self._patient_hints(subj, ctx, cond)  # 필요시 유지
        )

        return plan

# ------------------------------
# LLM Backend (OpenAI gpt-4o)
# ------------------------------
class LLMBackend:
    # """교체 가능한 인터페이스."""    
    # === [NEW] 앵커 단위 listwise 판단 (1콜) ===
    def judge_anchor_listwise(self, anchor, ctx, items):
        if not items: return []
        N = len(items)

        window = self._build_anchor_window(ctx, anchor, hops=2, limit=240)

        sys_msg = (
            "You are a regulatory compliance judge.\n"
            # 1) 판단 원칙
            "- Judge ONLY from the evidence provided ANCHOR and CONTEXT WINDOW. No external knowledge.\n"
            "- Violation standard: prefer explicit contradictions, but also accept strongly implied contradictions\n"
            "- When there is strong suspicion or implication of contradiction based on specific details in the window, classify as NON-COMPLIANT, even if the evidence is not fully explicit.\n"
            "  when the window contains concrete, specific facts that logically entail a violation (close paraphrases are OK).\n"
            "- Do not infer from pure silence; if key facts are missing or ambiguous, return INSUFFICIENT (or NOT_APPLICABLE if out of scope).\n"
            # 2) 출력 스키마 (루트=object + results 배열)
            f"Return a STRICT JSON OBJECT with exactly one key 'results' as an array of length {N} in the SAME ORDER.\n"
            "Each results[i] = {\n"
            "  cu_id, verdict ∈ {COMPLIANT, NON_COMPLIANT, NOT_APPLICABLE, INSUFFICIENT},\n"
            "  confidence ∈ [0,1],\n"
            "  why: ≤25 tokens,\n"
            "  evidence: [short quotes OR close paraphrases with span hints if a quote is unavailable],\n"
            "  missing: [which facts are missing],\n"
            "  proposed_queries: [short retrieval hints to obtain missing facts]\n"
            "}\n"
            # """
            # === VALID TEMPLATE (COPY VERBATIM AND FILL VALUES) ===
            # {"results":[
            # {"cu_id":"<ID_1>","verdict":"<LABEL_1>","confidence":<0..1>}%IF_N>1%, 
            # {"cu_id":"<ID_2>","verdict":"<LABEL_2>","confidence":<0..1>}%/IF%
            # ]}

            # INVALID (missing array wrapper):
            # {"cu_id":"...","verdict":"NON_COMPLIANT","confidence":0.8}
            # """
            "No code fences. No extra keys. No explanations outside JSON."
        )

        payload = {
            "task": "anchor_listwise_check",
            "N": N,
            "anchor": _render_anchor_bundle(anchor, ctx),          # 네가 주던 앵커 evidence
            "window": window,
            "candidates": items,                                   # 각 CU의 plan
            "candidate_cu_ids": [it["cu_id"] for it in items],
        }

        out = self._ask(sys_msg, payload)
        if RAW_DUMP: _raw_dump("listwise", anchor.id, sys_msg, payload, out)
        return self._coerce_results(out, N, items)

    # === [NEW] 참조조항 일괄 재검사 (위반된 CU만, 1콜) ===
    def judge_references_listwise(self, anchor, ctx, sets):
        if not sets: return []
        N = len(sets)

        sys_msg = (
            "You are a GDPR compliance judge for exceptions/derogations.\n"
            "- Decide ONLY from provided base CU and referenced CUs.\n"
            f"Return a STRICT JSON OBJECT with key 'results' as an array of length {N}.\n"
            "Each results[i] = { base_cu_id, overridden: bool,\n"
            "  final_verdict ∈ {COMPLIANT, NON_COMPLIANT, NOT_APPLICABLE, INSUFFICIENT},\n"
            "  confidence ∈ [0,1], winner: <ref_id|null>, why: ≤25 tokens, evidence: [quotes] }\n"
            "No code fences. No extra keys."
        )

        payload = {
            "task": "anchor_reference_override",
            "N": N,
            "anchor": _render_anchor_bundle(anchor, ctx),
            "bases": sets,
            "base_ids": [s["base_cu_id"] for s in sets],
        }

        out = self._ask(sys_msg, payload)
        if RAW_DUMP: _raw_dump("refs", anchor.id, sys_msg, payload, out)

        if isinstance(out, dict): out = out.get("results", out)
        if not isinstance(out, list):
            return [{"base_cu_id": s["base_cu_id"], "overridden": False,
                    "final_verdict":"NON_COMPLIANT", "score":0.5, "winner":None,
                    "why":"fallback", "evidence": []} for s in sets]

        res=[]
        allowed = {"COMPLIANT","NON_COMPLIANT","NOT_APPLICABLE","INSUFFICIENT"}
        for i, s in enumerate(sets):
            oi = out[i] if i < len(out) and isinstance(out[i], dict) else {}
            try: conf = float(oi.get("confidence", 0.5))
            except: conf = 0.5
            fv = str(oi.get("final_verdict","NON_COMPLIANT"))
            fv = fv if fv in allowed else "NON_COMPLIANT"
            ev = oi.get("evidence", []); ev = ev if isinstance(ev, list) else []
            res.append({
                "base_cu_id": s["base_cu_id"], "overridden": bool(oi.get("overridden", False)),
                "final_verdict": fv, "score": conf, "confidence": conf, "winner": oi.get("winner", None),
                "why": oi.get("why",""), "evidence": ev
            })
        return res
    
    def _build_anchor_window(self, ctx, anchor, hops: int = 1, limit: int = 120) -> dict:
        """
        출력 형태:
        {
          "triples": [
             {"id":"t1", "subj":{"id","name","type"}, "pred":"...", "obj":{"id","name","type"},
              "quote": "<evidence or ''>", "source":"context"}
             ...
          ]
        }
        - anchor triple이 항상 맨 앞(가능하면 증거 quote 포함)
        - hops=1: actor/obj와 같은 노드를 공유하는 triple을 포함
        - limit: triple 최대 개수(과도 토큰 방지)
        """
        def ent_blob(eid):
            e = ctx.get_entity(eid) if eid else None
            return {"id": getattr(e, "id", eid), "name": getattr(e, "name", None), "type": getattr(e, "type", None)}

        # 0) 모든 relation 인덱스화(간단한 양방향 adjacency)
        rels = list(ctx.iter_relations())
        by_ent = {}
        for r in rels:
            by_ent.setdefault(r.subj, []).append(r)
            by_ent.setdefault(r.obj, []).append(r)

        triples = []
        seen = set()

        # 1) 앵커 triple 우선
        def add_rel(r, tag="context"):
            rid = f"{r.subj}|{r.pred}|{r.obj}"
            if rid in seen:
                return
            seen.add(rid)
            triples.append({
                "id": rid,
                "subj": ent_blob(r.subj),
                "pred": r.pred,
                "obj": ent_blob(r.obj),
                "quote": ((r.evidence or {}).get("quote", "") or "")[:240],
                "source": tag
            })

        # anchor와 정확히 일치하는 relation이 있으면 그걸, 없으면 synthetic으로 구성
        anchor_matched = False
        for r in by_ent.get(anchor.actor, []):
            if r.subj == anchor.actor and r.obj == anchor.obj and r.pred == anchor.pred:
                add_rel(r); anchor_matched = True; break
        if not anchor_matched:
            # synthetic(증거는 anchor.evidence 사용)
            class _R: pass
            r = _R(); r.subj = anchor.actor; r.obj = anchor.obj; r.pred = anchor.pred
            r.evidence = anchor.evidence or {}
            add_rel(r)

        # 2) BFS로 hops 이웃 triple 채우기
        from collections import deque
        q = deque()
        seeds = [anchor.actor, anchor.obj]
        for s in seeds:
            if s is not None:
                q.append((s, 0))
        visited_ents = set(seeds)

        while q and len(triples) < limit:
            eid, d = q.popleft()
            if d >= hops:
                continue
            for r in by_ent.get(eid, []):
                add_rel(r)
                # 이웃 엔티티 enqueue
                for nb in (r.subj, r.obj):
                    if nb not in visited_ents:
                        visited_ents.add(nb)
                        q.append((nb, d + 1))
                if len(triples) >= limit:
                    break

        return {"triples": triples[:limit]}

    def _coerce_results(self, out, N: int, items, *, key_candidates=("results","items","answers","data")):
        """
        다양한 응답 형태(out)를 list[dict] 길이 N으로 강제 변환.
        - dict에 results/items/answers/data 중 하나가 있으면 그 리스트를 사용
        - dict가 {cu_id: {...}} 형태면 items 순서에 맞춰 매핑
        - list가 N보다 짧으면 INSUFFICIENT로 패딩, 길면 N에 맞춰 절단
        - dict/str/None 등 리스트가 아니면 전부 INSUFFICIENT[N]으로 대체
        """
        def _blank(it):
            return {"cu_id": it["cu_id"], "verdict":"INSUFFICIENT", "confidence":0.5, "score":0.5,
                    "why":"fallback: non_list", "evidence":[], "missing":["response not list"], "proposed_queries":[]}

        # 1) 이미 리스트면 그대로
        if isinstance(out, list):
            arr = out
        # 2) dict면 results 같은 후보 키를 찾아본다
        elif isinstance(out, dict):
            arr = None
            for k in key_candidates:
                v = out.get(k)
                if isinstance(v, list):
                    arr = v; break
            # {cu_id: {...}} 매핑 형태면 items 순서대로 변환
            if arr is None and all(isinstance(v, dict) for v in out.values()):
                by_id = {str(k): v for k, v in out.items()}
                arr = [ by_id.get(str(it["cu_id"]), {}) for it in items ]
            if arr is None:
                arr = []
        else:
            arr = []

        # 길이 보정
        if len(arr) < N:
            arr = list(arr) + [{} for _ in range(N - len(arr))]
        elif len(arr) > N:
            arr = arr[:N]

        # 각 항목 표준화(필수 키/타입 보장)
        allowed = {"COMPLIANT","NON_COMPLIANT","NOT_APPLICABLE","INSUFFICIENT"}
        res = []
        for i, it in enumerate(items):
            oi = arr[i] if isinstance(arr[i], dict) else {}
            v = str(oi.get("verdict","INSUFFICIENT")).upper().replace(" ","_")
            v = v if v in allowed else "INSUFFICIENT"
            try:
                conf = float(oi.get("confidence", oi.get("score", 0.5)))
            except Exception:
                conf = 0.5
            ev   = oi.get("evidence", []);  ev   = ev if isinstance(ev, list) else []
            miss = oi.get("missing",  []);  miss = miss if isinstance(miss, list) else []
            q    = oi.get("proposed_queries", []); q = q if isinstance(q, list) else []
            res.append({
                "cu_id": it["cu_id"],
                "verdict": v,
                "confidence": conf, "score": conf,
                "why": str(oi.get("why",""))[:200],
                "evidence": [str(x) for x in ev if x],
                "missing": [str(x) for x in miss if x],
                "proposed_queries": [str(x) for x in q if x],
            })
        return res


def _render_anchor_bundle(anchor: Anchor, ctx: ContextGraph) -> Dict[str, Any]:
    act = ctx.get_entity(anchor.actor) if anchor.actor else None
    obj = ctx.get_entity(anchor.obj) if anchor.obj else None
    return {
        "predicate": anchor.pred,
        "actor": {
            "id": act.id if act else None,
            "name": act.name if act else None,
            "type": act.type if act else None,
            "hypernyms": act.hypernyms() if act else [],
        },
        "object": {
            "id": obj.id if obj else None,
            "name": obj.name if obj else None,
            "type": obj.type if obj else None,
            "hypernyms": obj.hypernyms() if obj else [],
        },
        "evidence_quote": anchor.evidence.get("quote") if anchor.evidence else None,
    }

def _plan_to_dict(plan: Plan) -> Dict[str, Any]:
    return {
        "cu_id": plan.cu_id,
        "type": plan.cu_type,
        "subject": plan.subject_text,
        "constraint": plan.constraint_texts,
        "condition": plan.condition_text,
        "context": plan.context_text,
        "intent": plan.intent,
        "action_hint": plan.action_hint,
        "patient_hints": plan.patient_hints,
    }

def _plan_to_dict_from_attrs(cu_id: str, node: dict) -> dict:
    """compiler 실패 시 node.attrs만으로 최소 Plan dict 생성 (LLM listwise용)."""
    attrs = (node.get("attrs") or {})
    MAX = 280  # 토큰 폭주 방지용 길이 제한

    def _as_text(x):
        if isinstance(x, list): s = "; ".join(str(i) for i in x)
        else: s = str(x or "")
        return s[:MAX]

    def _as_list(x):
        if x is None: return []
        if isinstance(x, list): return [str(i)[:MAX] for i in x]
        return [str(x)[:MAX]]

    subject = attrs.get("subject") or attrs.get("data_category") or attrs.get("title") or ""
    # keywords / subject_hypernyms 등을 환자 힌트로 보조
    hints = []
    if isinstance(attrs.get("subject_hypernyms"), list):
        hints.extend([str(h) for h in attrs["subject_hypernyms"][:8]])
    if isinstance(attrs.get("keywords"), list):
        hints.extend([str(k) for k in attrs["keywords"][:8]])

    return {
        "cu_id": cu_id,
        "type": attrs.get("cu_type") or node.get("type") or "actor_cu",
        "subject": _as_text(subject),
        "constraint": _as_list(attrs.get("constraint")),
        "condition": _as_text(attrs.get("condition")),
        "context": _as_text(attrs.get("context")),
        "intent": _as_text(attrs.get("intent")),
        "action_hint": _as_text(attrs.get("action") or attrs.get("predicate")),
        "patient_hints": [h[:MAX] for h in hints],
    }

def _force_json(s: str) -> Dict[str, Any]:
    """관대한 JSON 파서: 첫 번째 { ... } 블록을 파싱 시도."""
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}

# --- REPLACE the existing OpenAILLM class with this one ---

class OpenAILLM(LLMBackend):
    """
    OpenAI LLM backend with verbose debug printing.
    - Supports Responses API first; falls back to Chat Completions; then legacy openai.ChatCompletion.
    - When debug=True, prints every system prompt, user JSON payload, raw model output, parsed JSON, usage, and latency.
    """
    def __init__(self, model: str = "gpt-4.1", temperature: float = 0.0, max_output_tokens: int = 16384,
                 debug: bool = False, log_file: Optional[str] = None, truncate: int = 1500, redact: bool = False,
                 monitor: Optional[ComplianceMonitor] = None):
        try:
            from openai import OpenAI  # pip install openai
        except Exception as e:
            raise RuntimeError("openai 패키지가 필요합니다. pip install openai") from e
        self._OpenAI = OpenAI
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.debug = debug or bool(int(os.getenv("CG_DEBUG", "0")))
        self.truncate = truncate
        self.redact = redact
        self._cache = {}
        self._log_fp = open(log_file, "a", encoding="utf-8") if (log_file and len(log_file) > 0) else None
        self.triage_model = 'gpt-4o-mini'
        self.monitor = monitor

    # -------------- internal helpers --------------
    def _clip(self, text: str) -> str:
        if text is None:
            return ""
        if self.truncate and len(text) > self.truncate:
            return text[: self.truncate] + f"\n... [truncated {len(text)-self.truncate} chars]"
        return text

    def _pr(self, *parts):
        msg = " ".join(str(p) for p in parts)
        if self._log_fp:
            print(msg, file=self._log_fp, flush=True)
        print(msg, flush=True)

    def _extract_usage(self, resp) -> Dict[str, Any]:
        """Try to read token usage from different SDK shapes."""
        usage = {}
        for path in ("usage", "response.usage"):
            try:
                node = resp
                for part in path.split("."):
                    node = getattr(node, part)
                if node:
                    usage = {
                        "prompt_tokens": getattr(node, "prompt_tokens", None) or getattr(node, "input_tokens", None) or getattr(node, "prompt", None),
                        "completion_tokens": getattr(node, "completion_tokens", None) or getattr(node, "output_tokens", None) or getattr(node, "completion", None),
                        "total_tokens": getattr(node, "total_tokens", None),
                    }
                    break
            except Exception:
                continue
        return usage

    # -------------- unified ask --------------
    def _ask(self, system_instruction: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        import json, time
        payload_str = json.dumps(user_payload, ensure_ascii=False)

        raw_text = ""
        resp = self.client.responses.create(
            model=self.model,
            input=[{"role":"system","content":system_instruction},
                    {"role":"user","content":payload_str}],
            text={"format": {"type": "json_object"}},
            max_output_tokens=self.max_output_tokens,
        )
        raw_text = resp.output_text
        print(f"[diag] json_mode=on model={self.model} len={len(raw_text)}")
        parsed = _force_json(raw_text)

        # --- 모니터 기록
        # if self.monitor:
        #     self.monitor.log_llm(
        #         task=user_payload.get("task",""),
        #         system=system_instruction,
        #         payload=user_payload,
        #         raw=raw_text,
        #         parsed=parsed,
        #         latency=elapsed,
        #         path_used=path_used,
        #         usage=usage
        #     )
        return parsed
    
    
# --- ADD: LocalFileLLM -------------------------------------------------------
class LocalFileLLM(LLMBackend):
    """
    Local LLM loader by file path (minimal drop-in replacement).
    - If model_path endswith '.gguf'  -> llama-cpp-python (llama.cpp)
    - Else                           -> HuggingFace transformers pipeline (with 4bit by default)
    Prompts / output JSON schema: identical to OpenAILLM (LLMBackend 상위가 모두 관리)
    """
    def __init__(self, model_path: str, temperature: float = 0.0,
                 max_output_tokens: int = 1024, debug: bool = False):
        self.model_path = str(model_path)
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_output_tokens)
        self.debug = bool(debug)
        self.backend = None              # ★ 항상 초기화

        if self.model_path.lower().endswith(".gguf"):
            # llama.cpp
            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise RuntimeError("`pip install llama-cpp-python` 필요 (.gguf 모델)") from e
            self.backend = "llama_cpp"
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=8192,
                n_gpu_layers=-1,          # GPU 빌드면 -1, CPU면 0
                verbose=self.debug
            )
        else:
            # transformers (4bit)
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
                import torch
            except ImportError as e:
                raise RuntimeError("`pip install transformers accelerate torch bitsandbytes` 필요") from e

            self.backend = "transformers"  # ★ 반드시 설정
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            tok = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
            mdl = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto"
            )

            self.pipe = pipeline(
                "text-generation",
                model=mdl, tokenizer=tok,
                do_sample=(self.temperature > 0.0),
                temperature=0.0,
                max_new_tokens=self.max_new_tokens,
                repetition_penalty=1.1,
            )

    def _ask(self, system_instruction: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        import json, time
        prompt = (
            f"{system_instruction}\n"
            "Return STRICT JSON only. No extra text.\n"
            "USER:\n"
            f"{json.dumps(user_payload, ensure_ascii=False)}\n"
            "JSON:"
        )
        t0 = time.time()

        if self.backend == "llama_cpp":
            out = self.llm.create_completion(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                stop=["\nUSER:", "\nJSON:\n\n"]
            )
            text = out["choices"][0]["text"]
        elif self.backend == "transformers":
            text = self.pipe(prompt, num_return_sequences=1)[0]["generated_text"]
            # 프롬프트가 에코되면 "JSON:" 이후만 취함
            if "JSON:" in text:
                text = text.split("JSON:", 1)[-1]
        else:
            raise RuntimeError("LocalFileLLM.backend not initialized")

        if self.debug:
            print(f"[LocalFileLLM:{self.backend}] out_len={len(text)}  elapsed={time.time()-t0:.2f}s")

        # 파싱(개선된 force_json이 있으면 그걸 써도 OK)
        return _force_json(text)
# ---------------------------------------------------------------------------

class OpenAICompatLLM(LLMBackend):
    """
    Any OpenAI-compatible endpoint:
      e.g., http://localhost:8000/v1 (vLLM),
            http://localhost:1234/v1 (LM Studio),
            http://localhost:8080/v1 (llama.cpp server, 호환 모드)
    """
    def __init__(self,
                 model: str,
                 base_url: str = "http://localhost:8000/v1",
                 api_key: str = "sk-local",
                 temperature: float = 0.0,
                 max_tokens: int = 2048,
                 top_p: float = 1.0,
                 debug: bool = False):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.top_p = float(top_p)
        self.debug = bool(debug)

    def _ask(self, system_instruction: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        import json, time
        payload = json.dumps(user_payload, ensure_ascii=False)
        t0 = time.time()
        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user",   "content": payload},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                # 가능한 서버에서 JSON 강제(지원 안 하면 무시됨)
                response_format={"type": "json_object"}
            )
            txt = (r.choices[0].message.content or "").strip()
            if self.debug:
                print(f"[OpenAICompatLLM] len={len(txt)}  elapsed={time.time()-t0:.2f}s")
            return _force_json(txt)
        except Exception as e:
            if self.debug:
                import traceback; traceback.print_exc()
                print(f"[OpenAICompatLLM][ERROR] {e}")
            # 필요시 여기서 raise로 올려도 됨
            return {}



# ------------------------------
# Compliance Gate (GoT)
# ------------------------------
@dataclass
class ComplianceGate:
    sim: Optional[SimilarityModel]=None
    pred_threshold: float=0.65
    # [PATCH] 최종 K는 기존 top_k를 그대로 사용, 프리셀렉트 K는 별도 추가
    top_k: int=8                       # 최종 K (기존 CLI와 호환)
    preselect_k: int=50               # 1차 선별 K (추천값)
    reranker_model: str="BAAI/bge-reranker-v2-m3"
    llm_backend: Optional[LLMBackend]=None
    monitor: Optional[ComplianceMonitor]=None
    judging_mode: str = "checklist"
    enable_meta_gate: bool = False   # ← True/False로 토글
    use_subject_only: bool = True

    def __post_init__(self):
        self.sim = self.sim or SimpleSimilarity()
        self.embedder = getattr(self, "embedder", SemanticEmbedder(backend="openai"))
        self.clusterer = getattr(self, "clusterer", PredicateClusterer(self.embedder, k=8))
        self.compiler = getattr(self, "compiler", CUCompiler(self.sim))
        # [PATCH] CandidateRetriever는 프리셀렉트 K로 넓게 뽑게 함
        self.cand_retriever = getattr(self, "cand_retriever",
            CandidateRetriever(self.sim, top_k=self.preselect_k,
                               embedder=self.embedder, clusterer=self.clusterer))
        self.llm_backend = self.llm_backend or OpenAILLM(model="gpt-4.1", temperature=0.0,
                                                         monitor=getattr(self,"monitor",None))
        # [NEW] 리랭커 주입
        self.reranker = LegalCrossReranker(self.reranker_model)
        self.cand_retriever.use_subject_only = self.use_subject_only
        self.disk_cache = getattr(self, "disk_cache", DiskVecCache())
        # 임베딩 모델 식별자 추출
        self.embedder_model_id = (
            getattr(self.embedder, "model", None)
            or getattr(self.embedder, "model_name", None)
            or getattr(self.embedder, "backend", None)
            or "unknown"
        )
        self.report = getattr(self, "report", _RunReport(
            max_show=8,
            to_file=True,
            file_path="monitor/compliance_run_report.json"
        ))


    def __call__(self, policy_graph: Dict[str, Any], context_graph: Dict[str, Any]) -> List[Decision]:
        _Diag.begin()
        import datetime
        # 0) 그래프 로딩
        pol = PolicyGraph(policy_graph)
        ctx = ContextGraph(context_graph)
        
        # 0) 그래프 로딩 직후
        total_actor_cu = sum(1 for _ in pol.iter_cus("actor_cu"))
        self.report.set_meta(
            policy_name = policy_graph.get("name") or "policy",
            total_actor_cu = total_actor_cu
        )

        try:
            if not self.use_subject_only:        # ← 추가
                self.clusterer.fit(ctx)
        except Exception:
            pass
        if not hasattr(self, "anchor_extractor") or self.anchor_extractor is None:
            self.anchor_extractor = AnchorExtractor(self.clusterer)
        anchors = self.anchor_extractor(ctx)
        _Diag.inc("anchors", len(anchors))
        print(f"[diag] anchors this run = {len(anchors)}")

        # 2) 메타 CU 1회 검증(있을 때만)
        if self.enable_meta_gate:
            meta_plans = []
            for cu_id, node in pol.iter_cus("meta_cu"):
                try:
                    plan = self.compiler(cu_id, node)
                except Exception:
                    plan = None
                if plan and getattr(plan, "cu_type", None) == "meta_cu":
                    meta_plans.append(plan)
            if meta_plans and hasattr(self.llm_backend, "judge_meta_gate"):
                now_iso = datetime.datetime.utcnow().isoformat() + "Z"
                meta_res = self.llm_backend.judge_meta_gate(meta_plans, ctx, now_iso=now_iso)
                if getattr(self, "monitor", None):
                    self.monitor.log_meta_gate(meta_res)

        # --- CU subject embedding: disk cache load or build ---
        policy_name = policy_graph.get("name") or "policy"
        loaded = self.disk_cache.load_cu_subj(policy_name)
        if loaded:
            cu_ids, vecs_mmap, meta = loaded
            # model/version guard to avoid stale cache
            if (meta or {}).get("model") != self.embedder_model_id:
                print(f"[cache] model changed ({(meta or {}).get('model')} → {self.embedder_model_id}) → rebuilding…")
                loaded = None
            elif hasattr(vecs_mmap, "shape") and meta and meta.get("dim") and vecs_mmap.shape[1] != int(meta["dim"]):
                print(f"[cache] dim mismatch ({vecs_mmap.shape[1]} ≠ {meta.get('dim')}); rebuilding…")
                loaded = None
        if loaded:
            # 정규화 벡터를 dict로 (mmap은 읽기 전용, 필요시 float32로 copy)
            print(f"[cache] HIT  file=cache/{policy_name}_cache.npy  count={len(cu_ids)}  dim={getattr(vecs_mmap,'shape',('?',))[1] if hasattr(vecs_mmap,'shape') else '?'}")
            norms = np.linalg.norm(vecs_mmap, axis=1, keepdims=True) + 1e-9
            V = (vecs_mmap / norms).astype(np.float32, copy=False)
            self._cu_subj_vec = {cid: V[i] for i, cid in enumerate(cu_ids)}
            print(f"[diag] cu_subj cache HIT: {len(cu_ids)} vecs loaded")
        else:
            print(f"[cache] MISS file=cache/{policy_name}_cache.npy → rebuilding…")
            cu_ids, subj_texts = [], []
            for cid, node in pol.iter_cus("actor_cu"):
                a = (node.get("attrs") or {})
                if (a.get("role","").lower() == "exception"): 
                    continue
                subj = a.get("subject") or a.get("data_category") or ""
                if not subj:
                    title = (a.get("title") or node.get("label") or "")
                    subj = title.split(":")[-1][:120]
                kw = a.get("keywords") or []
                if isinstance(kw, list): kw = " ".join(kw[:6])
                subj_texts.append(f"{subj} {kw}".strip())
                cu_ids.append(str(cid))

            # 일괄 임베딩 1~N콜 → float16로 저장
            vecs = self.embedder.encode(subj_texts)
            vecs = np.array(vecs, dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
            V = (vecs / norms).astype(np.float16)  # 저장은 16비트로 경량화
            meta = {"model": self.embedder_model_id, "dim": int(V.shape[1]), "count": int(V.shape[0])}
            self.disk_cache.save_cu_subj(policy_name, cu_ids, V, meta)
            # 런타임 dict (float32 권장)
            V32 = V.astype(np.float32)
            self._cu_subj_vec = {cid: V32[i] for i, cid in enumerate(cu_ids)}
            print(f"[diag] cu_subj cache BUILD: {len(cu_ids)} vecs saved")
            print(f"[cache] BUILD file=cache/{policy_name}_cache.npy  count={len(cu_ids)}  dim={V.shape[1]}")

        # 리트리버에 주입
        self.cand_retriever.cu_subj_vec = self._cu_subj_vec
        print(f"[diag] subject-only preselect: {len(self._cu_subj_vec)} CU vecs ready")

        # 3) 후보 랭킹
        cand_map = diag_time_call("preselect", self.cand_retriever, anchors, ctx, pol)
        _Diag.add("K_pre_sum", sum(len(v) for v in cand_map.values()))
        if getattr(self, "monitor", None):
            for anc in anchors:
                self.monitor.log_candidates(anc.id, cand_map.get(anc.id, []))
        
        # report
        def _cu_subject(pol, cu_id):
            n = pol.get_cu(cu_id) or {}
            a = (n.get("attrs") or {})
            return a.get("subject") or a.get("data_category") or a.get("title") or ""

        for anc in anchors:
            # 앵커 요약 저장
            act = ctx.get_entity(anc.actor).type if anc.actor else ""
            obj = ctx.get_entity(anc.obj).type if anc.obj else ""
            evid = (getattr(anc, "evidence", {}) or {}).get("quote", "")
            self.report.set_anchor(anc.id, {
                "id": anc.id, "predicate": anc.pred, "actor_type": act, "object_type": obj, "evidence": evid
            })
            pre = cand_map.get(anc.id, [])
            self.report.add_preselect(anc.id, [
                {"cu_id": cid, "score": float(sc), "subject": _cu_subject(pol, cid)} for cid, sc in pre
            ])
        
        reranked = {}
        for anc in anchors:
            pre = cand_map.get(anc.id, [])
            reranked[anc.id] = diag_time_call(
                "rerank_per_anchor",
                self.reranker.rerank_one_anchor, anc, ctx, pol, pre, self.cand_retriever, final_k=self.top_k
            )
            # [DEBUG] 리랭크 결과 상위 N 찍기
            head = reranked[anc.id][: min(5, len(reranked[anc.id]))]
            print(f"[rerank][anc={anc.id}] finalK={len(reranked[anc.id])} head={[ (cid, round(sc,3)) for cid, sc in head ]}")

        cand_map = reranked
        _Diag.add("K_final_sum", sum(len(v) for v in cand_map.values()))
        for anc in anchors:
            rr = cand_map.get(anc.id, [])
            self.report.add_rerank(anc.id, [
                {"cu_id": cid, "score": float(sc), "subject": _cu_subject(pol, cid)} for cid, sc in rr
            ])

        # === [REPLACE START] 앵커당 1콜 + (참조 1콜) 파이프라인 ===
        def _reference_closure(cu_id: str) -> list[str]:
            if hasattr(pol, "reference_closure"):
                return pol.reference_closure(cu_id)
            # fallback: 양방향 1-hop라도 수집
            out, visited, q = [], {cu_id}, [cu_id]
            if hasattr(pol, "neighbors_bidir"):
                while q:
                    nid = q.pop(0)
                    for nb in pol.neighbors_bidir(nid):
                        if nb not in visited:
                            visited.add(nb); q.append(nb); out.append(nb)
            return out

        decisions: list[Decision] = []

        for anc in anchors:
            # 1) 이 앵커의 CU 후보 준비(리랭크가 있다면 적용)
            pre = cand_map.get(anc.id, [])  # [(cu_id, score)]
            cu_ids = [cid for cid, _ in pre]
            # 역할 필터: primary에서 exception은 제외
            cu_ids = [cid for cid in cu_ids
                      if (pol.get_cu(cid).get("attrs", {}) or {}).get("role","").lower() != "exception"]

            # (선택) 의미컷/프레임컷 등 경량 필터는 기존 로직을 그대로 적용해도 됨.

            # 2) Plan 직렬화하여 LLM 1콜(listwise)
            items = []
            n_ok = n_none = n_exc = 0
            for cid in cu_ids:
                node = pol.get_cu(cid)
                attrs = (node.get("attrs") or {})
                # 핵심 필드 미리 스니핑
                subj = attrs.get("subject") or attrs.get("data_category") or attrs.get("title")
                cond = attrs.get("condition"); cons = attrs.get("constraint")
                # 컴파일 시도
                plan = None
                try:
                    plan = self.compiler(cid, node)
                except Exception as e:
                    n_exc += 1
                    # print(f"[plan][EXC] cu={cid}  subj='{(subj or '')[:80]}'  err={repr(e)}")
                    # 계속 진행(폴백 직렬화가 있을 경우)
                if plan and getattr(plan, "cu_type", "actor_cu") != "actor_cu":
                    # 비-actor_cu 스킵
                    # print(f"[plan][SKIP] cu={cid}  non-actor_cu type={getattr(plan,'cu_type','?')}")
                    continue

                if plan is None:
                    n_none += 1
                    # print(f"[plan][NONE] cu={cid}  subj='{(subj or '')[:80]}'  cond='{str(cond)[:60]}'  cons='{str(cons)[:60]}'")
                    # (있다면) 폴백 직렬화 사용
                    if '_plan_to_dict_from_attrs' in globals():
                        items.append({"cu_id": cid, "plan": _plan_to_dict_from_attrs(cid, node)})
                    else:
                        # 최소 안전가드: subject만 담아 보냄
                        items.append({"cu_id": cid, "plan": {
                            "cu_id": cid, "type": "actor_cu",
                            "subject": str(subj or ""), "constraint": cons or [], "condition": cond or "", "context": "", "intent": ""
                        }})
                else:
                    n_ok += 1
                    items.append({"cu_id": cid, "plan": _plan_to_dict(plan)})

            # 앵커 단위 요약
            print(f"[plan][summary][anc={anc.id}] ok={n_ok} none={n_none} exc={n_exc}  feed_items={len(items)}")

            # 대강의 입력 길이(문자수)로 페이로드 크기 감 잡기
            approx_chars = sum(len(json.dumps(it["plan"], ensure_ascii=False)) for it in items)
            # print(f"[llm][in] anc={anc.id} items={len(items)} approx_chars={approx_chars}")

            listwise_res = diag_time_call("llm_listwise", self.llm_backend.judge_anchor_listwise, anc, ctx, items)
            # print(f"[llm][out] anc={anc.id} results={len(listwise_res)}")

            # LLM이 스스로 부족하다고 표시한 경우에만, 제안 질의어로 ctx에서 추가 evidence 스니펫 수집 (간단 substring/키워드)
            # needs_more = any((r.get("verdict") == "INSUFFICIENT" and r.get("proposed_queries")) for r in listwise_res)
            # if needs_more:
            #     # hints = {h for r in listwise_res for h in (r.get("proposed_queries") or [])}
            #     # extra = ctx.search(hints, top=5)   # 간단한 문자열/키워드 검색만 (룰 아님, LLM 지시 준수)
            #     # # 같은 콜에 evidence_pool로 다시 제공
            #     # items2 = [{"cu_id": it["cu_id"], "plan": it["plan"], "extra_evidence": extra} for it in items]
            #     listwise_res = self.llm_backend.judge_anchor_suspect(anc, ctx, items, evidence_pool=None, allow_second_pass=True)  # 재평가 1회

            # listwise_res: [{"cu_id","verdict","score","why","evidence":[]}, ...]
            self.report.add_listwise(anc.id, [
                {"cu_id": r.get("cu_id"), "verdict": r.get("verdict"),
                "score": float(r.get("score",0.0)), "why": r.get("why","")}
                for r in listwise_res
            ])

            # 3) 위반으로 나온 것만 참조조항 한 번 더(앵커당 최대 1콜)
            violated = [r for r in listwise_res if r.get("verdict") == "NON_COMPLIANT"]
            overrides = {}
            if violated:
                sets=[]
                for r in violated:
                    cid = r["cu_id"]
                    node = pol.get_cu(cid)
                    plan = self.compiler(cid, node)
                    ref_ids = _reference_closure(cid)
                    ref_nodes = []
                    for rid in ref_ids:
                        rn = pol.get_cu(rid)
                        ref_nodes.append({
                            "id": rn.get("id"),
                            "type": rn.get("type"),
                            "subject": rn.get("attrs", {}).get("subject"),
                            "constraint": rn.get("attrs", {}).get("constraint"),
                            "condition": rn.get("attrs", {}).get("condition"),
                            "context": rn.get("attrs", {}).get("context"),
                            "role": (rn.get("attrs", {}) or {}).get("role")
                        })
                    sets.append({"base_cu_id": cid, "plan": _plan_to_dict(plan), "references": ref_nodes})

                ref_res = self.llm_backend.judge_references_listwise(anc, ctx, sets)
                for rr in ref_res:
                    overrides[rr["base_cu_id"]] = rr
                            # overrides dict가 아니라 ref_res 원본을 기록
                self.report.add_refs(anc.id, [
                    {"base_cu_id": rr.get("base_cu_id"), "overridden": bool(rr.get("overridden")),
                    "final_verdict": rr.get("final_verdict"), "score": float(rr.get("score",0.0)),
                    "winner": rr.get("winner"), "why": rr.get("why","")}
                    for rr in (ref_res or [])
                ])
            

            # 4) 최종 Decision 생성
            for r in listwise_res:
                cid = r["cu_id"]
                final_verdict = r["verdict"]
                final_score   = float(r.get("score", r.get("confidence", 0.5)))
                final_why     = r.get("why","")
                ev            = r.get("evidence", [])

                if cid in overrides:
                    ov = overrides[cid]
                    if ov.get("overridden"):
                        final_verdict = ov.get("final_verdict","COMPLIANT")
                        # 보수적으로 점수는 평균/최소 중 택1 (여기선 평균)
                        final_score = float(f"{(final_score + float(ov.get('score',0.5)))/2:.3f}")
                        final_why = f"{final_why} | override: {ov.get('why','')}"
                    else:
                        # override 불가면 verdict 유지
                        pass

                decisions.append(Decision(
                    cu_id=cid,
                    article=_article_from_cu_id(cid),
                    cu_type="actor_cu",
                    verdict=final_verdict,
                    score=final_score,
                    why=final_why,
                    evidence=ev,
                    anchor=anc
                ))

        # 기사 단위 집계/정렬(위반 우선) – 기존 로직 유지
        best_by_article: Dict[str, Decision] = {}
        for d in decisions:
            prev = best_by_article.get(d.article)
            if not prev:
                best_by_article[d.article] = d; continue
            if d.verdict == "NON_COMPLIANT" and (prev.verdict != "NON_COMPLIANT" or d.score > prev.score):
                best_by_article[d.article] = d
            elif d.verdict != "NON_COMPLIANT" and prev.verdict != "NON_COMPLIANT" and d.score > prev.score:
                best_by_article[d.article] = d

        final = sorted(best_by_article.values(), key=lambda x: (x.verdict != "NON_COMPLIANT", -x.score))
        if getattr(self, "monitor", None):
            for d in final:
                self.monitor.log_final_decision(d)

        # 최종 결과를 리포트에 기록
        self.report.set_final(final)

        # 요약 프린트 + JSON 저장
        self.report.print_human(pol, max_show=8)
        self.report.dump_json()

        _Diag.report(enable_meta=getattr(self, "enable_meta_gate", False))

        return final

    
class ComplianceMonitor:
    """JSONL 모니터. 한 줄에 하나씩 이벤트 기록."""
    def __init__(self, path: str = "logs/compliance_monitor.jsonl", enabled: bool = True, redact: bool = False):
        self.path = path
        self.enabled = enabled
        self.redact = redact
        self.run_id = str(_uuid.uuid4())
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _write(self, rec: Dict[str, Any]):
        if not self.enabled:
            return
        rec = {**rec, "ts": datetime.datetime.utcnow().isoformat() + "Z", "run_id": self.run_id}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # --- 이벤트별 헬퍼 ---
    def log_anchor(self, anchor: Anchor, ctx: "ContextGraph"):
        a = ctx.get_entity(anchor.actor) if anchor.actor else None
        o = ctx.get_entity(anchor.obj) if anchor.obj else None
        self._write({
            "type": "anchor",
            "anchor_id": anchor.id,
            "predicate": anchor.pred,
            "actor": {"id": getattr(a, "id", None), "name": getattr(a, "name", None), "type": getattr(a, "type", None)},
            "object": {"id": getattr(o, "id", None), "name": getattr(o, "name", None), "type": getattr(o, "type", None)},
            "evidence": (None if self.redact else (anchor.evidence or {}).get("quote"))
        })

    def log_candidates(self, anchor_id: str, items: List[Tuple[str, float]]):
        self._write({"type": "candidates", "anchor_id": anchor_id,
                     "candidates": [{"cu_id": cu, "score": sc} for cu, sc in items]})

    def log_plan(self, anchor_id: str, plan: "Plan"):
        self._write({
            "type": "plan",
            "anchor_id": anchor_id,
            "cu_id": plan.cu_id,
            "intent": plan.intent,
            "action_hint": plan.action_hint,
            "patient_hints": plan.patient_hints,
            "subject": plan.subject_text,
            "constraint": plan.constraint_texts,
            "condition": plan.condition_text,
            "context": plan.context_text
        })

    def log_llm(self, task: str, system: str, payload: Dict[str, Any],
                raw: str, parsed: Dict[str, Any], latency: float, path_used: str, usage: Dict[str, Any]):
        if self.redact:
            payload = {**payload}
            # 필요 시 payload에서 evidence_quote 등 마스킹
            if "anchor" in payload and "evidence_quote" in payload["anchor"]:
                payload["anchor"]["evidence_quote"] = "[REDACTED]"
        self._write({
            "type": "llm_call",
            "task": task,
            "path": path_used,
            "latency_sec": round(latency, 3),
            "system": system,
            "payload": payload,
            "raw": raw,
            "parsed": parsed,
            "usage": usage
        })

    def log_initial_verdict(self, anchor_id: str, cu_id: str, verdict: str, score: float, why: str, evidence: List[str]):
        self._write({"type": "initial_verdict", "anchor_id": anchor_id, "cu_id": cu_id,
                     "verdict": verdict, "score": round(score, 3), "why": why, "evidence": evidence})

    def log_reference_closure(self, cu_id: str, closure_ids: List[str]):
        self._write({"type": "reference_closure", "cu_id": cu_id, "closure_size": len(closure_ids), "closure": closure_ids})

    def log_reference_override(self, anchor_id: str, cu_id: str, changed: bool, to_verdict: Optional[str], note: str):
        self._write({"type": "reference_override", "anchor_id": anchor_id, "cu_id": cu_id,
                     "changed": changed, "to_verdict": to_verdict, "note": note})

    def log_final_decision(self, d: "Decision"):
        self._write({"type": "final_decision", "article": d.article, "cu_id": d.cu_id,
                     "verdict": d.verdict, "score": d.score, "why": d.why, "evidence": d.evidence,
                     "anchor": (d.anchor.as_tuple() if d.anchor else None)})
        
    def log_notebook(self, anchor_id: str, cu_id: str, path: str):
        self._write({"type": "notebook", "anchor_id": anchor_id, "cu_id": cu_id, "path": path})

    def log_meta_gate(self, result: Dict[str, Any]):
        self._write({"type": "meta_gate", **result})


# ------------------------------
# CLI
# ------------------------------
def _cli():
    ap=argparse.ArgumentParser(description="Compliance Gate (GoT, bidirectional references)")
    ap.add_argument("--policy","-p", default="gdpr_snapshot.json")
    ap.add_argument("--context","-c", default="context_graphs.json")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--pred_th", type=float, default=0.5)
    ap.add_argument("--model", type=str, default="gpt-4.1")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--model_path", type=str, default=None, help="로컬 LLM 경로(디렉터리=transformers, .gguf=llama.cpp)")
    args=ap.parse_args()

    if not (os.path.exists(args.policy) and os.path.exists(args.context)):
        print(f"[!] JSON not found: --policy {args.policy}, --context {args.context}"); sys.exit(1)

    with open(args.policy,"r",encoding="utf-8") as f: policy_graph=json.load(f)
    with open(args.context,"r",encoding="utf-8") as f: context_graph=json.load(f)
    
    import os
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY","sk-local")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","sk-local")

    llm = OpenAILLM(model=args.model, temperature=0.0)
    if getattr(args, "model_path", None):
        # llm = LocalFileLLM(model_path=args.model_path, temperature=args.temperature, max_output_tokens=1024)
        llm = OpenAICompatLLM(model=args.model, base_url="http://localhost:8000/v1", api_key=OPENAI_API_KEY,
                              temperature=0.0, max_tokens=3072, debug=False)

    gate = ComplianceGate(pred_threshold=args.pred_th, top_k=args.topk, llm_backend=llm)

    results=gate(policy_graph, context_graph)

    print("\n=== Compliance Gate (GoT) Results ===")
    for d in results:
        print(f"- {d.article} | CU: {d.cu_id} | Verdict: {d.verdict} | Score: {d.score}")
        print("  Why:", d.why)
        for q in d.evidence[:3]:
            print(f"  Evidence: “{q}”")
        if d.anchor: print(f"  Anchor: {d.anchor.as_tuple()}")
        print()

if __name__=="__main__":
    _cli()
