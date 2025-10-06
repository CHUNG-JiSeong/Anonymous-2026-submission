# -*- coding: utf-8 -*-
"""
kg_builder.py

- 프롬프트: YAML 파일 필수 (기본 프롬프트 없음)
- 처리 단위: Point 단위(★ subpoint 레벨 LLM 호출 금지). subpoint 텍스트는 point 텍스트에 포함/병합하여 처리
- 입력 토큰(시스템+유저 포함) 예산 초과 시 문장 분할로 폴백(가능하면 point 전체를 1청크로 유지)
- 배치(멀티팩): 토큰 예산 + pack_size 준수
- 캐시: 파일 기반 키-밸류 캐시 (질의 텍스트 단위)
- 앙상블: LLM 다중 패스 + 임베딩 기반 합의(역할별 클러스터링, 공발생 링크 지지율)
- 레퍼런스: Article/Chapter/Directive/Regulation 참조를 REFERS_TO 엣지로 연결
- 그래프: 조건/예외 그룹(AND/OR) → EFFECT 에 IMPLIES/EXCEPTS 연결

의존:
- (선택) tiktoken: 더 정확한 토큰 카운팅
- (선택) sentence-transformers: 로컬 임베딩 백엔드
- (선택) openai: OpenAI LLM/임베딩
- (선택) networkx: to_networkx() 사용시

CLI:
    python kg_builder.py --input data/gdpr.json --output data/gdpr_kg_snapshot.json \
        --use-llm --prompt configs/prompts.yaml --cache data/kg_cache.jsonl --max-input-tokens 2400
"""
import json, os, re, hashlib, math
import numpy as np
from collections import defaultdict, Counter
from string import Template
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------- optional deps ----------
try:
    import yaml
except ImportError:
    yaml = None

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


# ---------- small utils ----------
def _slug(s: str) -> str:
    s = re.sub(r"\s+", "_", (s or "").strip())
    s = re.sub(r"[^A-Za-z0-9_:\-]", "", s)
    return s or "DOC"

def _short_label(kind: str, number: Any, title: str) -> str:
    if number is not None and str(number) != "":
        return f"{kind.capitalize()} {number}: {title[:60]}"
    return f"{kind.capitalize()}: {title[:60]}"

_ARTICLE_REF_RE = re.compile(r'\bArticle\s+(\d+)(?:\((\d+)\))?(?:\(([a-z])\))?', re.I)
_CHAPTER_REF_RE = re.compile(r'\bChapter\s+([IVXLC]+|\d+)\b', re.I)
_DIRECTIVE_RE   = re.compile(r'\bDirective\s+([0-9]{4}/\d+)\b', re.I)
_REGULATION_RE  = re.compile(r'\bRegulation\s+(?:\(EU\)\s*)?No\.?\s*([0-9/]+)', re.I)
_INTENT_LABELS = ("PROHIBIT", "OBLIGE", "PERMIT", "PERMIT_COND")
_ROLE_LABELS   = ("EFFECT", "EXCEPTION", "DEFINE", "SCOPE", "REMEDY")


# ---------- LLM client ----------
class LLMClient:
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.0, max_tokens: int = 1200):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enabled = bool(os.environ.get("OPENAI_API_KEY"))
        self._use_new_sdk = False
        self._client = None
        if self.enabled:
            try:
                from openai import OpenAI
                self._client = OpenAI()
                self._use_new_sdk = True
            except Exception:
                try:
                    import openai
                    self._client = openai
                    self._use_new_sdk = False
                except Exception:
                    self.enabled = False

    def chat(self, sys: str, usr: str) -> str:
        if not self.enabled or self._client is None:
            raise RuntimeError("OPENAI_API_KEY is not set or SDK not available. Set it or run with use_llm=False.")
        if self._use_new_sdk:
            r = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": usr}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return r.choices[0].message.content
        else:
            r = self._client.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": usr}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return r["choices"][0]["message"]["content"]


# ---------- data model ----------
@dataclass
class Node:
    id: str
    kind: str
    label: str
    attrs: Dict[str, Any | List[Any] | Tuple[Any, ...]] = field(default_factory=dict)
    type: Optional[str] = None

@dataclass
class Edge:
    src: str
    dst: str
    etype: str
    attrs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccuracyOptions:
    # 기존
    process_all: bool = True
    mask_premise_in_norm: bool = True
    ensemble_premise: int = 1
    ensemble_triple: int = 1
    verify_passes: int = 0
    refs_mode: str = "llm"
    build_logic_groups: bool = True
    detect_conflicts: bool = True
    # 임베딩 합의
    use_embed_consensus: bool = True
    embed_backend: str = "openai"           # "openai" | "local"
    embed_model: str = "text-embedding-3-large"
    local_embed_model: str = "sentence-transformers/all-mpnet-base-v2"
    embed_batch_size: int = 64
    sim_th: float = 0.86
    tau: float = 0.60
    link_support_ratio: float = 0.40
    skip_premise_points: bool = True


# ---------- prompts ----------
class PromptStore:
    REQUIRED = {
        "premise":          ["system", "user", "user_batch"],
        "compliance_unit":  ["system", "user", "user_batch"],
    }

    def __init__(self, prompt_path: str):
        if not prompt_path:
            raise RuntimeError("Prompt path is required. Provide a valid YAML config with prompts.")
        if yaml is None:
            raise RuntimeError("PyYAML is required to load prompt YAML. pip install pyyaml")
        with open(prompt_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}

        self.model = y.get("llm", {}).get("model", "gpt-3.5-turbo")
        self.temperature = float(y.get("llm", {}).get("temperature", 0.0))
        self.prompts = y.get("prompts", {})
        self.prompt_path = prompt_path
        self._validate()

    def _validate(self):
        for name, keys in self.REQUIRED.items():
            if name not in self.prompts:
                raise ValueError(f"Missing 'prompts.{name}' in {self.prompt_path}")
            for k in keys:
                v = self.prompts[name].get(k)
                if not isinstance(v, str) or not v.strip():
                    raise ValueError(f"Missing 'prompts.{name}.{k}' in {self.prompt_path}")

    def get(self, name: str, batch: bool = False):
        entry = self.prompts[name]
        return entry["system"], (entry["user_batch"] if batch else entry["user"])

    @staticmethod
    def render_batch(tmpl: str, items_block: str) -> str:
        # 배치 템플릿은 $items 플레이스홀더 권장(중괄호 충돌 방지)
        return Template(tmpl).substitute(items=items_block)


# ---------- token estimator ----------
class TokenEstimator:
    """
    입력 토큰(시스템+유저)을 추산. tiktoken이 있으면 정확 카운팅, 없으면 휴리스틱(약 4 chars ≈ 1 token).
    """
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.enc = None
        if tiktoken is not None:
            try:
                self.enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.enc = None

    def count(self, text: str) -> int:
        if self.enc is not None:
            try:
                return len(self.enc.encode(text))
            except Exception:
                pass
        return max(1, math.ceil(len(text) / 4))

    def chat_input(self, system: str, user: str, overhead: int = 12) -> int:
        return self.count(system) + self.count(user) + overhead


# ---------- simple file cache ----------
class FileCache:
    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.mem: Dict[str, Any] = {}
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        self.mem[obj["key"]] = obj["val"]
                    except Exception:
                        continue

    def get(self, key: str) -> Optional[Any]:
        return self.mem.get(key)

    def set(self, key: str, val: Any):
        self.mem[key] = val
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "val": val}, ensure_ascii=False) + "\n")


def _key(task: str, model: str, text: str) -> str:
    h = hashlib.sha1((task + "|" + model + "|" + text).encode("utf-8")).hexdigest()
    return h


# ---------- core builder ----------
class _CoreKGBuilder:
    def __init__(self, use_llm: bool, llm_client, prompt_store,
                 granularity: str = "point", pack_size: int = 1,
                 cache=None, gate_keywords=None,
                 min_chars_for_llm: int = 60,
                 max_input_tokens: int = 2400,
                 accuracy: Optional[AccuracyOptions] = None):
        self.use_llm = use_llm
        self.llm = llm_client
        self.prompts = prompt_store
        self.granularity = granularity
        self.pack_size = max(1, int(pack_size))
        self.cache = cache
        self.gate_keywords = gate_keywords or ["shall", "shall not", "may", "must", "applies", "does not apply", "means", "refers to"]
        self.min_chars_for_llm = min_chars_for_llm
        self.max_input_tokens = max_input_tokens
        self.tok = TokenEstimator(model=self.prompts.model)
        self.accuracy = accuracy or AccuracyOptions()

        # graph stores
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.article_map: Dict[str, str] = {}
        self.point_map: Dict[Tuple[str, str], str] = {}
        self.subpoint_map: Dict[Tuple[str, str, str], str] = {}
        self._text_index: Dict[str, Dict[str, Any]] = {}
        self._child_seq: Dict[str, int] = {}
        self._warnings: List[str] = []
        self._article_points: Dict[str, List[str]] = {}  # article번호 -> point node_id 리스트

        # premise spans/texts to mask in norm extraction
        self._premise_texts_by_nid: Dict[str, List[str]] = {}

        # embed cache & backend
        self._emb_cache: Dict[str, np.ndarray] = {}
        self._embedder = None  # local sentence-transformers
        self._openai_embed_client = None  # OpenAI Embeddings

    # ------------------ 빌드 파이프라인 ------------------
    def build(self, doc: Dict[str, Any]) -> Tuple[List[Node], List[Edge], List[str]]:
        self._reset()
        root_title = doc.get("title", "Document")
        root_id = f"DOC:{_slug(doc.get('abbrv') or root_title)}"
        self._add_node(root_id, "document", root_title,
                       {"abbrv": doc.get("abbrv"), "regulation": doc.get("regulation")})
        for ch in doc.get("chapters", []):
            self._walk(root_id, ch, ctx_article=None)
        # self._phase_premise()
        self._phase_compliance_unit()
        self._phase_cu_type()
        self._phase_intent_role()
        self._phase_reference()
        self._integrity_checks()
        return list(self.nodes.values()), self.edges, self._warnings

    # ----- walkers -----
    def _walk(self, parent_id: str, node: Dict[str, Any], ctx_article: Optional[str]):
        kind = (node.get("type") or "unknown").lower()
        number = node.get("number")
        title = node.get("title") or node.get("text") or kind
        if number is None or str(number) == "":
            seq = self._child_seq.get(parent_id, 0) + 1
            self._child_seq[parent_id] = seq
            number_for_id = f"auto{seq}"
        else:
            number_for_id = str(number)
        node_id = f"{parent_id}/{kind.upper()}:{number_for_id}"
        label = _short_label(kind, number, title)
        _PREMISE_PAT = re.compile(r"(subject[- ]matter|definition)", re.I)
        section_role = "unknown"
        if kind == "article" and _PREMISE_PAT.search(title or ""):
            section_role = "premise"          # ← 상위 Article 자체를 premise 로 확정
        parent_node = self.nodes.get(parent_id)
        if (
            parent_node
            and parent_node.attrs.get("section_role") == "premise"
        ):
            section_role = "premise"

        self._add_node(node_id, kind, label, {"number": number, "title": node.get("title"), "text": node.get("text"), "section_role": section_role})
        self._add_edge(parent_id, node_id, "CONTAIN")

        if kind == "article" and number is not None:
            self.article_map[str(number)] = node_id
            ctx_article = str(number)
            self._article_points.setdefault(ctx_article, [])

        raw_text = (node.get("text") or "").strip()
        if raw_text:
            art_meta = ctx_article if kind != "article" else (str(number) if number is not None else None)
            self._text_index[node_id] = {
                "kind": kind,
                "article": art_meta,
                "point": number if kind == "point" else None,
                "subpoint": None,
                "raw_text": raw_text,
                "section_role": section_role,
            }

        for child in node.get("contents", []):
            self._walk(node_id, child, ctx_article)

        if kind == "point":
            # point index
            if ctx_article and number is not None:
                self.point_map[(str(ctx_article), str(number))] = node_id
                self._article_points.setdefault(str(ctx_article), []).append(node_id)

            # subpoints (구조만 저장, LLM 호출은 하지 않음)
            for sp in node.get("subpoints", []) or []:
                sp_kind = (sp.get("type") or "subpoint").lower()
                sp_num = sp.get("number")
                sp_text = (sp.get("text") or "").strip()
                if sp_num is None or str(sp_num) == "":
                    seqsp = self._child_seq.get(node_id, 0) + 1
                    self._child_seq[node_id] = seqsp
                    sp_num_for_id = f"auto{seqsp}"
                else:
                    sp_num_for_id = str(sp_num)
                sp_id = f"{node_id}/{sp_kind.upper()}:{sp_num_for_id}"
                sp_label = _short_label(sp_kind, sp_num, sp_text or sp_kind)
                self._add_node(sp_id, sp_kind, sp_label, {"number": sp_num, "text": sp_text})
                self._add_edge(node_id, sp_id, "CONTAIN")
                if ctx_article and node.get("number") is not None and sp_num is not None:
                    a, p, s = str(ctx_article), str(node.get("number")), str(sp_num)
                    self.subpoint_map[(a, p, s)] = sp_id
                if sp_text:
                    self._text_index[sp_id] = {
                        "kind": sp_kind, "article": ctx_article,
                        "point": node.get("number"), "subpoint": sp_num,
                        "raw_text": sp_text
                    }

    # ----- phases -----
    def _phase_premise(self):
        self._run_llm_phase(task="premise")

    def _phase_triple(self):
        self._run_llm_phase(task="triple")

    def _phase_compliance_unit(self):
        self._run_llm_phase(task="compliance_unit")
    
    def _phase_intent_role(self):
        cu_nodes = [n for n in self.nodes.values() if getattr(n, "kind", "") == "compliance_unit"]
        # 문서 적응 centroid 생성에 쓰도록 보관
        self._cu_nodes_for_centroid = cu_nodes

        # # 1) 임베딩 centroid 준비
        # cents = self._ensure_intent_centroids(self, cu_nodes)

        for n in cu_nodes:
            a = n.attrs or {}

            # # 2) 임베딩 추론
            # intent_emb, conf_emb = self._embed_infer_intent(self, a, cents)

            # 3) LLM 재판정(저신뢰 또는 UNSPEC일 때만)
            out = self._llm_infer_intent_role(a)
            a["intent"], a["intent_conf"] = out["intent"], float(out["conf"])
            a["role"],   a["role_conf"]   = out["role"],   float(out["role_conf"])
            if out.get("spans"): a["evidence_spans"] = out["spans"]
            n.attrs = a

    # 매우 보편적인 seed(정책 무종속) — 문서 적응 centroid가 없을 때만 사용
    _SEED_INTENT_TEXTS = {
        "PROHIBIT":     ["shall not process", "is prohibited", "must not do"],
        "OBLIGE":       ["shall ensure", "must implement", "is obliged to"],
        "PERMIT":       ["may process", "is permitted to", "is allowed to"],
        "PERMIT_COND":  ["may ... only if ...", "subject to ...", "provided that ..."],
    }

    def _cos(self, a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0: return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _text_of_cu(self, attrs: dict) -> str:
        """
        Robust stringify for CU attributes:
        - Accepts str / number / bool / list / dict / None
        - dict: 우선순위 키(text,label,name,value,expr,subject,predicate,object,field,op)를 이용해 요약,
                없으면 JSON으로 직렬화
        - list/tuple/set: 각 원소를 재귀적으로 문자열화 후 공백 결합
        반환값: 임베딩 입력용 단일 문자열
        """
        def _strf(v):
            if v is None:
                return ""
            if isinstance(v, str):
                return v
            if isinstance(v, (int, float, bool)):
                return str(v)
            if isinstance(v, (list, tuple, set)):
                return " ".join(s for s in (_strf(x) for x in v) if s)
            if isinstance(v, dict):
                # 가장 정보성 높은 키를 우선 사용
                for k in ("text", "label", "name", "value", "expr",
                        "subject", "predicate", "object", "field", "op"):
                    if k in v and isinstance(v[k], (str, int, float, bool)):
                        return str(v[k])
                # fallback: 짧은 JSON
                try:
                    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                except Exception:
                    return str(v)
            return str(v)

        subj = _strf(attrs.get("subject")).strip()
        cons = _strf(attrs.get("constraint")).strip()

        cond = attrs.get("condition")
        if isinstance(cond, dict):
            # 흔한 스키마: all/any/not/exists 등 혼재 가능
            parts = [
                _strf(cond.get("all")),
                _strf(cond.get("any")),
                _strf(cond.get("not")),
                _strf(cond.get("exists")),
            ]
            cond_s = " ".join(p for p in parts if p).strip()
        else:
            cond_s = _strf(cond).strip()

        ctx  = _strf(attrs.get("context")).strip()

        # 중복·공백 제거하여 한 줄씩 결합
        pieces = [x for x in (cons, cond_s, subj, ctx) if x]
        return "\n".join(pieces)

    def _normalize_intent(self, label: str) -> str:
        s = (label or "").upper().replace("-", "_").replace(" ", "_")
        if "PROHIBIT" in s or "FORBID" in s: return "PROHIBIT"
        if "OBLIGE" in s or s == "OBLIGATION" or "MUST" in s: return "OBLIGE"
        if "PERMIT_COND" in s or "CONDIT" in s or "ONLY_IF" in s: return "PERMIT_COND"
        if "PERMIT" in s or "ALLOW" in s: return "PERMIT"
        return "UNSPEC"

    def _normalize_role(self, label: str) -> str:
        s = (label or "").upper()
        if "EXCEPT" in s or "DEROG" in s: return "EXCEPTION"
        if "DEFINE" in s or "MEAN" in s:  return "DEFINE"
        if "SCOPE" in s or "APPL" in s:   return "SCOPE"
        if "REMEDY" in s or "RIGHT" in s or "COMPLAINT" in s or "JUDICIAL" in s:
            return "REMEDY"
        return "EFFECT"  # 기본은 효과 규범

    # ---- 임베딩 백엔드 (openai / local / 없음) --------------------------------
    def _embed_many(self, texts: list[str]) -> list[np.ndarray]:
        """KGbuilder 내에서 accuracy.embed_backend 설정을 따름."""
        if not texts: return []
        backend = getattr(self.accuracy, "embed_backend", "openai")
        if backend == "openai":
            try:
                from openai import OpenAI
                client = OpenAI()
                model = getattr(self.accuracy, "embed_model", "text-embedding-3-large")
                out = client.embeddings.create(model=model, input=texts)
                vecs = [np.array(d.embedding, dtype=np.float32) for d in out.data]
                return vecs
            except Exception:
                pass  # 폴백
        if backend in ("local", "sentence-transformers"):
            try:
                from sentence_transformers import SentenceTransformer
                model_name = getattr(self.accuracy, "local_embed_model", "sentence-transformers/all-mpnet-base-v2")
                _m = getattr(self, "_embedder_local", None)
                if _m is None or getattr(self, "_embedder_local_name", "") != model_name:
                    self._embedder_local = SentenceTransformer(model_name)
                    self._embedder_local_name = model_name
                vecs = self._embedder_local.encode(texts, batch_size=getattr(self.accuracy, "embed_batch_size", 64), normalize_embeddings=False)
                return [np.array(v, dtype=np.float32) for v in vecs]
            except Exception:
                pass
        # 최종 폴백: bag-of-words 가벼운 해시(성능 낮음, 그러나 크래시 방지)
        import hashlib
        def _bow_hash(t):
            h = hashlib.md5(t.encode("utf-8")).digest()
            # 128-d toy vector
            v = np.frombuffer((h * 8)[:128], dtype=np.uint8).astype(np.float32)
            return (v - v.mean()) / (v.std() + 1e-6)
        return [_bow_hash(t) for t in texts]

    def _ensure_intent_centroids(self, cu_nodes: list) -> dict[str, np.ndarray]:
        """문서 적응 centroid: 규칙/LLM로 이미 확정된 샘플이 있으면 사용, 없으면 seed 사용."""
        if getattr(self, "_intent_centroids", None) is not None:
            return self._intent_centroids
        buckets = {k: [] for k in _INTENT_LABELS}
        # 문서 내 확정 샘플 수집
        for n in cu_nodes:
            a = n.attrs or {}
            lab = self._normalize_intent(a.get("intent"))
            conf = float(a.get("intent_conf") or 0.0)
            if lab in buckets and conf >= 0.8:
                buckets[lab].append(self._text_of_cu(a))
        # 부족한 라벨은 seed로 보완
        for lab in _INTENT_LABELS:
            if not buckets[lab]:
                buckets[lab] = self._SEED_INTENT_TEXTS[lab]
        # 임베딩 → centroid
        centroids = {}
        for lab, texts in buckets.items():
            vecs = self._embed_many(texts)
            if not vecs:
                continue
            centroids[lab] = np.mean(vecs, axis=0)
        self._intent_centroids = centroids
        return centroids

    # ---- (1) 임베딩 기반 추론 ---------------------------------------------------
    def _embed_infer_intent(self, attrs: dict, centroids: dict[str, np.ndarray] | None = None) -> tuple[str, float]:
        """
        returns (intent_label, confidence in [0,1])
        - 정책 무종속: deontic 의미를 임베딩으로 근사, centroid(문서 적응)와 코사인 유사도 최대값을 conf로.
        """
        text = self._text_of_cu(attrs)
        if not text.strip():
            return "UNSPEC", 0.0
        cents = centroids or self._ensure_intent_centroids(self, getattr(self, "_cu_nodes_for_centroid", []))
        if not cents:
            return "UNSPEC", 0.0
        [v] = self._embed_many([text])
        sims = {lab: self._cos(v, cvec) for lab, cvec in cents.items()}
        if not sims:
            return "UNSPEC", 0.0
        top_lab = max(sims, key=sims.get)
        conf = max(0.0, min(1.0, sims[top_lab]))  # 코사인 [0,1]로 사용
        return top_lab, conf

    # ---- (2) LLM 기반 재판정 -----------------------------------------------------
    def _llm_infer_intent_role(self, attrs: dict, model: str = "gpt-4o", temperature: float = 0.0) -> dict:
        """
        returns {"intent": str, "conf": float, "role": str, "role_conf": float, "spans": list}
        - JSON 강제 파싱; 예외 시 UNSPEC/EFFECT로 폴백.
        - prompts.yaml 없이도 동작하도록 inline system/user 구성(정책-무종속).
        """
        try:
            from openai import OpenAI
            client = OpenAI()
            sys_msg = (
                "You classify a legal norm fragment (compliance unit) into policy-agnostic labels.\n"
                "Intent labels: PROHIBIT, OBLIGE, PERMIT, PERMIT_COND, or UNSPEC.\n"
                "Role labels: EFFECT, EXCEPTION, DEFINE, SCOPE, REMEDY.\n"
                "Return STRICT JSON with fields: {intent, intent_conf, role, role_conf, evidence_spans}.\n"
                "evidence_spans is a list of short quotes that justified your decision."
            )
            user = {
                "subject": attrs.get("subject"),
                "constraint": attrs.get("constraint"),
                "condition": attrs.get("condition"),
                "context": attrs.get("context"),
            }
            # Chat Completions (Responses API 미지원 환경 호환)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys_msg},
                        {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
                temperature=temperature
            )
            raw = resp.choices[0].message.content or "{}"
            try:
                out = json.loads(raw)
            except Exception:
                # 관대한 파서: 첫 {..} 블록 추출
                m = re.search(r"\{.*\}", raw, flags=re.S)
                out = json.loads(m.group(0)) if m else {}
            intent = self._normalize_intent(out.get("intent"))
            role   = self._normalize_role(out.get("role"))
            iconf  = float(out.get("intent_conf", 0.6))
            rconf  = float(out.get("role_conf", iconf))
            spans  = out.get("evidence_spans") or []
            if not isinstance(spans, list): spans = [str(spans)]
            # 안전 클램프
            iconf = max(0.0, min(1.0, iconf)); rconf = max(0.0, min(1.0, rconf))
            # 폴백
            if intent not in _INTENT_LABELS: intent = "UNSPEC"
            if role not in _ROLE_LABELS:     role   = "EFFECT"
            return {"intent": intent, "conf": iconf, "role": role, "role_conf": rconf, "spans": spans}
        except Exception:
            # LLM 사용 불가 시 폴백
            return {"intent": "UNSPEC", "conf": 0.0, "role": "EFFECT", "role_conf": 0.0, "spans": []}

    # ----- item collection (POINT 단위로만) -----
    def _collect_items_for_task(self, task: str) -> List[Tuple[str, str]]:
        """
        LLM 추출 대상 텍스트 수집.
        - 원칙: POINT 단위. subpoint는 LLM 대상에서 제외하되, 텍스트는 point에 병합하여 컨텍스트로 제공.
        - 예외: 포인트가 전혀 없는 기사(Article)는 Article 텍스트 자체를 대상으로 함.
        """
        out: List[Tuple[str, str]] = []

        # 1) Article별 포인트가 있으면 포인트 단위
        for (a_num, plist) in self._article_points.items():
            if plist:
                for pid in plist:
                    txt = self._aggregate_point_text(pid)  # point 텍스트 + subpoint 텍스트 병합
                    if txt:
                        out.append((pid, txt))
            else:
                # 2) 포인트가 전혀 없는 Article은 그 Article 텍스트 사용
                aid = self.article_map.get(str(a_num))
                if aid:
                    txt = (self._text_index.get(aid, {}) or {}).get("raw_text", "")
                    if txt:
                        out.append((aid, txt))

        # 일부 문서 구조에서 chapter/section에 고립 텍스트가 있을 수 있음 (fallback)
        # 이미 article/point에서 커버되지 않은 것만 추가
        covered = set(nid for nid, _ in out)
        for nid, meta in self._text_index.items():
            if nid in covered:
                continue
            if meta.get("kind") in {"point", "subpoint"}:
                continue  # point 경유로 이미 처리됨
            if meta.get("kind") == "article":
                continue  # 위에서 처리됨
            # 기타 텍스트 노드(드물다)
            txt = (meta.get("raw_text") or "").strip()
            if txt:
                out.append((nid, txt))
        return out

    def _aggregate_point_text(self, pid: str) -> str:
        """
        point 텍스트 + 해당 point의 모든 subpoint 텍스트를 병합하여 하나의 청크로 반환.
        """
        base = (self._text_index.get(pid, {}) or {}).get("raw_text", "") or ""
        # subpoint children
        subs = []
        for nid in self.nodes:
            if nid.startswith(pid + "/SUBPOINT:"):
                st = (self._text_index.get(nid, {}) or {}).get("raw_text", "")
                if st:
                    subs.append(st)
        if subs:
            return (base + "\n" + "\n".join(subs)).strip()
        return base.strip()

    # ----- token-budgeted LLM runner ---------------------------------------
    def _run_llm_phase(self, task: str):
        # 0) 프롬프트 로드
        sys, usr_single = self.prompts.get(task, batch=False)
        _,  usr_batch  = self.prompts.get(task, batch=True)

        # 1) 처리 대상 수집 (POINT 단위, sub-point는 point에 병합)
        items = self._collect_items_for_task(task)

        # 1-a) CU 단계: premise 라벨 노드 완전 배제
        if task == "compliance_unit":
            items = [
                (nid, txt) for nid, txt in items
                if self._text_index.get(nid, {}).get("section_role") != "premise"
            ]

        # 2) 길이 또는 규범 키워드(gate_keywords)로 최소 게이트
        kw_re = re.compile("|".join(re.escape(k) for k in self.gate_keywords), re.I)
        gated = [
            (nid, text) for nid, text in items
            if len(text) >= self.min_chars_for_llm or kw_re.search(text)
        ]
        if not gated:
            return  # 보낼 게 없음

        # 3) premise/ CU 단계에서 premise 텍스트 마스킹
        if task in {"triple", "compliance_unit"} and self.accuracy.mask_premise_in_norm:
            masked = []
            for nid, text in gated:
                for m in self._premise_texts_by_nid.get(nid, []):
                    text = re.sub(re.escape(m), " ", text, flags=re.I)
                masked.append((nid, re.sub(r"\s+", " ", text).strip()))
            gated = masked

        # 4) 캐시 조회
        misses, cache_hits = [], {}
        for nid, text in gated:
            key = _key(task, self.prompts.model, text)
            val = self.cache.get(key) if self.cache else None
            if val is None:
                misses.append((nid, text, key))
            else:
                cache_hits[nid] = val

        # 5) 토큰 예산 기반 배치 포장
        batches = self._pack_by_token_budget(misses, sys, usr_batch)

        # 6) 패스별 결과 모음
        results_passes: Dict[str, List[Dict[str, Any]]] = {}
        for nid, obj in cache_hits.items():
            results_passes.setdefault(nid, []).append(obj)

        # 7) LLM 호출 (premise는 앙상블 K≥1, CU는 K=1)
        if self.use_llm and self.llm and batches:
            K = max(1, self.accuracy.ensemble_premise) if task == "premise" else 1

            for pack in batches:
                items_block = self._format_items_block([(nid, txt) for nid, txt, _ in pack])
                prompt_user  = self.prompts.render_batch(usr_batch, items_block)

                # 토큰 초과 시 → 단건 / 문장 분할 폴백
                def _call_single(nid_, txt_, key_):
                    single_user = usr_single.replace("{chunk}", txt_)
                    for _ in range(K):
                        try:
                            out = self.llm.chat(sys, single_user)
                            obj = self._parse_json(out)
                            results_passes.setdefault(nid_, []).append(obj)
                            if self.cache:
                                self.cache.set(key_, obj)
                        except Exception:
                            continue

                if self.tok.chat_input(sys, prompt_user) > self.max_input_tokens:
                    for nid, txt, key in pack:
                        # 단건으로 재시도
                        if self.tok.chat_input(sys, usr_single.replace("{chunk}", txt)) <= self.max_input_tokens:
                            _call_single(nid, txt, key)
                        else:
                            # 문장 분할
                            for ptxt in self._split_by_token_budget(txt, sys, usr_batch, nid):
                                _call_single(nid, ptxt, _key(task, self.prompts.model, ptxt))
                    continue

                # 정상 배치 호출
                for _ in range(K):
                    try:
                        out = self.llm.chat(sys, prompt_user)
                        obj = self._parse_json(out)
                        for it in (obj.get("results") or []):
                            nid_r = it.get("nid")
                            if nid_r:
                                results_passes.setdefault(nid_r, []).append(it)
                        # 캐시에 마지막 패스 저장
                        if self.cache:
                            for nid, txt, key in pack:
                                lst = results_passes.get(nid, [])
                                if lst:
                                    self.cache.set(key, lst[-1])
                    except Exception:
                        # 배치 실패 → 단건 폴백
                        for nid, txt, key in pack:
                            _call_single(nid, txt, key)

        # 8) 합의·머지
        for nid, passes in results_passes.items():
            # if task == "premise":
            #     agreed = self._consensus_premises(passes)

            #     # 8-a) 마스킹용 premise 저장
            #     if agreed:
            #         self._premise_texts_by_nid[nid] = [
            #             p["text"] for p in agreed if p.get("text")
            #         ]

            #     # 8-b) **premise 라벨 확정 & 하위 노드 전파**
            #     self.nodes[nid].attrs["section_role"] = "premise"
            #     if nid in self._text_index:
            #         self._text_index[nid]["section_role"] = "premise"
            #     # prefix 기반 하위 노드 전파
            #     for cid in self.nodes:
            #         if cid.startswith(nid + "/"):
            #             self.nodes[cid].attrs["section_role"] = "premise"
            #             if cid in self._text_index:
            #                 self._text_index[cid]["section_role"] = "premise"

            #     # 8-c) premise 노드 materialize
            #     for pm in agreed:
            #         txt = (pm.get("text") or "").strip()
            #         if not txt:
            #             continue
            #         pid = f"{nid}/PREMISE:{abs(hash((nid, txt, pm.get('category', '')))) % (10**12)}"
            #         self._add_node(pid, "premise", txt[:80],
            #                        {"category": pm.get("category"), "text": txt})
            #         self._add_edge(nid, pid, "HAS_PREMISE")

            if task == "compliance_unit":
                cu_list = self._merge_compliance_units(passes)
                self._materialize_compliance_units(nid, cu_list)

            elif task == "triple":
                triples = self._consensus_triples_with_links(
                    passes,
                    tau=self.accuracy.tau,
                    sim_th=self.accuracy.sim_th,
                    link_support_ratio=self.accuracy.link_support_ratio
                )
                self._materialize_logic_and_norms(nid, triples)

    # ================== CU 전용 메서드 ==================
    _SUBJ_RE = re.compile(
        r"^(?:The|A|An)?\s*(?P<subj>[^,.;:]+?)\s+"
        r"(?:shall|must|may|is\s+entitled|has\s+the\s+right|has\s+an\s+obligation)\b",
        re.I,
    )

    def _extract_subject_and_constraint(self, effect: str) -> Tuple[str, str]:
        m = self._SUBJ_RE.match(effect.strip())
        if m:
            subj = m.group("subj").strip()
            rest = effect[m.end():].lstrip(" ,.:;")
            return subj, rest
        return "", effect.strip()

    def _find_context_text(self, nid: str) -> str:
        parts = nid.split("/")
        if len(parts) < 2:
            return ""
        parent = "/".join(parts[:-1])
        meta = self._text_index.get(parent) or {}
        return (meta.get("raw_text") or "").strip()

    def _merge_compliance_units(self, passes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen, out = set(), []
        for p in passes:
            for cu in (p.get("compliance_units") or []):
                key = json.dumps(cu, sort_keys=True, ensure_ascii=False)
                if key not in seen:
                    seen.add(key)
                    out.append(cu)
        return out

    def _materialize_compliance_units(self, nid: str, cu_list: List[Dict[str, Any]]):
        for cu in cu_list:
            body_json = json.dumps(cu, ensure_ascii=False)
            cu_id = f"{nid}/CU:{abs(hash(body_json)) % (10**12)}"
            self._add_node(cu_id, "compliance_unit", body_json[:80], cu)
            self._add_edge(nid, cu_id, "DERIVES")
    
    # === (B) _CoreKGBuilder 내부에 신규 메서드 추가 ===
    def _phase_cu_type(self):
        """
        LLM으로 compliance_unit 노드의 최상위 node.type을 'actor_cu' 또는 'meta_cu'로 분류.
        - 사전/정규식/룰 없이 CU의 label/attrs 텍스트만 사용
        - prompts.yaml 의 prompts.cu_type {system, user_batch} 필요(없으면 조용히 스킵)
        - 이미 node.type 이 있으면 건너뜀(재실행 안전)
        - 결과는 '최상위 type'만 기록하고, attrs 내부의 과거 type 관련 키는 제거
        """
        # 프롬프트 존재 확인
        if not hasattr(self.prompts, "prompts") or "cu_type" not in self.prompts.prompts:
            return
        sys_tmpl, _ = self.prompts.get("cu_type", batch=False)
        _, usr_batch_tmpl = self.prompts.get("cu_type", batch=True)

        def _prune_cu_type_attrs(n: Node):
            # 과거 실행에서 남았을 수 있는 추적/설명 키 제거
            for k in ("type", "type_confidence", "type_subject_span", "type_rationale"):
                try:
                    if k in n.attrs:
                        n.attrs.pop(k, None)
                except Exception:
                    pass

        # 대상 CU 수집
        cu_nodes = [n for n in self.nodes.values()
                    if n.kind == "compliance_unit" and n.type not in {"actor_cu", "meta_cu"}]
        if not cu_nodes or not self.use_llm or not self.llm:
            return

        # 입력 텍스트 구성(사전/룰 없이 있는 값만 직조)
        items: List[Tuple[str, str]] = []
        for n in cu_nodes:
            a = n.attrs or {}
            parts = []
            if isinstance(n.label, str) and n.label.strip():
                parts.append(f"label: {n.label.strip()}")
            for k in ("subject", "condition", "constraint", "context"):
                v = a.get(k)
                if v:
                    parts.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
            txt = "\n".join(parts).strip()
            if txt:
                items.append((n.id, txt))
        if not items:
            return

        # 토큰 예산 기반 간단 배치 포장
        batches: List[List[Tuple[str, str]]] = []
        cur, cur_tokens = [], 0
        for nid, txt in items:
            snippet = txt if len(txt) < 3000 else (txt[:2970] + "...")
            probe = Template(usr_batch_tmpl).substitute(
                items=json.dumps([{"id": nid, "text": snippet}], ensure_ascii=False)
            )
            est = self.tok.chat_input(sys_tmpl, probe)
            if cur and (cur_tokens + est) > self.max_input_tokens:
                batches.append(cur); cur, cur_tokens = [], 0
            cur.append((nid, snippet)); cur_tokens += est
        if cur:
            batches.append(cur)

        # LLM 호출(배치) + 결과 반영
        def _parse_lines_as_jsons(s: str) -> List[Dict[str, Any]]:
            out = []
            try:
                obj = self._parse_json(s)
                if isinstance(obj, dict) and "items" in obj:
                    return obj["items"]
                if isinstance(obj, list):
                    return obj
            except Exception:
                pass
            for line in (s or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
            return out

        for pack in batches:
            block = "\n".join(json.dumps({"id": nid, "text": txt}, ensure_ascii=False)
                            for nid, txt in pack)
            prompt_user = Template(usr_batch_tmpl).substitute(items=block)

            # 캐시
            key = _key("cu_type", self.prompts.model, prompt_user)
            raw = self.cache.get(key) if self.cache else None
            if not raw:
                raw = self.llm.chat(sys_tmpl, prompt_user)
                if self.cache:
                    self.cache.set(key, raw)

            results = _parse_lines_as_jsons(raw)
            for r in results:
                nid = r.get("id")
                ty = (r.get("type") or "").strip().lower()
                if nid and ty in {"actor_cu", "meta_cu"} and nid in self.nodes:
                    node = self.nodes[nid]
                    node.type = ty                  # ← 최상위에만 기록
                    _prune_cu_type_attrs(node)      # ← attrs 내부 잔여 키 제거


    # ----- packing / splitting helpers -----
    def _pack_by_token_budget(self, misses: List[Tuple[str, str, str]], sys: str, usr_batch_tmpl: str) -> List[List[Tuple[str, str, str]]]:
        batches: List[List[Tuple[str, str, str]]] = []
        cur: List[Tuple[str, str, str]] = []
        for trip in misses:
            probe_items = cur + [trip]
            items_block = self._format_items_block([(n, t) for n, t, _ in probe_items])
            user_probe = Template(usr_batch_tmpl).substitute(items=items_block)
            tokens = self.tok.chat_input(sys, user_probe)
            if tokens <= self.max_input_tokens and len(probe_items) <= self.pack_size:
                cur.append(trip)
            else:
                if cur:
                    batches.append(cur)
                cur = [trip]
        if cur:
            batches.append(cur)
        return batches

    def _split_by_token_budget(self, text: str, sys: str, usr_batch_tmpl: str, base_nid: str) -> List[str]:
        """
        문장 기반 분할로 입력 토큰 예산을 만족시키도록 청크를 나눔.
        (가능하면 point 전체를 1청크로 유지)
        """
        sents = re.split(r'(?<=[.?!])\s+', text)
        chunks, buf = [], ""
        for s in sents:
            candidate = (buf + " " + s).strip() if buf else s
            items_block = self._format_items_block([(base_nid, candidate)])
            user_probe = Template(usr_batch_tmpl).substitute(items=items_block)
            if self.tok.chat_input(sys, user_probe) <= self.max_input_tokens:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                    buf = s
                else:
                    chunks.extend(self._hard_cut(s, sys, usr_batch_tmpl, base_nid))
                    buf = ""
        if buf:
            chunks.append(buf)
        return chunks

    def _hard_cut(self, text: str, sys: str, usr_batch_tmpl: str, base_nid: str, step_chars: int = 800) -> List[str]:
        out = []
        i = 0
        while i < len(text):
            j = min(len(text), i + step_chars)
            piece = text[i:j]
            items_block = self._format_items_block([(base_nid, piece)])
            user_probe = Template(usr_batch_tmpl).substitute(items=items_block)
            if self.tok.chat_input(sys, user_probe) <= self.max_input_tokens:
                out.append(piece)
                i = j
            else:
                step_chars = max(200, step_chars // 2)
        return out or [text[:step_chars]]

    def _format_items_block(self, pairs: List[Tuple[str, str]]) -> str:
        blocks = []
        for nid, txt in pairs:
            blocks.append(f"### ITEM\nNID: {nid}\nTEXT:\n{txt}\n---")
        return "\n".join(blocks)

    # ----- references -----
    def _phase_reference(self):
        for nid, meta in self._text_index.items():
            text = meta.get("raw_text", "")
            if not text:
                continue
            refs = self._scan_refs(text)
            for r in refs:
                tgt = self._resolve_ref(r)
                if tgt:
                    self._add_edge(nid, tgt, "REFERS_TO")
                else:
                    ext_id = f"{nid}/EXTERN:{r}"
                    self._add_node(ext_id, "extern_ref", r, {})
                    self._add_edge(nid, ext_id, "REFERS_TO")

    def _scan_refs(self, text: str) -> List[str]:
        refs = set()
        for m in _ARTICLE_REF_RE.finditer(text):
            a, p, s = m.groups()
            r = f"Article {a}"
            if p: r += f"({p})"
            if s: r += f"({s})"
            refs.add(r)
        for m in _CHAPTER_REF_RE.finditer(text):
            refs.add(f"Chapter {m.group(1)}")
        for m in _DIRECTIVE_RE.finditer(text):
            refs.add(f"Directive {m.group(1)}")
        for m in _REGULATION_RE.finditer(text):
            refs.add(f"Regulation No {m.group(1)}")
        return sorted(refs)

    def _resolve_ref(self, ref: str) -> Optional[str]:
        m = _ARTICLE_REF_RE.match(ref)
        if m:
            a, p, s = m.groups()
            if a and p and s:
                return self.subpoint_map.get((str(a), str(p), str(s)))
            if a and p:
                return self.point_map.get((str(a), str(p)))
            if a:
                return self.article_map.get(str(a))
        return None

    # ----- graph ops / misc -----
    def _add_node(self, id: str, kind: str, label: str, attrs: Dict[str, Any]) -> str:
        if id in self.nodes:
            self.nodes[id].attrs.update({k: v for k, v in attrs.items() if v is not None})
            return id
        self.nodes[id] = Node(id=id, kind=kind, label=label, attrs=attrs or {})
        return id

    def _add_edge(self, src: str, dst: str, etype: str, attrs: Optional[Dict[str, Any]] = None):
        self.edges.append(Edge(src=src, dst=dst, etype=etype, attrs=attrs or {}))

    def _integrity_checks(self):
        parents: Dict[str, int] = {}
        for e in self.edges:
            if e.etype == "CONTAIN":
                parents[e.dst] = parents.get(e.dst, 0) + 1
        for nid, cnt in parents.items():
            if cnt > 1 and self.nodes[nid].kind in {"point", "subpoint", "section"}:
                self._warnings.append(f"[MULTI-PARENT] {nid} has {cnt} CONTAIN parents")

    def _reset(self):
        self.nodes.clear()
        self.edges.clear()
        self.article_map.clear()
        self.point_map.clear()
        self.subpoint_map.clear()
        self._text_index.clear()
        self._child_seq.clear()
        self._warnings.clear()
        self._article_points.clear()
        self._premise_texts_by_nid.clear()
        self._emb_cache.clear()

    def _parse_json(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except Exception:
            cleaned = re.sub(r"^```(json)?\s*|\s*```$", "", text.strip(), flags=re.M)
            return json.loads(cleaned)

    # =========================
    #  임베딩 합의 (Premise & Triple)
    # =========================
    def _prep_text(self, s: str) -> str:
        return re.sub(r'\s+', ' ', (s or '')).strip()

    # --- Embedding backends ---
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        payload, idx = [], []
        for t in texts:
            k = self._prep_text(t)
            if k not in self._emb_cache:
                payload.append(k); idx.append(k)
        if payload:
            vecs = self._embed_backend()(payload)
            for k, v in zip(idx, vecs):
                self._emb_cache[k] = v
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return np.stack([self._emb_cache[self._prep_text(t)] for t in texts], axis=0)

    def _embed_backend(self):
        bk = (self.accuracy.embed_backend or "openai").lower()
        if bk == "local":
            if self._embedder is None:
                if SentenceTransformer is None:
                    self._warnings.append("[EMBED] sentence-transformers 미설치 → 토큰합의 폴백")
                    return self._embed_backend_fallback
                self._embedder = SentenceTransformer(self.accuracy.local_embed_model)
            def local_fn(texts: List[str]) -> np.ndarray:
                return self._embedder.encode(texts, normalize_embeddings=True)
            return local_fn
        else:
            # OpenAI embeddings
            if not os.environ.get("OPENAI_API_KEY"):
                self._warnings.append("[EMBED] OPENAI_API_KEY 없음 → 토큰합의 폴백")
                return self._embed_backend_fallback
            try:
                from openai import OpenAI
                if self._openai_embed_client is None:
                    self._openai_embed_client = OpenAI()
            except Exception:
                self._warnings.append("[EMBED] openai SDK 문제 → 토큰합의 폴백")
                return self._embed_backend_fallback
            model = self.accuracy.embed_model
            bs = max(1, int(self.accuracy.embed_batch_size))
            def openai_fn(texts: List[str]) -> np.ndarray:
                out_vecs = []
                for i in range(0, len(texts), bs):
                    seg = texts[i:i+bs]
                    resp = self._openai_embed_client.embeddings.create(model=model, input=seg)
                    out_vecs.extend([np.array(d.embedding, dtype=np.float32) for d in resp.data])
                return np.stack(out_vecs, axis=0) if out_vecs else np.zeros((0,1),dtype=np.float32)
            return openai_fn

    def _embed_backend_fallback(self, texts: List[str]) -> np.ndarray:
        # 아주 보수적인 bag-of-words 폴백
        vocab = {}
        toks_list = []
        for t in texts:
            toks = self._prep_text(t).lower().split()
            toks_list.append(toks)
            for w in toks:
                if w not in vocab: vocab[w] = len(vocab)
        if not vocab:
            return np.zeros((len(texts), 1), dtype=np.float32)
        arr = np.zeros((len(texts), len(vocab)), dtype=np.float32)
        for i, toks in enumerate(toks_list):
            for w in toks:
                arr[i, vocab[w]] += 1.0
        n = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        return arr / n

    # --- clustering & consensus ---
    def _cosine(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
        return A @ B.T

    def _cluster_threshold(self, texts: List[str], vecs: np.ndarray, sim_th: float) -> List[List[int]]:
        n = len(texts); clusters: List[List[int]] = []
        for i in range(n):
            placed = False
            for c in clusters:
                rep = c[0]
                v = float(np.dot(vecs[i], vecs[rep]) /
                          ((np.linalg.norm(vecs[i]) + 1e-9) * (np.linalg.norm(vecs[rep]) + 1e-9)))
                if v >= sim_th:
                    c.append(i); placed = True; break
            if not placed:
                clusters.append([i])
        # 메도이드 갱신
        for c in clusters:
            if len(c) >= 2:
                sub = vecs[c]
                S = self._cosine(sub, sub)
                avg = (np.sum(S, axis=1) - 1.0) / max(1, len(c)-1)
                rep_idx = int(np.argmax(avg))
                head = c[rep_idx]; c.remove(head); c.insert(0, head)
        return clusters

    def _consensus_role_texts(self, all_texts: List[str], K: int, tau: float, sim_th: float) -> List[str]:
        all_texts = [self._prep_text(t) for t in all_texts if t]
        if not all_texts:
            return []
        uniq = []
        for t in all_texts:
            if t not in uniq:
                uniq.append(t)
        vecs = self._embed_texts(uniq)
        clusters = self._cluster_threshold(uniq, vecs, sim_th=sim_th)
        counts = Counter(all_texts)
        min_support = max(1, math.ceil(K * tau))
        reps = []
        for c in clusters:
            support = sum(counts[uniq[i]] for i in c)
            if support >= min_support:
                reps.append(uniq[c[0]])
        seen, out = set(), []
        for r in reps:
            if r not in seen:
                seen.add(r); out.append(r)
        return out

    def _consensus_premises(self, passes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """premise 합의: 텍스트 임베딩 + 카테고리 다수결"""
        K = max(1, len(passes))
        texts, cats = [], defaultdict(Counter)
        for p in passes:
            for it in (p.get("premises") or []):
                t = self._prep_text(it.get("text") or "")
                if t:
                    texts.append(t)
                    cats[t][(it.get("category") or "other")] += 1
        reps = self._consensus_role_texts(texts, K=K, tau=self.accuracy.tau, sim_th=self.accuracy.sim_th)
        out = []
        for r in reps:
            cat = (cats[r].most_common(1)[0][0] if cats[r] else "other")
            out.append({"text": r, "category": cat})
        return out

    def _consensus_triples_with_links(self, passes: List[Dict[str, Any]],
                                      tau: float, sim_th: float, link_support_ratio: float) -> List[Dict[str, Any]]:
        """triple 합의: 역할별 임베딩 합의 + 공발생 링크 지지 기반 연결"""
        K = max(1, len(passes))
        cond_all, eff_all, exc_all, ops = [], [], [], []
        cooc = []  # (conds, effect, excs, op)
        for r in passes:
            for tr in (r.get("triples") or []):
                # triple 스키마: condition: [..], exception: [..], effect: str, op: AND/OR
                C_in = tr.get("condition")
                X_in = tr.get("exception")
                E_in = tr.get("effect")
                # 단일 문자열로 온 경우 배열화
                C = []
                if isinstance(C_in, list):
                    C = [self._prep_text(x) for x in C_in if x]
                elif isinstance(C_in, str):
                    C = [self._prep_text(C_in)]
                X = []
                if isinstance(X_in, list):
                    X = [self._prep_text(x) for x in X_in if x]
                elif isinstance(X_in, str):
                    X = [self._prep_text(X_in)]
                E = self._prep_text(E_in or "")
                if E:
                    cond_all.extend(C); eff_all.append(E); exc_all.extend(X)
                    op = (tr.get("op") or tr.get("logic_op") or "AND")
                    ops.append(op); cooc.append((C, E, X, op))
        cond_star = self._consensus_role_texts(cond_all, K=K, tau=tau, sim_th=sim_th)
        eff_star  = self._consensus_role_texts(eff_all,  K=K, tau=tau, sim_th=sim_th)
        exc_star  = self._consensus_role_texts(exc_all,  K=K, tau=tau, sim_th=sim_th)
        OP = Counter([o for o in ops if o in ("AND", "OR")])
        op_star = "AND" if OP["AND"] >= OP["OR"] else "OR"

        min_link = max(1, math.ceil(K * link_support_ratio))
        cond_star_set, eff_star_set, exc_star_set = set(cond_star), set(eff_star), set(exc_star)
        CE, XE = defaultdict(int), defaultdict(int)
        for C, E, X, _ in cooc:
            for c in C:
                if c in cond_star_set and E in eff_star_set:
                    CE[(c, E)] += 1
            for x in X:
                if x in exc_star_set and E in eff_star_set:
                    XE[(x, E)] += 1
        triples = []
        for e in eff_star:
            cs = [c for (c, ee), cnt in CE.items() if ee == e and cnt >= min_link]
            xs = [x for (x, ee), cnt in XE.items() if ee == e and cnt >= min_link]
            triples.append({"condition": cs, "effect": e, "exception": xs, "op": op_star})
        return triples

    def _materialize_logic_and_norms(self, nid: str, triples: List[Dict[str, Any]]):
        """합의된 triples를 실제 그래프 노드/엣지로 반영 (POINT 단위)"""
        for tr in triples:
            eff  = (tr.get("effect") or "").strip()
            conds = [c.strip() for c in (tr.get("condition") or []) if c]
            excs  = [x.strip() for x in (tr.get("exception") or []) if x]
            if not eff:
                continue
            eff_id = f"{nid}/NORM:EFF:{abs(hash((nid, eff)))%(10**12)}"
            self._add_node(eff_id, "norm", eff[:80], {"role": "EFFECT", "text": eff})
            self._add_edge(nid, eff_id, "BELONGS_TO")
            op = (tr.get("op") or "AND")
            if conds:
                gid = f"{nid}/LOGIC:G{abs(hash((nid, 'C', tuple(sorted(conds)), op)))%(10**9)}"
                self._add_node(gid, "logic", op, {"op": op})
                for c in conds:
                    cid = f"{nid}/NORM:COND:{abs(hash((nid, c)))%(10**12)}"
                    self._add_node(cid, "norm", c[:80], {"role": "CONDITION", "text": c})
                    self._add_edge(cid, gid, "IN_GROUP")
                    self._add_edge(nid, cid, "BELONGS_TO")
                self._add_edge(gid, eff_id, "IMPLIES")
            if excs:
                xid = f"{nid}/LOGIC:E{abs(hash((nid, 'X', tuple(sorted(excs)), op)))%(10**9)}"
                self._add_node(xid, "logic", op, {"op": op})
                for x in excs:
                    xid0 = f"{nid}/NORM:EXC:{abs(hash((nid, x)))%(10**12)}"
                    self._add_node(xid0, "norm", x[:80], {"role": "EXCEPTION", "text": x})
                    self._add_edge(xid0, xid, "IN_GROUP")
                    self._add_edge(nid, xid0, "BELONGS_TO")
                self._add_edge(xid, eff_id, "EXCEPTS")


# ---------- user-facing Graph wrapper ----------
class _Graph:
    def __init__(self, nodes: List[Node], edges: List[Edge], warnings: List[str], conflicts: Optional[List[str]] = None):
        self.nodes = nodes
        self.edges = edges
        self.warnings = warnings or []
        self.conflicts = conflicts or []

    def to_networkx(self):
        if nx is None:
            raise RuntimeError("pip install networkx")
        G = nx.MultiDiGraph()
        for n in self.nodes:
            G.add_node(n.id, kind=n.kind, label=n.label, **n.attrs)
        for e in self.edges:
            G.add_edge(e.src, e.dst, key=e.etype, etype=e.etype, **e.attrs)
        return G

    def to_json(self) -> Dict[str, Any]:
        return {
            "nodes": [(lambda d, n: (d.update({"type": n.type}) if n.type else None) or d)(
                    dict(id=n.id, kind=n.kind, label=n.label, attrs=n.attrs), n
                )
                for n in self.nodes],
            "edges": [dict(src=e.src, dst=e.dst, etype=e.etype, attrs=e.attrs) for e in self.edges],
            "meta": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "warnings": self.warnings,
                "warning_count": len(self.warnings),
                "conflicts": self.conflicts,
                "conflict_count": len(self.conflicts),
            }
        }

    def export_snapshot(self, path: Optional[str] = None, fmt: str = "json") -> Dict[str, Any]:
        snap = self.to_json()
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)
        return snap


# ---------- public API ----------
class KGbuilder:
    """
    g = KGbuilder(doc, use_llm=True, prompt_path="prompt.yaml",
                  granularity="point", pack_size=8,
                  cache_path="/mnt/data/kg_cache.jsonl",
                  max_input_tokens=500, accuracy=AccuracyOptions())
    snap = g.export_snapshot("/mnt/data/gdpr_kg_snapshot.json")
    """
    def __init__(self, doc: Dict[str, Any], use_llm: bool = True, prompt_path: Optional[str] = None,
                 model: Optional[str] = None, temperature: Optional[float] = None,
                 granularity: str = "point", pack_size: int = 8,
                 cache_path: Optional[str] = None,
                 min_chars_for_llm: int = 60,
                 max_input_tokens: int = 500,
                 accuracy: Optional[AccuracyOptions] = None):
        if not prompt_path:
            raise RuntimeError("Prompt path is required. Provide a valid YAML config with prompts.")
        ps = PromptStore(prompt_path)
        if model is not None:
            ps.model = model
        if temperature is not None:
            ps.temperature = temperature
        llm_client = LLMClient(model=ps.model, temperature=ps.temperature) if use_llm else None
        cache = FileCache(cache_path) if cache_path else None

        core = _CoreKGBuilder(use_llm=use_llm, llm_client=llm_client, prompt_store=ps,
                              granularity=granularity, pack_size=pack_size,
                              cache=cache, min_chars_for_llm=min_chars_for_llm,
                              max_input_tokens=max_input_tokens, accuracy=accuracy)
        nodes, edges, warnings = core.build(doc)
        self._graph = _Graph(nodes, edges, warnings)

    def export_snapshot(self, path: Optional[str] = None, fmt: str = "json") -> Dict[str, Any]:
        return self._graph.export_snapshot(path, fmt)
    
    def import_snapshot(self, snap_or_path: Union[str, Dict[str, Any]]) -> "_Graph":
        """
        Load a saved snapshot (path or dict) into THIS instance, replacing self._graph.
        Returns the in-memory _Graph.
        """
        if isinstance(snap_or_path, str):
            with open(snap_or_path, "r", encoding="utf-8") as f:
                snap = json.load(f)
        elif isinstance(snap_or_path, dict):
            snap = snap_or_path
        else:
            raise TypeError("snap_or_path must be a filepath (str) or a snapshot dict")

        self._graph = _graph_from_snapshot_dict(snap)
        return self._graph

    @classmethod
    def from_snapshot(cls, snap_or_path: Union[str, Dict[str, Any]]) -> "KGbuilder":
        """
        Construct a KGbuilder backed only by a snapshot (no build/LLM run).
        Only graph-centric methods (to_json/export_snapshot/to_networkx) are available.
        """
        if isinstance(snap_or_path, str):
            with open(snap_or_path, "r", encoding="utf-8") as f:
                snap = json.load(f)
        elif isinstance(snap_or_path, dict):
            snap = snap_or_path
        else:
            raise TypeError("snap_or_path must be a filepath (str) or a snapshot dict")

        self = object.__new__(cls)     # bypass __init__
        self._graph = _graph_from_snapshot_dict(snap)
        return self

    def to_networkx(self):
        return self._graph.to_networkx()

    def to_json(self) -> Dict[str, Any]:
        return self._graph.to_json()
    
def _graph_from_snapshot_dict(snap: Dict[str, Any]) -> "_Graph":
    """
    Snapshot(dict) -> _Graph
    - snap: {"nodes":[{id,kind,label,attrs,type?}, ...],
             "edges":[{src,dst,etype,attrs}, ...],
             "meta": {"warnings":[], "conflicts":[]}}
    """
    nodes_raw = snap.get("nodes") or []
    edges_raw = snap.get("edges") or []
    meta      = snap.get("meta")  or {}

    nodes: List[Node] = []
    for n in nodes_raw:
        nodes.append(
            Node(
                id=n["id"],
                kind=n.get("kind", "unknown"),
                label=n.get("label") or n["id"],
                attrs=n.get("attrs") or {},
                type=n.get("type")    # may be None
            )
        )

    edges: List[Edge] = []
    for e in edges_raw:
        edges.append(
            Edge(
                src=e["src"],
                dst=e["dst"],
                etype=e.get("etype", "REFERS_TO"),
                attrs=e.get("attrs") or {},
            )
        )

    warnings = meta.get("warnings") or []
    conflicts = meta.get("conflicts") or []
    return _Graph(nodes, edges, warnings, conflicts)


# -------------- CLI --------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/gdpr.json")
    ap.add_argument("--output", type=str, default="data/gdpr_kg_snapshot.json")
    ap.add_argument("--use-llm", action="store_true")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--cache", type=str, default=None)
    ap.add_argument("--max-input-tokens", type=int, default=500)
    ap.add_argument("--pack-size", type=int, default=8)
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        doc = json.load(f)

    g = KGbuilder(doc, use_llm=args.use_llm, prompt_path=args.prompt,
                  cache_path=args.cache, max_input_tokens=args.max_input_tokens,
                  pack_size=args.pack_size)
    snap = g.export_snapshot(args.output)
    print(f"Saved snapshot to: {args.output} (nodes={snap['meta']['node_count']}, edges={snap['meta']['edge_count']}, warnings={snap['meta']['warning_count']})")
