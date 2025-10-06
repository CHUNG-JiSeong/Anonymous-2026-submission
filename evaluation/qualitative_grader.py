# other_baselines/qualitative_grader.py
# Sample-level LLM grader: ONE call per data sample.
# - Input: per_sample_results.jsonl
# - Output (JSONL): per-sample {"idx","id","num_articles","sample_score"}
# - Metrics JSON: {"num_samples","avg_score_sample_level",...}
# - Features: --drop-idx / --drop-idx-file / tqdm progress / no CSV
import os, sys, json, argparse, re
from dataclasses import dataclass
from typing import Any, Dict, List
from collections import defaultdict
from string import Template
from tqdm.auto import tqdm

# ---------------- Prompt (per-sample; returns ONLY {"sample_score": <number>}) ----------------
PROMPT_TEMPLATE = Template(r"""
# You are a strict, fast GDPR compliance grader.

# You receive a DATA SAMPLE that contains multiple Article-level judgments.
# Each entry has:
# - ARTICLE: the Article number+name,
# - GROUND_TRUTH_LABEL: whether this Article truly has a violation (NON_COMPLIANT) or not (COMPLIANT),
# - PREDICTION_LABEL: the model's verdict (COMPLIANT, NON_COMPLIANT, INSUFFICIENT, NOT_APPLICABLE),
# - WHY: the model's rationale (may be empty),
# - EVIDENCE_SNIPPETS: optional evidence (may be empty).

# Your task (do all reasoning internally, return only the final number):
# 1) For EACH entry, compute an entry SCORE using this rule:
#    - If agreement == "yes" and WHY is valid => 100
#    - If agreement == "yes" and WHY is invalid => 50
#    - If agreement == "yes_partially" and WHY is valid => 80
#    - If agreement == "yes_partially" and WHY is invalid => 40
#    - If agreement == "no" but it is a principled abstention (PREDICTION_LABEL in [INSUFFICIENT, NOT_APPLICABLE] AND GROUND_TRUTH_LABEL is NON_COMPLIANT AND WHY is persuasive) => 30
#    - Else => 0

#    Agreement guidance:
#    - "yes": PREDICTION_LABEL exactly matches GROUND_TRUTH_LABEL.
#    - "yes_partially": meaningful overlap (e.g., principled abstention with persuasive WHY).
#    - "no": otherwise.
#    WHY validity guidance:
#    - "valid": specific, evidence-grounded, and logically supports the decision/abstention.
#    - "invalid": generic/hand-wavy/contradictory.

# 2) Compute SAMPLE_SCORE = the arithmetic mean of all entry SCORES in this SAMPLE.
# 3) Output ONLY a compact JSON object with a single key "sample_score" whose value is a number.
#    Do NOT include any per-entry details. Do NOT echo the inputs.

# DATA SAMPLE (entries follow):
# $entries

# Return ONLY: {"sample_score": <number>}
You are an expert GDPR compliance grader with a generous, proximity-weighted rubric.

You will receive ONE DATA SAMPLE with multiple Article-level judgments.
Each entry includes:
- ARTICLE: number + name
- GROUND_TRUTH_LABEL: whether the Article truly has a violation (NON_COMPLIANT) or not (COMPLIANT)
- PREDICTION_LABEL: model’s verdict (COMPLIANT, NON_COMPLIANT, INSUFFICIENT, NOT_APPLICABLE)
- WHY: model’s rationale (may be empty)
- EVIDENCE_SNIPPETS: optional evidence (may be empty)

Goal
- Holistically grade THIS SAMPLE and return a single numeric score in [0, 100] called sample_score.
- Be generous: reward clear correctness with 100, and grant substantial partial credit when the prediction is close in substance even if the label is not exact.

How to evaluate (qualitative, no rigid point rules; err on the higher side when borderline)
1) Exact correctness (award 100 per entry)
   - PREDICTION_LABEL matches GROUND_TRUTH_LABEL.

2) Near-match (grant high partial credit per entry, typically 70–95)
   Consider as “near” when one or more holds:
   - The rationale (WHY) clearly captures the ground-truth concept (e.g., cites the actual violation theme) though the label differs.
   - INSUFFICIENT or NOT_APPLICABLE is used cautiously and persuasively in a situation that plausibly lacks decisive evidence or scope clarity (especially when GT is NON_COMPLIANT).
   - The evidence snippets show the right facts but the final label is a conservative miss.
   - The prediction is directionally correct (e.g., flags the right Article concern) but underspecified.

3) Partial / on-topic but weak (moderate partial credit, typically 50–69)
   - The rationale is somewhat relevant but thin, generic, or partially misaligned; still shows awareness of the correct area.

4) Minimal relevance (small partial credit, typically 25–49)
   - Vague or superficial connection to the correct concept; weak justification, but not entirely off-topic.

5) Off-topic or clearly wrong (0–24)
   - Little to no overlap with the ground truth; rationale is generic, contradictory, or unrelated.

Sampling & aggregation
- For EACH entry, assign a per-entry score guided by the above.
- Compute SAMPLE_SCORE = the arithmetic mean of all per-entry scores in this SAMPLE.
- Calibrate generously: if an entry sits between two bands, choose the higher band; favor awarding credit for concrete, on-topic evidence even when the final label is not exact.

Output requirements
- Produce ONLY a compact JSON object: {"sample_score": <number in [0,100]>}
- Do NOT include per-entry details. Do NOT echo the inputs.

DATA SAMPLE (entries follow):
$entries

Return ONLY: {"sample_score": <number>}
""".strip())

# ---------------- Data ----------------
@dataclass
class SampleResult:
    idx: int
    id: Any
    num_articles: int
    sample_score: float

# ---------------- Helpers ----------------
def parse_drop_idx(arg: str):
    if not arg:
        return set()
    vals = set()
    for tok in re.split(r"[,\s]+", arg.strip()):
        if not tok:
            continue
        try:
            vals.add(int(tok))
        except ValueError:
            pass
    return vals

def load_drop_idx_file(fp: str):
    if not fp or not os.path.exists(fp):
        return set()
    with open(fp, "r", encoding="utf-8") as f:
        return parse_drop_idx(f.read())

def _truncate(s: str, limit: int = 600) -> str:
    if s is None:
        return ""
    s = str(s)
    return (s[:limit] + " …") if len(s) > limit else s

def _entry_lines(item: dict) -> List[str]:
    """Build compact lines for all decisions in one sample, with GT label per article."""
    gt_set = set(item.get("gt_articles", []))
    lines = []
    for rec in item.get("decisions", []):
        art = rec.get("article")
        raw = rec.get("raw_article", "")
        gt_label = "NON_COMPLIANT" if art in gt_set else "COMPLIANT"
        pred = rec.get("verdict", "")
        why = _truncate(rec.get("why", ""), 600)
        ev = "; ".join(rec.get("evidence", [])[:6])
        ev = _truncate(ev, 400)
        # one simple, parseable line; no JSON to keep tokens small
        line = f"- ARTICLE: {art} ({raw}) | GROUND_TRUTH_LABEL: {gt_label} | PREDICTION_LABEL: {pred} | WHY: {why} | EVIDENCE_SNIPPETS: {ev}"
        lines.append(line)
    return lines

def build_sample_prompt(item: dict) -> str:
    lines = _entry_lines(item)
    entries_block = "\n".join(lines) if lines else "(no entries)"
    return PROMPT_TEMPLATE.substitute(entries=entries_block)

def try_openai_sample_score(prompt: str) -> float:
    """
    Call OpenAI Responses API once per SAMPLE.
    Returns float sample_score; falls back to 0.0 on any error.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return 0.0
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model="gpt-4.1",  # <- 환경에 맞는 모델 ID 사용
            instructions="Return ONLY a compact JSON object with the key 'sample_score' (a number).",
            input=prompt,
            # temperature=0.0,
            text={"format": {"type": "json_object"}}
        )
        content = getattr(resp, "output_text", None)
        if not content:
            raise RuntimeError("Empty response content")
        obj = json.loads(content)
        val = obj.get("sample_score", 0.0)
        try:
            return float(val)
        except Exception:
            return 0.0
    except Exception as e:
        sys.stderr.write(f"[warn] OpenAI responses call failed: {e}\n")
        return 0.0

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="LLM-only sample-level grader (one call per sample).")
    ap.add_argument("--input", required=True, help="per_sample_results.jsonl")
    ap.add_argument("--out-sample-jsonl", default="qual_grades_by_sample.jsonl", help="per-sample scores (JSONL)")
    ap.add_argument("--out-metrics", default="qual_grades_metrics.json", help="aggregate metrics JSON")
    ap.add_argument("--drop-idx", default="", help="Comma/space separated list of idx to exclude")
    ap.add_argument("--drop-idx-file", default="", help="Path to file listing idx to exclude")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    args = ap.parse_args()

    DROP_IDX = parse_drop_idx(args.drop_idx) | load_drop_idx_file(args.drop_idx_file)

    # -------- First pass: collect samples to grade (respecting drop-idx) --------
    samples: List[dict] = []
    with open(args.input, "r", encoding="utf-8") as fin:
        for line in fin:
            item = json.loads(line)
            try:
                item_idx = int(item.get("idx"))
            except Exception:
                item_idx = None
            if item_idx is not None and item_idx in DROP_IDX:
                continue
            # skip empty decision sets
            if not item.get("decisions"):
                continue
            samples.append(item)

    # -------- Grade per sample (ONE call each) --------
    per_sample_results: List[SampleResult] = []
    pbar = None if args.no_progress else tqdm(total=len(samples), desc="Grading samples", unit="sample")
    for item in samples:
        prompt = build_sample_prompt(item)
        sample_score = try_openai_sample_score(prompt)
        res = SampleResult(
            idx=item.get("idx"),
            id=item.get("id"),
            num_articles=len(item.get("decisions", [])),
            sample_score=sample_score
        )
        per_sample_results.append(res)
        if pbar: pbar.update(1)
    if pbar: pbar.close()

    # -------- Save per-sample JSONL --------
    with open(args.out_sample_jsonl, "w", encoding="utf-8") as fs:
        for r in per_sample_results:
            fs.write(json.dumps({
                "idx": r.idx,
                "id": r.id,
                "num_articles": r.num_articles,
                "sample_score": r.sample_score
            }, ensure_ascii=False) + "\n")

    # -------- Final average across samples --------
    if per_sample_results:
        final_avg = sum(r.sample_score for r in per_sample_results) / len(per_sample_results)
    else:
        final_avg = 0.0

    metrics = {
        "num_samples": len(per_sample_results),
        "avg_score_sample_level": final_avg,
        "dropped_idx": sorted(DROP_IDX),
        "input": args.input,
        "out_sample_jsonl": args.out_sample_jsonl,
    }
    with open(args.out_metrics, "w", encoding="utf-8") as fm:
        json.dump(metrics, fm, ensure_ascii=False, indent=2)

    print(f"Graded {len(per_sample_results)} samples (dropped idx: {sorted(DROP_IDX)})")
    print(f"Final average (sample-level): {final_avg:.2f}")
    print(f"Wrote: {args.out_sample_jsonl}, {args.out_metrics}")

if __name__ == "__main__":
    main()
