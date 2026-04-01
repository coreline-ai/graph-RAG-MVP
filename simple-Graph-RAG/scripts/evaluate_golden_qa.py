"""Golden Q&A Evaluation Script — GAS + AAR automated regression testing.

Usage:
    EMBEDDING_DEVICE=mps python scripts/evaluate_golden_qa.py
    EMBEDDING_DEVICE=mps python scripts/evaluate_golden_qa.py --tag aar
    EMBEDDING_DEVICE=mps python scripts/evaluate_golden_qa.py --tag gas --model claude-sonnet-4-6
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_QA_PATH = PROJECT_ROOT / "data" / "golden_qa.json"
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_TIMEOUT = 120


def load_golden_qa(tag_filter: str | None = None) -> list[dict]:
    with open(GOLDEN_QA_PATH) as f:
        items = json.load(f)
    if tag_filter:
        items = [item for item in items if tag_filter in item.get("tag", [])]
    return items


def evaluate_gas(item: dict, response: dict) -> dict:
    """Evaluate Grounded Answer Score: do expected keywords appear in the answer?"""
    answer = response.get("answer", "")
    sources = response.get("sources", [])
    expect_kw = item.get("expect_keywords", [])
    expect_min_src = item.get("expect_min_sources", 1)
    expect_mode = item.get("expect_mode", "llm")

    kw_found = []
    kw_missing = []
    for kw in expect_kw:
        if re.search(re.escape(kw), answer, re.IGNORECASE):
            kw_found.append(kw)
        else:
            kw_missing.append(kw)

    mode_ok = expect_mode == "any" or response.get("answer_mode") == expect_mode
    src_ok = len(sources) >= expect_min_src
    kw_ok = len(kw_missing) == 0

    passed = mode_ok and src_ok and kw_ok

    return {
        "type": "gas",
        "passed": passed,
        "keyword_found": kw_found,
        "keyword_missing": kw_missing,
        "sources_count": len(sources),
        "sources_ok": src_ok,
        "mode": response.get("answer_mode"),
        "mode_ok": mode_ok,
    }


def evaluate_aar(item: dict, response: dict) -> dict:
    """Evaluate Abstain Accuracy Rate: does the answer indicate lack of evidence?"""
    answer = response.get("answer", "")
    abstain_kw = item.get("expect_abstain_keywords", [])

    abstain_matched = []
    for pattern in abstain_kw:
        if re.search(pattern, answer):
            abstain_matched.append(pattern)

    abstained = len(abstain_matched) > 0

    # Check for hallucination: answer confidently states facts without abstaining
    hallucination_indicators = ["확인되었습니다", "총 \\d+건", "아래와 같습니다"]
    hallucinated = False
    if not abstained:
        for pattern in hallucination_indicators:
            if re.search(pattern, answer):
                hallucinated = True
                break

    passed = abstained and not hallucinated

    return {
        "type": "aar",
        "passed": passed,
        "abstained": abstained,
        "hallucinated": hallucinated,
        "abstain_patterns_matched": abstain_matched,
        "answer_preview": answer[:150],
    }


async def run_query(client: httpx.AsyncClient, question: str, model: str, base_url: str) -> dict:
    r = await client.post(
        f"{base_url}/query",
        json={"question": question, "model": model, "top_k": 5},
    )
    r.raise_for_status()
    return r.json()


async def run_evaluation(
    items: list[dict],
    model: str,
    base_url: str,
    timeout: int,
) -> list[dict]:
    results = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for item in items:
            qid = item["id"]
            question = item["question"]
            tags = item.get("tag", [])
            is_aar = "aar" in tags

            start = time.time()
            try:
                response = await run_query(client, question, model, base_url)
                elapsed = time.time() - start

                if is_aar:
                    eval_result = evaluate_aar(item, response)
                else:
                    eval_result = evaluate_gas(item, response)

                result = {
                    "id": qid,
                    "question": question,
                    "tags": tags,
                    "elapsed_s": round(elapsed, 1),
                    **eval_result,
                }
            except httpx.ReadTimeout:
                result = {
                    "id": qid,
                    "question": question,
                    "tags": tags,
                    "elapsed_s": round(time.time() - start, 1),
                    "type": "aar" if is_aar else "gas",
                    "passed": False,
                    "error": "TIMEOUT",
                }
            except Exception as exc:
                result = {
                    "id": qid,
                    "question": question,
                    "tags": tags,
                    "elapsed_s": round(time.time() - start, 1),
                    "type": "aar" if is_aar else "gas",
                    "passed": False,
                    "error": str(exc)[:100],
                }

            status = "PASS" if result["passed"] else "FAIL"
            err = result.get("error", "")
            suffix = f" ({err})" if err else ""
            print(f"  [{status}] {qid:8s} {result['elapsed_s']:>6.1f}s | {question[:45]}{suffix}", flush=True)
            results.append(result)

    return results


def print_report(results: list[dict], model: str) -> None:
    gas_results = [r for r in results if r["type"] == "gas"]
    aar_results = [r for r in results if r["type"] == "aar"]

    gas_pass = sum(1 for r in gas_results if r["passed"])
    aar_pass = sum(1 for r in aar_results if r["passed"])
    total_pass = sum(1 for r in results if r["passed"])
    total = len(results)

    print()
    print("=" * 100)
    print(f"  Golden Q&A Evaluation Report — Model: {model}")
    print("=" * 100)
    print()

    # Summary
    print(f"  Total: {total} | Pass: {total_pass} | Fail: {total - total_pass} | Rate: {total_pass/total*100:.0f}%")
    if gas_results:
        print(f"  GAS:   {len(gas_results)} | Pass: {gas_pass} | Fail: {len(gas_results)-gas_pass} | Rate: {gas_pass/len(gas_results)*100:.0f}%")
    if aar_results:
        print(f"  AAR:   {len(aar_results)} | Pass: {aar_pass} | Fail: {len(aar_results)-aar_pass} | Rate: {aar_pass/len(aar_results)*100:.0f}%")
    print()

    # Detail table
    print(f"  {'ID':8s} | {'Type':4s} | {'Status':6s} | {'Time':>6s} | {'Details':40s} | Question")
    print("-" * 100)

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        flag = "\u2705" if r["passed"] else "\u274c"
        elapsed = f"{r['elapsed_s']:.1f}s"

        if r.get("error"):
            details = f"ERROR: {r['error'][:35]}"
        elif r["type"] == "aar":
            abstained = "\u2705" if r.get("abstained") else "\u274c"
            halluc = "\u274c halluc" if r.get("hallucinated") else ""
            details = f"abstain={abstained} {halluc}".strip()
        else:
            found = r.get("keyword_found", [])
            missing = r.get("keyword_missing", [])
            src = r.get("sources_count", 0)
            details = f"kw={len(found)}/{len(found)+len(missing)} src={src}"
            if missing:
                details += f" miss=[{','.join(missing[:3])}]"

        print(f"  {r['id']:8s} | {r['type']:4s} | {flag} {status:4s} | {elapsed:>6s} | {details:40s} | {r['question'][:35]}")

    print("=" * 100)

    # Failed details
    failed = [r for r in results if not r["passed"]]
    if failed:
        print()
        print("  FAILED DETAILS:")
        for r in failed:
            print(f"    {r['id']}: {r['question']}")
            if r.get("error"):
                print(f"      error: {r['error']}")
            elif r["type"] == "aar":
                print(f"      abstained={r.get('abstained')} hallucinated={r.get('hallucinated')}")
                print(f"      answer: {r.get('answer_preview', '')[:120]}")
            else:
                print(f"      missing keywords: {r.get('keyword_missing', [])}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Golden Q&A evaluation for GAS/AAR")
    parser.add_argument("--tag", help="Filter by tag (gas, aar, entity, assignee, etc.)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--json-output", help="Save results to JSON file")
    args = parser.parse_args()

    items = load_golden_qa(args.tag)
    if not items:
        print(f"No test cases found (tag={args.tag})")
        return 1

    print(f"Running {len(items)} golden Q&A tests (model={args.model}, tag={args.tag or 'all'})")
    print()

    results = asyncio.run(run_evaluation(items, args.model, args.base_url, args.timeout))
    print_report(results, args.model)

    if args.json_output:
        with open(args.json_output, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  Results saved to {args.json_output}")

    total_pass = sum(1 for r in results if r["passed"])
    return 0 if total_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
