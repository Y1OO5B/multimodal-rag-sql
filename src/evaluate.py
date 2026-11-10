"""
3단계 평가 — 평가 질문(config.EVAL_QUESTIONS)을 텍스트 전용 RAG와 멀티모달 RAG에 모두 던지고 과제 산출물을 만듭니다.
    Before/After : 정답이 그림에만 있는 질문 하나의 두 답변 (화면 출력을 캡처해서 docs/before_after.png)
    질의응답 5건 : 질문 / 답변 / 근거가 된 원본 종류 → docs/qa_table.md
    두 결과 비교 : 정답 키워드·정답 근거 검색으로 채점 (src/compare.py) → docs/comparison.md
    질문별 원자료 : docs/eval_results.json (python main.py compare 로 다시 채점할 때 사용)
실행: python main.py evaluate   (질문 하나만 확인: python main.py ask "질문")
"""
import json
import time

from src import compare, config
from src.chain import build_multimodal_chain, build_text_only_chain
from src.retriever import build_retriever, build_text_retriever, is_image, load_results

MAX_ANSWER_CHARS = 300  # 질의응답 표 한 칸에 넣을 답변 길이 (넘으면 … 으로 자름, 전체 답변은 eval_results.json)


def to_cell(text, limit=None):
    """마크다운 표 한 칸에 들어가게 줄바꿈을 없애고 표 구분자 | 는 이스케이프합니다."""
    text = " ".join(text.split()).replace("|", "\\|")
    return text if limit is None or len(text) <= limit else text[:limit] + "…"


def classify(docs, data):
    """검색된 원본을 이미지 / 표 / 텍스트로 나눕니다. 표는 1단계에서 추출한 표 목록에 있는지로 판단합니다."""
    b64_to_name = dict(zip(data["img_base64_list"], data["image_names"]))
    tables = set(data["tables"])
    evidence = []
    for doc in docs:
        content = str(getattr(doc, "page_content", doc))  # docstore 원본은 문자열, Document 면 본문만
        if is_image(content):
            evidence.append({"type": "이미지", "name": b64_to_name.get(content, "?")})
        elif content in tables:
            evidence.append({"type": "표", "text": content})
        else:
            evidence.append({"type": "텍스트", "text": content})
    return evidence


def format_evidence(evidence):
    """근거가 된 원본 종류를 "이미지 2 (figure-p058-20.jpg, …) · 표 1 · 텍스트 1" 형태로 만듭니다."""
    parts = []
    for kind in compare.SOURCE_TYPES:
        items = [e for e in evidence if e["type"] == kind]
        if items and kind == "이미지":
            parts.append(f"이미지 {len(items)} ({', '.join(e['name'] for e in items)})")
        elif items:
            parts.append(f"{kind} {len(items)}")
    return " · ".join(parts) or "없음"


def run_question(item, retriever, chain, text_only_chain, data):
    """질문 하나를 두 체인에 던지고 검색 근거·답변·소요 시간을 모읍니다."""
    evidence = classify(retriever.invoke(item["question"]), data)  # 멀티모달 체인이 받는 것과 같은 검색 결과

    start = time.time()
    after = chain.invoke(item["question"])
    after_seconds = time.time() - start

    start = time.time()
    before = text_only_chain.invoke(item["question"])
    before_seconds = time.time() - start

    return {
        "question": item["question"],
        "evidence": [  # 표·텍스트는 앞부분만 저장 (원문 전체를 저장소에 남기지 않도록)
            {"type": e["type"], "name": e["name"]} if e["type"] == "이미지"
            else {"type": e["type"], "preview": " ".join(e["text"].split())[:60]}
            for e in evidence
        ],
        "evidence_hit": compare.evidence_hit(item, evidence),
        "before": before,
        "after": after,
        "before_seconds": round(before_seconds, 1),
        "after_seconds": round(after_seconds, 1),
    }


def format_headline(result):
    """Before/After 캡처용 출력 — 같은 질문에 대한 두 답변."""
    line = "=" * 70
    return (
        f"{line}\n질문: {result['question']}\n{line}\n"
        f"[BEFORE] 텍스트 전용 RAG\n{result['before']}\n\n"
        f"[AFTER] 멀티모달 RAG (검색 근거: {format_evidence(result['evidence'])})\n{result['after']}\n{line}"
    )


def format_qa_table(results):
    """질의응답 5건 — 질문 / 답변(멀티모달) / 근거가 된 원본 종류."""
    lines = ["| # | 질문 | 답변 | 근거가 된 원본 종류 |", "|---|---|---|---|"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"| {i} | {to_cell(r['question'])} | {to_cell(r['after'], MAX_ANSWER_CHARS)} | {format_evidence(r['evidence'])} |"
        )
    return "\n".join(lines)


def run(data, retriever, chain, text_only_chain):
    """
    평가 질문을 모두 실행하고 산출물 파일을 저장합니다 (질문 수 x 2번 API 호출).
    반환값: {"results": 질문별 원자료, "headline": Before/After 출력, "qa_table": 마크다운, "comparison": 마크다운}
    """
    results = []
    for i, item in enumerate(config.EVAL_QUESTIONS, 1):
        results.append(run_question(item, retriever, chain, text_only_chain, data))
        print(f"{i}/{len(config.EVAL_QUESTIONS)} 완료 — 근거 {format_evidence(results[-1]['evidence'])}")
        time.sleep(1)  # 연속 호출로 인한 RateLimitError 예방

    config.DOCS_DIR.mkdir(exist_ok=True)
    with open(compare.RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    qa_table = format_qa_table(results)
    (config.DOCS_DIR / "qa_table.md").write_text(qa_table + "\n", encoding="utf-8")
    comparison = compare.compare(results)  # docs/comparison.md 도 여기서 저장

    return {
        "results": results,
        "headline": format_headline(results[config.BEFORE_AFTER_INDEX]),
        "qa_table": qa_table,
        "comparison": comparison,
    }


def build_all():
    """1·2단계 결과를 불러와 검색기와 두 체인을 만듭니다."""
    config.load_api_key()
    data = load_results()
    retriever = build_retriever(data)
    chain = build_multimodal_chain(retriever)
    text_only_chain = build_text_only_chain(build_text_retriever(data["texts_2k_token"]))
    return data, retriever, chain, text_only_chain


def ask(question):
    """질문 하나로 검색된 원본과 답변을 출력합니다. 정답 근거가 검색되는 질문인지 평가 전에 확인할 때 씁니다."""
    data, retriever, chain, _ = build_all()
    evidence = classify(retriever.invoke(question), data)
    print("\n질문:", question)
    for e in evidence:
        print(f"  [{e['type']}]", e["name"] if e["type"] == "이미지" else " ".join(e["text"].split())[:120])
    start = time.time()
    answer = chain.invoke(question)
    print(f"\n답변 ({time.time() - start:.1f}초)\n{answer}")


def main():
    data, retriever, chain, text_only_chain = build_all()
    report = run(data, retriever, chain, text_only_chain)
    print()
    print(report["headline"])
    print()
    print(report["qa_table"])
    print()
    print(report["comparison"])
    print("\n저장: docs/qa_table.md, docs/comparison.md, docs/eval_results.json")


if __name__ == "__main__":
    main()
