"""
두 결과를 규칙으로 비교합니다 — 텍스트 전용 RAG(Before)와 멀티모달 RAG(After).
    정답 판정 : 질문마다 정한 정답 키워드가 답변에 모두 들어 있는지 (띄어쓰기 무시). LLM 채점이 아니라 다시 돌려도 같은 결과
    근거 검색 : 정답 근거가 검색됐는지 — 이미지는 정답 그림의 쪽 번호(figure-p058-…), 표·텍스트는 정답 문구가 든 원본 (그림으로 된 표는 그 쪽 그림도 인정)
    요약     : 근거 종류(이미지 / 표 / 텍스트)별, 구간(config.SECTIONS)별로 두 RAG의 정답 수를 셉니다
python main.py evaluate 가 끝에 자동으로 실행합니다. 키워드만 바꿔 다시 채점: python main.py compare (API 호출 없음)
"""
import json
import re

from src import config

RESULTS_JSON = config.DOCS_DIR / "eval_results.json"  # evaluate 가 저장한 질문별 두 답변과 검색 근거
SOURCE_TYPES = ["이미지", "표", "텍스트"]
REFUSAL_PHRASES = [  # 체인 지시문("자료에 없는 내용은 모른다고 답하세요")에 따라 모른다고 답할 때 나오는 표현
    "모른", "모릅", "모르겠", "자료에 없", "알 수 없", "확인할 수 없", "찾을 수 없",
    "제공되지 않", "나와 있지 않", "언급되어 있지 않", "포함되어 있지 않",
]


def normalize(text):
    """띄어쓰기·줄바꿈을 없애고 소문자로 바꿉니다 ("관할 부서"와 "관할부서"를 같게 보기 위해)."""
    return re.sub(r"\s+", "", text).lower()


def grade(answer, keywords):
    """
    답변에 들어 있는 정답 키워드 수로 판정합니다.
    정답: 키워드 전부 / 부분: 일부 / 모름: 하나도 없고 모른다고 답함 / 오답: 하나도 없음
    """
    text = normalize(answer)
    found = [k for k in keywords if normalize(k) in text]
    if len(found) == len(keywords):
        verdict = "정답"
    elif found:
        verdict = "부분"
    elif any(normalize(p) in text for p in REFUSAL_PHRASES):
        verdict = "모름"
    else:
        verdict = "오답"
    return {"verdict": verdict, "found": len(found), "total": len(keywords)}


def evidence_hit(item, evidence):
    """
    정답 근거가 검색됐는지 판단합니다.
    item    : config.EVAL_QUESTIONS 의 질문 하나 (source, page, match, as_image)
    evidence: 검색된 원본 목록 [{"type": "이미지", "name": ...} 또는 {"type": "표"/"텍스트", "text": ...}]
    """
    image_hit = any(e["type"] == "이미지" and f"p{item['page']:03d}" in e["name"] for e in evidence)
    if item["source"] == "이미지":
        return image_hit
    text_hit = any(e["type"] != "이미지" and normalize(item["match"]) in normalize(e["text"]) for e in evidence)
    return text_hit or (item.get("as_image", False) and image_hit)  # 그림으로 된 표는 그 쪽 그림이 검색돼도 적중


def section_of(page):
    """쪽 번호가 속한 구간 이름을 돌려줍니다."""
    for name, start, end in config.SECTIONS:
        if start <= page <= end:
            return name
    return "기타"


def score(results):
    """evaluate 결과에 config.EVAL_QUESTIONS 의 정답 정보를 붙여 채점합니다. 질문 문장으로 짝을 맞춥니다."""
    answers = {r["question"]: r for r in results}
    rows = []
    for i, item in enumerate(config.EVAL_QUESTIONS, 1):
        r = answers.get(item["question"])
        if r is None:  # 질문 문장을 바꿨으면 python main.py evaluate 를 다시 돌려야 함
            print(f"채점 건너뜀 — 평가 결과에 없는 질문: {item['question']}")
            continue
        rows.append({
            "no": i,
            "source": item["source"],
            "page": item["page"],
            "keywords": item["keywords"],
            "section": section_of(item["page"]),
            "hit": r["evidence_hit"],
            "before": grade(r["before"], item["keywords"]),
            "after": grade(r["after"], item["keywords"]),
        })
    return rows


def summarize_group(rows):
    """묶음마다 근거 검색 적중, 두 RAG의 정답 수와 키워드 포함 수를 셉니다."""
    return {
        "n": len(rows),
        "hit": sum(r["hit"] for r in rows),
        "before": sum(r["before"]["verdict"] == "정답" for r in rows),
        "after": sum(r["after"]["verdict"] == "정답" for r in rows),
        "before_kw": sum(r["before"]["found"] for r in rows),
        "after_kw": sum(r["after"]["found"] for r in rows),
        "total_kw": sum(r["before"]["total"] for r in rows),
    }


def summary_table(title, groups):
    """묶음별 요약을 마크다운 표로 만듭니다."""
    lines = [
        f"| {title} | 문항 | 근거 검색 | Before 정답 (키워드) | After 정답 (키워드) |",
        "|---|---|---|---|---|",
    ]
    for name, g in groups.items():
        if g["n"]:
            lines.append(
                f"| {name} | {g['n']} | {g['hit']}/{g['n']} "
                f"| {g['before']}/{g['n']} ({g['before_kw']}/{g['total_kw']}) | {g['after']}/{g['n']} ({g['after_kw']}/{g['total_kw']}) |"
            )
    return lines


def interpret(rows, by_type, total):
    """숫자로부터 해석 문장을 만듭니다 (규칙 기반이라 숫자에 없는 추측은 넣지 않음)."""
    lines = [
        f"- 전체 {total['n']}문항 중 텍스트 전용 RAG는 {total['before']}문항, 멀티모달 RAG는 {total['after']}문항을 맞혔습니다 "
        f"(차이 {total['after'] - total['before']:+d}, 정답 키워드 포함 {total['before_kw']}/{total['total_kw']} → {total['after_kw']}/{total['total_kw']})."
    ]

    for kind in SOURCE_TYPES:
        g = by_type.get(kind)
        if g and g["n"]:
            lines.append(f"- 근거가 {kind}인 문항: 텍스트 전용 {g['before']}/{g['n']}, 멀티모달 {g['after']}/{g['n']}.")

    refused = [r["no"] for r in rows if r["before"]["verdict"] == "모름"]
    if refused:
        lines.append(f"- 텍스트 전용 RAG가 '모른다'고 답한 문항: {refused} — 텍스트 검색으로는 근거를 찾지 못했습니다.")

    missed = [r["no"] for r in rows if not r["hit"]]
    if missed:
        lines.append(f"- 정답 근거를 검색하지 못한 문항: {missed} — 질문 표현이나 요약 프롬프트를 점검해야 합니다.")
    else:
        lines.append(f"- 모든 문항에서 정답 근거가 검색됐습니다 ({total['hit']}/{total['n']}).")

    hit_but_wrong = [r["no"] for r in rows if r["hit"] and r["after"]["verdict"] != "정답"]
    if hit_but_wrong:
        lines.append(
            f"- 정답 근거를 찾고도 멀티모달 답이 정답이 아닌 문항: {hit_but_wrong} — 원본 판독 문제이거나 "
            "답이 정답 키워드와 다른 표현을 썼을 수 있습니다. docs/eval_results.json 의 답변을 직접 확인하세요."
        )
    return lines


def build_report(rows):
    """질문별 채점표, 근거 종류별·구간별 요약표, 해석을 마크다운으로 만듭니다."""
    def cell(g):
        return f"{g['verdict']} ({g['found']}/{g['total']})"

    lines = [
        "#### 질문별 채점 (정답 키워드 기준, 번호는 질의응답 표와 같음)",
        "",
        "| # | 구간 | 정답 근거 | 정답 키워드 | 근거 검색 | Before 텍스트 전용 | After 멀티모달 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['no']} | {r['section']} | {r['source']} ({r['page']}쪽) | {', '.join(r['keywords'])} "
            f"| {'적중' if r['hit'] else '빗나감'} | {cell(r['before'])} | {cell(r['after'])} |"
        )

    by_type = {kind: summarize_group([r for r in rows if r["source"] == kind]) for kind in SOURCE_TYPES}
    by_section = {
        f"{name} ({start}~{end}쪽)": summarize_group([r for r in rows if r["section"] == name])
        for name, start, end in config.SECTIONS
    }
    total = summarize_group(rows)

    lines += ["", "#### 근거 종류별 요약", "", *summary_table("근거 종류", {**by_type, "전체": total})]
    lines += ["", "#### 구간별 요약", "", *summary_table("구간", {**by_section, "전체": total})]
    lines += ["", "#### 해석 (규칙으로 자동 생성)", "", *interpret(rows, by_type, total)]
    return "\n".join(lines)


def compare(results):
    """채점하고 docs/comparison.md 로 저장한 뒤 보고서 문자열을 돌려줍니다."""
    report = build_report(score(results))
    config.DOCS_DIR.mkdir(exist_ok=True)
    (config.DOCS_DIR / "comparison.md").write_text(report + "\n", encoding="utf-8")
    return report


def main():
    """저장된 평가 결과(docs/eval_results.json)로 다시 채점합니다. API를 부르지 않습니다."""
    results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))  # FileNotFoundError 나면 python main.py evaluate 먼저
    print(compare(results))
    print("\n저장:", config.DOCS_DIR / "comparison.md")


if __name__ == "__main__":
    main()
