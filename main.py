"""
SQL 교재 멀티모달 RAG — 실행 진입점입니다. 프로젝트 루트에서 단계별로 실행합니다.

    python main.py extract          1단계: PDF → 텍스트·표·그림 추출 → data/processed/extracted.pkl
    python main.py summarize        2단계: 텍스트·표·그림 요약 (API 비용 발생) → data/processed/summaries.pkl
    python main.py ask "질문"        질문 하나로 검색 근거와 답변 확인 (평가 질문 고를 때)
    python main.py evaluate         3단계: Before/After, 질의응답 5건 표, 두 결과 비교 → docs/
    python main.py compare          저장된 평가 결과로 다시 채점만 (API 호출 없음)
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="SQL 교재 멀티모달 RAG")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("extract", help="1단계: PDF에서 텍스트·표·그림 추출")
    commands.add_parser("summarize", help="2단계: 텍스트·표·그림 요약 (API 비용 발생)")
    ask_parser = commands.add_parser("ask", help="질문 하나로 검색 근거와 답변 확인")
    ask_parser.add_argument("question", help="물어볼 질문 (따옴표로 감싸기)")
    commands.add_parser("evaluate", help="3단계: Before/After, 질의응답 5건 표, 두 결과 비교")
    commands.add_parser("compare", help="저장된 평가 결과로 다시 채점만 (API 호출 없음)")
    args = parser.parse_args()

    # 단계별 모듈은 필요할 때만 불러옴 (extract 의 Unstructured 처럼 무거운 라이브러리를 다른 단계에서 읽지 않도록)
    if args.command == "extract":
        from src import extract
        extract.main()
    elif args.command == "summarize":
        from src import summarize
        summarize.main()
    elif args.command == "ask":
        from src import evaluate
        evaluate.ask(args.question)
    elif args.command == "evaluate":
        from src import evaluate
        evaluate.main()
    elif args.command == "compare":
        from src import compare
        compare.main()


if __name__ == "__main__":
    main()
