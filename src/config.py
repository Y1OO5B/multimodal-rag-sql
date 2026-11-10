"""
설정값을 한 곳에 모은 모듈입니다.
경로, 사용할 PDF와 쪽 범위, 모델 이름, 평가 질문은 이 파일에서만 바꾸면 나머지 모듈에 그대로 반영됩니다.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 1. 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 이 파일(src/config.py)의 두 단계 위 = 프로젝트 루트
DATA_DIR = PROJECT_ROOT / "data"                       # PDF를 넣는 폴더
FIGURES_DIR = DATA_DIR / "figures"                     # 추출한 그림
EXCERPT_DIR = DATA_DIR / "excerpt"                     # 쪽 범위만 잘라낸 PDF
PROCESSED_DIR = DATA_DIR / "processed"                 # 단계 사이를 잇는 중간 결과(pkl)
EXTRACTED_PKL = PROCESSED_DIR / "extracted.pkl"        # 1단계 결과
SUMMARIES_PKL = PROCESSED_DIR / "summaries.pkl"        # 2단계 결과
DOCS_DIR = PROJECT_ROOT / "docs"                       # 과제 산출물 (Before/After, 질의응답 표)

# 2. 사용할 PDF
PDF_NAME = None  # None 이면 data/ 에 있는 PDF 1개를 자동으로 사용

# 2-1. 사용할 구간 — 445쪽 중 29쪽 (과제 기준 10~30쪽). PDF 뷰어 기준 쪽 번호, 양 끝 포함 (책에 인쇄된 쪽 번호 + 3)
    # 그림과 표가 함께 있도록 그림이 많은 구간과 표가 있는 구간을 하나씩 고름
SECTIONS = [
    ("관계·식별자", 44, 58),              # 1과목 1장 4절 관계, 5절 식별자 앞부분 — ERD 표기법(IE·Barker), 식별자 예시 그림
    ("계층형 질의와 셀프 조인", 324, 337),  # 2과목 2장 3절 — 계층 구조 그림, 가상 칼럼 표, 실행 결과 표
]
PAGE_RANGES = [(start, end) for _, start, end in SECTIONS]  # extract.py 가 쓰는 형식. None 으로 바꾸면 책 전체

# 3. 모델과 처리 설정
LLM_MODEL = "gpt-4o-mini"                   # 텍스트·표·그림 요약 (비전 지원, 호출 수가 많아 저렴한 모델)
ANSWER_MODEL = "gpt-4o-mini"                # 두 체인의 답변 모델 (같은 모델이라 비교 조건이 같음). 사용한 API 키는 gpt-4o 권한이 없음(403)
EMBEDDING_MODEL = "text-embedding-3-small"  # 요약문 임베딩
CHUNK_SIZE = 2000                           # 텍스트 분할 단위 (토큰)
CHUNK_OVERLAP = 200                         # 블록 경계에서 문맥이 끊기지 않도록 겹치는 양 (토큰)
MIN_SIDE_PX = 100                           # 가로·세로 중 짧은 쪽이 이보다 작은 그림은 아이콘으로 보고 요약에서 제외
MAX_CONCURRENCY = 5                         # 요약 API 동시 요청 수 (크게 올리면 RateLimitError)
RETRIEVER_K = 4                             # 질문마다 가져올 원본 수

# 4. 평가 질문 5개 — 정답 근거의 원본 종류가 섞이도록 이미지 2 · 표 2 · 텍스트 1
    # source: 정답 근거의 원본 종류 (이미지 / 표 / 텍스트)
    # page: 정답 근거가 있는 PDF 쪽 — 구간을 나누고, 이미지는 검색된 그림 파일 이름(figure-p058-…)과 맞춰 근거 검색 적중을 판단
    # match: (표·텍스트) 정답 근거 원본에 들어 있는 문구 — 검색된 원본에 이 문구가 있으면 근거 검색 적중 (띄어쓰기 무시)
    # as_image: (표) PDF에 그림으로 들어 있는 표 — 표(OCR)와 그림 파일 두 가지로 추출되므로 그 쪽 그림이 검색돼도 적중
    # keywords: 답변에 모두 들어 있으면 정답 (띄어쓰기 무시)
    # 이미지 질문 2개는 정답이 PDF 텍스트 층과 추출 텍스트(OCR 포함)에 없고 그림에만 있음을 확인함
EVAL_QUESTIONS = [
    {
        "question": "식별자/비식별자 관계 조정 그림에 따르면, 엔터티 사이의 관계 유형은 무엇에 의해 결정되나요?",
        "keywords": ["업무", "특징", "주식별자", "SQL"],  # 업무특징, 자식엔터티의 주식별자구성, SQL 전략 ([그림 Ⅰ-1-46])
        "source": "이미지",
        "page": 58,
    },
    {
        "question": "식별자의 분류 그림에서 교육이력 엔터티의 주식별자는 어떤 속성들로 구성되고, 어떤 식별자로 분류되나요?",
        "keywords": ["수강일자", "복합식별자"],  # 사번(FK) + 수강일자, 주식별자·복합식별자·본질식별자 ([그림 Ⅰ-1-42])
        "source": "이미지",
        "page": 54,
    },
    {
        "question": "SYS_CONNECT_BY_PATH 함수 예제의 실행 결과에서 사원 D의 경로는 무엇인가요?",
        "keywords": ["/A/C/D"],  # 루트사원 A, 경로 /A/C/D
        "source": "표",
        "page": 328,
        "match": "/A/C/D",
    },
    {
        "question": "계층형 질의에서 CONNECT_BY_ROOT는 무엇을 표시하고, 어떤 종류의 연산자인가요?",
        "keywords": ["루트", "단항"],  # 현재 전개할 데이터의 루트 데이터를 표시, 단항 연산자 ([표 Ⅱ-2-3])
        "source": "표",
        "page": 328,
        "match": "단항 연산자",
        "as_image": True,  # [표 Ⅱ-2-3]은 PDF 텍스트 층에 없는 그림 표 — 표 요소(OCR)와 figure-p328-… 그림으로 함께 추출됨
    },
    {
        "question": "계층형 질의에서 START WITH절과 CONNECT BY절은 각각 무엇을 지정하나요?",
        "keywords": ["시작", "루트", "자식"],  # START WITH: 전개 시작 위치(루트 데이터), CONNECT BY: 다음에 전개될 자식 데이터
        "source": "텍스트",
        "page": 325,
        "match": "시작 위치",
    },
]
BEFORE_AFTER_INDEX = 0  # Before/After 캡처에 쓸 질문 (EVAL_QUESTIONS 순번) — 정답이 그림에만 있는 질문


def load_api_key():
    """프로젝트 루트의 .env 에서 OPENAI_API_KEY 를 읽고, 없거나 비어 있으면 멈춥니다."""
    load_dotenv(PROJECT_ROOT / ".env", override=True)  # .env 수정 후 다시 실행해도 새 키로 교체
    try:
        api_key = os.environ["OPENAI_API_KEY"]
        if not api_key.strip():
            raise ValueError("OPENAI_API_KEY 값이 비어 있습니다. .env 파일을 확인해 주세요")
    except KeyError:
        raise ValueError("OPENAI_API_KEY 키가 없습니다. .env 파일을 확인해 주세요")
    print("API 키 로드 완료")  # 키는 일부도 출력하지 않음 (노트북 출력·캡처에 남지 않도록)
    return api_key
