"""
1단계 — PDF에서 텍스트·표·그림을 추출하고, 텍스트를 토큰 단위로 다시 나눠 extracted.pkl 로 저장합니다.
쪽수에 비례해서 오래 걸립니다 (29쪽 약 2분). 실행: python main.py extract
"""
import os
import pickle
import re
import shutil
import subprocess
import time
from collections import Counter

from langchain_text_splitters import CharacterTextSplitter
from pypdf import PdfReader, PdfWriter
from unstructured.partition.pdf import partition_pdf

from src import config

os.environ.setdefault("UNSTRUCTURED_DISABLE_NOTICE", "1")  # 추출할 때마다 출력되는 Unstructured 유료 서비스 안내 문구를 끔


def check_environment():
    """pip 으로 설치되지 않는 Poppler·Tesseract(한국어 포함)를 확인합니다. 없으면 추출 도중에 실패하므로 먼저 멈춥니다."""
    missing = []
    if not shutil.which("pdftoppm"):
        missing.append("Poppler (pdftoppm)")
    if not shutil.which("tesseract"):
        missing.append("Tesseract")
    else:
        result = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True)
        if "kor" not in (result.stdout + result.stderr).split():
            missing.append("Tesseract 한국어 데이터 (kor.traineddata)")
    if missing:
        raise RuntimeError("설치가 필요합니다: " + ", ".join(missing) + " — 설치 후 Path 를 확인하고 터미널을 다시 여세요")


def find_pdf():
    """사용할 PDF 경로를 돌려줍니다. config.PDF_NAME 이 없으면 data/ 에 있는 PDF 1개를 고릅니다."""
    if config.PDF_NAME:
        return config.DATA_DIR / config.PDF_NAME
    pdf_files = sorted(config.DATA_DIR.glob("*.pdf"))
    if len(pdf_files) != 1:
        raise FileNotFoundError(
            f"data/ 에 PDF가 {len(pdf_files)}개 있습니다. 1개만 두거나 config.PDF_NAME 에 파일명을 적으세요."
        )
    return pdf_files[0]


def cut_pages(pdf_path, page_ranges):
    """
    쪽 범위만 잘라 새 PDF로 저장합니다.

    pdf_path   : 원본 PDF 경로
    page_ranges: [(시작, 끝), ...] PDF 뷰어 기준 쪽 번호, 양 끝 포함. None 이면 원본을 그대로 씀
    반환값     : (추출에 쓸 PDF 경로, 잘라낸 PDF의 각 쪽이 원본 몇 쪽인지 담은 리스트)
    """
    reader = PdfReader(pdf_path)
    if not page_ranges:
        return pdf_path, list(range(1, len(reader.pages) + 1))

    original_pages = []
    for start, end in page_ranges:
        if not 1 <= start <= end <= len(reader.pages):
            raise ValueError(f"쪽 범위 ({start}, {end}) 가 잘못됐습니다. 이 PDF는 1~{len(reader.pages)}쪽입니다.")
        original_pages.extend(range(start, end + 1))

    writer = PdfWriter()
    for page_no in original_pages:
        writer.add_page(reader.pages[page_no - 1])
    config.EXCERPT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_".join(f"{start}-{end}" for start, end in page_ranges)
    excerpt_path = config.EXCERPT_DIR / f"{pdf_path.stem}_p{tag}.pdf"  # *.pdf 라서 git 에는 올라가지 않음
    with open(excerpt_path, "wb") as f:
        writer.write(f)
    return excerpt_path, original_pages


def extract_pdf_elements(pdf_file, image_dir):
    """
    PDF 파일에서 이미지, 테이블, 텍스트 블록을 추출합니다.

    pdf_file : PDF 파일 경로
    image_dir: 추출된 이미지가 저장될 폴더
    """
    return partition_pdf(
        filename=str(pdf_file),
        strategy="hi_res",                # 정밀 추출 전략
        languages=["kor", "eng"],         # 한국어 교재 — OCR(Tesseract)에 한국어+영어 사용
        extract_images_in_pdf=True,       # 이미지 추출
        infer_table_structure=True,       # 표 구조 추론
        chunking_strategy="by_title",     # 타이틀 기준으로 텍스트 분할
        max_characters=4000,              # 최대 4000자로 블록 제한
        new_after_n_chars=3800,           # 3800자 이후 새 블록 생성
        combine_text_under_n_chars=2000,  # 2000자 이하는 결합
        extract_image_block_output_dir=str(image_dir),  # 이미지 저장 경로 (구 image_output_dir_path)
    )


def categorize_elements(raw_pdf_elements):
    """
    PDF에서 추출한 요소들을 테이블과 텍스트로 분류합니다.
    raw_pdf_elements: unstructured.documents.elements 리스트
    """
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))   # 테이블 요소
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))    # 텍스트 요소
    return texts, tables


def rename_figures(image_dir, original_pages):
    """
    그림 파일 이름의 쪽 번호를 원본 PDF 쪽 번호로 바꿉니다 (figure-3-5.jpg → figure-p047-5.jpg).
    잘라낸 PDF는 1쪽부터 다시 세기 때문에, 그대로 두면 근거 그림을 책에서 찾기 어렵습니다.
    """
    for f in image_dir.glob("figure-*"):
        match = re.fullmatch(r"figure-(\d+)-(\d+)", f.stem)
        if match:
            page, number = int(match.group(1)), match.group(2)
            f.rename(f.with_name(f"figure-p{original_pages[page - 1]:03d}-{number}{f.suffix}"))


def split_texts(texts):
    """텍스트 블록을 이어 붙인 뒤 토큰 기준으로 다시 자릅니다 (블록 경계에서 문맥이 끊기지 않도록 겹쳐 자름)."""
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    return text_splitter.split_text(" ".join(texts))


def main():
    check_environment()

    # 1. 사용할 PDF와 쪽 범위
    pdf_path = find_pdf()
    target_pdf, original_pages = cut_pages(pdf_path, config.PAGE_RANGES)
    print("PDF:", pdf_path.name, "| 쪽 범위:", config.PAGE_RANGES or "전체", f"| 추출할 쪽 {len(original_pages)}쪽")

    # 2. 이전 실행에서 남은 그림을 지우고 추출 (다른 PDF·쪽 범위의 그림이 섞이지 않도록)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for old_file in config.FIGURES_DIR.glob("figure-*"):
        old_file.unlink()

    start = time.time()
    raw_pdf_elements = extract_pdf_elements(target_pdf, config.FIGURES_DIR)
    texts, tables = categorize_elements(raw_pdf_elements)
    rename_figures(config.FIGURES_DIR, original_pages)

    print(f"추출 완료 — {time.time() - start:.1f}초 소요")
    for element_type, count in Counter(type(el).__name__ for el in raw_pdf_elements).most_common():
        print(f"  {element_type:22s}: {count}개")
    print("텍스트 블록:", len(texts), "| 표 블록:", len(tables), "| 그림:", len(list(config.FIGURES_DIR.glob("figure-*"))), "개")

    # 3. 텍스트 분할
    texts_2k_token = split_texts(texts)
    print("분할 후 텍스트 블록:", len(texts_2k_token))
    print(texts_2k_token[0][:300] if texts_2k_token else "(텍스트 없음)")  # 한글이 깨지지 않고 읽히는지 확인

    # 4. 저장 (2단계 summarize 에서 불러옴)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.EXTRACTED_PKL, "wb") as f:
        pickle.dump(
            {
                "texts": texts,
                "tables": tables,
                "texts_2k_token": texts_2k_token,
                "source": {"pdf": pdf_path.name, "page_ranges": config.PAGE_RANGES},  # 다음 단계에서 어떤 PDF인지 표시
            },
            f,
        )
    print("저장 완료:", config.EXTRACTED_PKL)


if __name__ == "__main__":
    main()
