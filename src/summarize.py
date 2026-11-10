"""
2단계 — 텍스트·표·그림을 요약해 summaries.pkl 로 저장합니다. OpenAI API 비용이 발생합니다.
검색은 요약으로 하고 답변은 원본으로 하므로, 요약에는 질문과 매칭될 단어(엔터티·컬럼 이름 등)가 들어가야 합니다.
실행: python main.py summarize
"""
import base64
import hashlib
import pickle
import re
import time

import tiktoken
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from PIL import Image

from src import config

# 1. 요약 프롬프트
TEXT_SUMMARY_PROMPT = """당신은 표와 텍스트를 요약하여 검색에 활용할 수 있도록 돕는 도우미입니다.
이 요약본들은 임베딩되어 원본 텍스트나 표 요소를 검색하는 데 사용될 것입니다.
주어진 표나 텍스트의 내용을 검색에 최적화된 간결한 요약으로 작성해주세요.
요약할 표 또는 텍스트: {element}"""

# 1-1. 표 요약 프롬프트 — 표는 값 자체가 정보라서 열 이름과 값을 지우지 않도록 따로 지시 (위 프롬프트로는 값이 빠져, 값을 묻는 질문에 표가 검색되지 않음)
TABLE_SUMMARY_PROMPT = """당신은 표를 요약하여 검색에 활용할 수 있도록 돕는 도우미입니다.
이 요약본들은 임베딩되어 원본 표를 검색하는 데 사용될 것입니다.
표가 무엇을 보여주는지 한 문장으로 쓰고, 이어서 열 이름과 각 행의 값을 빠짐없이 그대로 적어주세요.
OCR로 띄어쓰기가 깨진 글자는 바로잡되, 표에 없는 값은 지어내지 마세요.
요약할 표: {element}"""

# 1-2. 그림 요약 프롬프트 — SQL 학습자료에 맞게 보강 (그림 속 이름·숫자·문장이 요약에 들어가야 질문과 매칭됨)
IMAGE_SUMMARY_PROMPT = """You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
The images come from a Korean SQL study guide: ERDs in IE / Barker notation, identifier and relationship diagrams, hierarchical query trees, organization charts, SQL syntax boxes and result tables.
Provide a concise summary of the image that is well optimized for retrieval.
Include the figure caption, entity / attribute / relationship names, notation symbols, numbers, table values and any sentences written in the image.
The summary should be written in Korean (Hangul)."""


def generate_text_summaries(texts, tables, summarize_texts=False):
    """
    텍스트 및 표 데이터를 요약하여 검색에 활용할 요약본을 생성합니다.

    texts          : 텍스트 리스트
    tables         : 표 리스트
    summarize_texts: 텍스트 요약을 활성화할지 여부 (False 면 원본을 그대로 요약 자리에 씀)
    """
    model = ChatOpenAI(temperature=0, model=config.LLM_MODEL)

    def summarize_chain(template):
        """입력 포장 + 프롬프트 + LLM + 출력 파서"""
        return {"element": lambda x: x} | ChatPromptTemplate.from_template(template) | model | StrOutputParser()

    text_summaries = []
    table_summaries = []
    if texts and summarize_texts:
        text_summaries = summarize_chain(TEXT_SUMMARY_PROMPT).batch(texts, {"max_concurrency": config.MAX_CONCURRENCY})
    elif texts:
        text_summaries = texts
    if tables:
        table_summaries = summarize_chain(TABLE_SUMMARY_PROMPT).batch(tables, {"max_concurrency": config.MAX_CONCURRENCY})
    return text_summaries, table_summaries


def encode_image(image_path):
    """이미지를 base64 문자열로 변환"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def select_images(figures_dir):
    """
    요약할 그림만 고릅니다 — 그림 수만큼 API 비용이 들기 때문.
    쪽 번호 순으로 정렬하고, 작은 아이콘과 똑같은 그림(반복되는 로고 등)은 뺍니다.
    """
    all_images = sorted(
        (f for f in figures_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")),
        key=lambda f: [int(n) for n in re.findall(r"\d+", f.stem)],
    )
    image_files = []
    seen_hashes = set()
    skipped_small = skipped_dup = 0
    for f in all_images:
        with Image.open(f) as im:
            if min(im.size) < config.MIN_SIDE_PX:
                skipped_small += 1
                continue
        digest = hashlib.md5(f.read_bytes()).hexdigest()
        if digest in seen_hashes:
            skipped_dup += 1
            continue
        seen_hashes.add(digest)
        image_files.append(f)

    print(f"추출 그림 {len(all_images)}개 → 요약 대상 {len(image_files)}개 (작은 그림 {skipped_small}개, 중복 {skipped_dup}개 제외)")
    return image_files


def image_summarize(img_base64, prompt):
    """이미지 요약 생성 (교재의 gpt-4-vision-preview는 지원 종료 → 비전 기능이 있는 gpt-4o-mini 사용)"""
    chat = ChatOpenAI(temperature=0, model=config.LLM_MODEL, max_tokens=1024)  # 그림 속 글자를 받아 적는 작업이라 0 (다시 실행해도 요약이 덜 바뀜)
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},  # 'data:image/jpeg;base64,' 접두사 필수
                    },
                ]
            )
        ]
    )
    return msg.content


def generate_img_summaries(image_files, prompt):
    """
    이미지의 요약과 Base64 인코딩 문자열을 생성합니다.
    image_files: 요약할 이미지 파일 목록 (select_images 결과)
    """
    img_base64_list = [encode_image(f) for f in image_files]
    summarize_one = RunnableLambda(lambda b64: image_summarize(b64, prompt))
    image_summaries = summarize_one.batch(img_base64_list, {"max_concurrency": config.MAX_CONCURRENCY})  # 동시에 여러 장 요청, 결과 순서는 그대로
    return img_base64_list, image_summaries


def main():
    config.load_api_key()

    # 1. 1단계 결과 불러오기
    with open(config.EXTRACTED_PKL, "rb") as f:  # FileNotFoundError 나면 python main.py extract 먼저
        data = pickle.load(f)
    texts_2k_token, tables = data["texts_2k_token"], data["tables"]
    print("원본 PDF:", data["source"]["pdf"], "| 쪽 범위:", data["source"]["page_ranges"] or "전체")

    # 2. 텍스트·표 요약 (API 비용 발생)
    start = time.time()
    text_summaries, table_summaries = generate_text_summaries(texts_2k_token, tables, summarize_texts=True)
    print(f"텍스트·표 요약 완료 — {time.time() - start:.1f}초 | 텍스트 {len(text_summaries)}건, 표 {len(table_summaries)}건")
    if text_summaries:
        orig, summ = len(texts_2k_token[0]), len(text_summaries[0])
        print(f"압축률(첫 블록): {orig}자 → {summ}자 ({summ / orig * 100:.1f}%)")
    if table_summaries:
        print("표 요약 예시:", " ".join(table_summaries[0].split())[:200])
    else:
        print("표 요약이 없습니다. 그림으로 된 표는 그림 요약이 처리하므로, 표가 그림이면 정상입니다.")

    # 3. 그림 요약 (API 비용 발생 — 그림 수에 비례)
    image_files = select_images(config.FIGURES_DIR)
    start = time.time()
    img_base64_list, image_summaries = generate_img_summaries(image_files, IMAGE_SUMMARY_PROMPT)
    print(f"그림 요약 완료 — {time.time() - start:.1f}초 | {len(image_summaries)}장")

    # 3-1. 그림 파일과 요약문을 나란히 출력 — data/figures 의 그림을 열어 요약이 맞는지 확인하고, 평가 질문을 고를 때 참고
    for img_file, summary in zip(image_files, image_summaries):
        print(f"\n[{img_file.name}]\n{summary}")

    # 4. 토큰 사용량 (추정 — 그림 토큰은 해상도에 따라 달라서 제외)
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    input_tokens = sum(len(enc.encode(t)) for t in texts_2k_token + tables)
    output_tokens = sum(len(enc.encode(s)) for s in text_summaries + table_summaries)
    print(f"\n토큰 — 입력 {input_tokens:,} / 출력 {output_tokens:,} / 합계 {input_tokens + output_tokens:,} (그림 {len(image_summaries)}장 별도)")

    # 5. 저장 (3단계 retriever 에서 불러옴)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.SUMMARIES_PKL, "wb") as f:
        pickle.dump(
            {
                "text_summaries": text_summaries,
                "table_summaries": table_summaries,
                "image_summaries": image_summaries,
                "img_base64_list": img_base64_list,
                "image_names": [f.name for f in image_files],  # 검색된 그림이 어느 파일인지 표시용
            },
            f,
        )
    print("저장 완료:", config.SUMMARIES_PKL)


if __name__ == "__main__":
    main()
