"""
다중 벡터 검색기 — 요약은 Chroma(색인 카드 서랍)에서 검색하고, 원본은 InMemoryStore(서가)에서 꺼냅니다.
요약과 원본은 같은 doc_id(청구기호)로 연결됩니다. 비교용 텍스트 전용 검색기도 여기서 만듭니다.
"""
import pickle
import re
import uuid

try:  # langchain 1.x / 0.3 어느 버전을 설치했든 동작하도록 두 경로를 모두 시도
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
except ImportError:
    from langchain.retrievers.multi_vector import MultiVectorRetriever
try:
    from langchain_core.stores import InMemoryStore
except ImportError:
    from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src import config

ID_KEY = "doc_id"                                            # 요약과 원본을 잇는 연결 키 (= 청구기호)
BASE64_PATTERN = re.compile(r"^[A-Za-z0-9+/]{100,}={0,2}$")  # 영문/숫자/+/= 만 길게 이어지면 Base64 이미지


def load_results():
    """1·2단계 결과(extracted.pkl, summaries.pkl)를 하나의 딕셔너리로 불러옵니다."""
    with open(config.EXTRACTED_PKL, "rb") as f:  # FileNotFoundError 나면 extract → summarize 순서로 먼저 실행
        extracted = pickle.load(f)
    with open(config.SUMMARIES_PKL, "rb") as f:
        summaries = pickle.load(f)
    data = {**extracted, **summaries}

    # 원본 수와 요약 수가 각각 같아야 정상 (표는 0건일 수 있음)
    print("원본 PDF:", data["source"]["pdf"], "| 쪽 범위:", data["source"]["page_ranges"] or "전체")
    print(f"텍스트 원본 {len(data['texts_2k_token']):3d}건 / 요약 {len(data['text_summaries']):3d}건")
    print(f"테이블 원본 {len(data['tables']):3d}건 / 요약 {len(data['table_summaries']):3d}건")
    print(f"이미지 원본 {len(data['img_base64_list']):3d}건 / 요약 {len(data['image_summaries']):3d}건")
    return data


def add_documents(retriever, doc_summaries, doc_contents):
    """요약본은 벡터 DB에, 원본은 docstore에 '같은 id'로 저장합니다."""
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [
        Document(page_content=s, metadata={ID_KEY: doc_ids[i]})
        for i, s in enumerate(doc_summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)          # 요약본 → 검색용 벡터 DB
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))  # 원본 → 전달용 docstore


def build_retriever(data):
    """텍스트·표·그림의 요약과 원본을 모두 넣은 다중 벡터 검색기를 만듭니다."""
    vectorstore = Chroma(
        collection_name="multimodal_rag",
        embedding_function=OpenAIEmbeddings(model=config.EMBEDDING_MODEL),
    )
    vectorstore.reset_collection()  # 같은 프로세스에서 다시 만들면 이전 데이터가 남아 있어서 비우고 시작 (안 비우면 요약이 중복)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=InMemoryStore(),  # 메모리에만 저장 — 실행할 때마다 새로 만듦
        id_key=ID_KEY,
        search_kwargs={"k": config.RETRIEVER_K},
    )
    pairs = [
        (data["text_summaries"], data["texts_2k_token"]),
        (data["table_summaries"], data["tables"]),
        (data["image_summaries"], data["img_base64_list"]),
    ]
    for doc_summaries, doc_contents in pairs:
        if doc_summaries:  # 데이터가 있을 때만 저장
            add_documents(retriever, doc_summaries, doc_contents)
    print("색인 완료 — 벡터 DB에 저장된 요약 수:", len(vectorstore.get()["ids"]))
    return retriever


def build_text_retriever(texts):
    """비교용 텍스트 전용 검색기 — 그림과 표를 뺀 것 말고는 모든 조건이 같습니다."""
    text_vectorstore = Chroma(
        collection_name="text_only_rag",
        embedding_function=OpenAIEmbeddings(model=config.EMBEDDING_MODEL),
    )
    text_vectorstore.reset_collection()  # build_retriever 와 같은 이유로 비우고 시작
    text_vectorstore.add_texts(texts)
    return text_vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_K})


def is_image(content):
    """Base64 이미지 문자열인지 생김새로 판단합니다."""
    return bool(BASE64_PATTERN.match(content))


def split_image_text_types(docs):
    """검색 결과를 Base64 이미지와 텍스트로 나눕니다."""
    b64_images = []
    texts_out = []
    for doc in docs:
        content = str(doc.page_content if isinstance(doc, Document) else doc)
        if is_image(content):
            b64_images.append(content)
        else:
            texts_out.append(content)
    return {"images": b64_images, "texts": texts_out}
