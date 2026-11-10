"""
답변 체인 — 멀티모달 체인(검색된 그림과 텍스트를 함께 LLM에 전달)과 비교용 텍스트 전용 체인입니다.

    {검색 + 질문}  →  img_prompt_func  →  멀티모달 LLM  →  StrOutputParser
       재료 모으기      메시지로 조립        답변 생성        문자열로 꺼내기
"""
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from src import config
from src.retriever import split_image_text_types

ANSWER_INSTRUCTION = "아래 자료를 근거로 질문에 한국어로 답하세요.\n자료에 없는 내용은 모른다고 답하세요."  # 두 체인이 같은 지시문을 씀


def img_prompt_func(data_dict):
    """검색된 원본을 멀티모달 LLM이 읽을 형태로 조립합니다."""
    messages = []
    images = data_dict["context"]["images"]

    # 1. 이미지를 먼저 넣습니다 (모델이 그림을 먼저 보게)
    for image in images:
        messages.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image}",
                "detail": "high",  # 고해상도로 보게 함 (기본값 auto 는 모델이 저해상도로 볼 수 있어 작은 숫자·글자를 놓침)
            },
        })

    # 2. 텍스트 컨텍스트와 질문을 넣습니다
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    image_note = f"위 그림 {len(images)}장도 자료입니다. 그림 속 글자와 숫자까지 확인하세요.\n" if images else ""  # 지시문의 "아래 자료"만 보지 않도록
    messages.append({
        "type": "text",
        "text": (
            f"{ANSWER_INSTRUCTION}\n{image_note}\n"
            f"질문: {data_dict['question']}\n\n"
            f"참고 텍스트:\n{formatted_texts}"
        ),
    })

    return [HumanMessage(content=messages)]


def build_multimodal_chain(retriever):
    """다중 벡터 검색기로 찾은 그림·텍스트를 멀티모달 LLM에 넘겨 답하는 체인을 만듭니다."""
    model = ChatOpenAI(temperature=0, model=config.ANSWER_MODEL, max_tokens=1024)
    return (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )


def format_docs(docs):
    """검색된 문서들의 본문을 줄바꿈으로 합칩니다."""
    return "\n".join(doc.page_content for doc in docs)


def build_text_only_chain(text_retriever):
    """비교용 텍스트 전용 RAG 체인 — 같은 지시문과 모델, 텍스트만 검색."""
    model = ChatOpenAI(temperature=0, model=config.ANSWER_MODEL, max_tokens=1024)
    prompt = ChatPromptTemplate.from_template(
        ANSWER_INSTRUCTION + "\n\n질문: {question}\n\n참고 텍스트:\n{context}"
    )
    return (
        {"context": text_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
