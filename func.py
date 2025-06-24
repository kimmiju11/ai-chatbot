from langchain_aws import ChatBedrock
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def llmSet():
    # .env 내용 불러오기
    load_dotenv(dotenv_path="../.env")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

    llm = ChatBedrock(
        model_id="apac.anthropic.claude-3-7-sonnet-20250219-v1:0",
        model_kwargs=dict(temperature=0.1),
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name="ap-northeast-2"  # 서울 리전
    )
    return llm

def embeddingSet():
    # langchain-aws를 사용하여 embedding 생성
    bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",
                                           region_name="ap-northeast-2")
    return bedrock_embeddings

def vectordbSet():

    # pdf 파일 로드
    loader = PyPDFLoader("doc/국민연금법.pdf")
 
    #text spliter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )
    document_list=loader.load_and_split(text_splitter=text_splitter)

    # embedding 생성
    bedrock_embeddings = embeddingSet()

    # 로컬 경로 지정
    local_path = "vector"

    database = Chroma.from_documents(
        documents=document_list, 
        embedding=bedrock_embeddings, 
        persist_directory=local_path
        )
    return database

def chainSet():
    llm = llmSet()
    database = vectordbSet()

    # 질문을 위한 프롬프트 템플릿 생성
    # AI가 대화의 맥락을 이해하고 적절한 답변을 할수 있도록함
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
    다음은 사람과 AI 어시스턴트의 친근한 대화입니다.

    대화 기록: {chat_history}
    인간: {question}

    AI 어시스턴트: 이전 대화를 고려하여 질문에 답변해 드리겠습니다.
    """)

    # QA를 위한 프롬프트 템플릿 생성
    QA_PROMPT = PromptTemplate.from_template("""
    당신은 국민연금에 대해 전문적인 지식을 가진 상담사입니다.
    주어진 컨텍스트를 기반으로 질문에 명확하고 정확하게 답변해 주세요.
    정보가 해당하는 법령도 포함해서 알려주세요.
    정보를 찾은 PDF페이지 번호도 알려주세요.

    컨텍스트: {context}
    질문: {question}

    답변:
    """)

    # 메모리 설정
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    retriever = database.as_retriever()

    # ConversationalRetrievalChain 생성 시 프롬프트 템플릿 적용
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={'prompt': QA_PROMPT}
    )
    return qa
