import fitz 
import pandas as pd
import os.path as osp

# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

from tqdm import tqdm
import os
import unicodedata

def normalize_path(path):
    """경로 유니코드 정규화"""
    return unicodedata.normalize('NFC', path)

def process_pdf(file_path, chunk_size, chunk_overlap):
    """
    PDF 텍스트 추출 -> chunk 단위 분할

    """

    # PDF 파일 열기
    doc = fitz.open(file_path)
    text = ''
    # 모든 페이지의 텍스트 추출
    for page in doc:
        text += page.get_text()
    # 텍스트를 chunk로 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunk_temp = splitter.split_text(text)
    # Document 객체 리스트 생성
    chunks = [Document(page_content=t) for t in chunk_temp]
    return chunks


def create_vector_db(chunks, model_path_id):
    """FAISS DB 생성"""

    # 임베딩 모델 설정
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path_id,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    # FAISS DB 생성 및 반환
    db = FAISS.from_documents(chunks, embedding=embeddings)
    return db


def process_pdfs_from_dataframe(root, chunk_args, vector_db_args, retreiver_args):

    """딕셔너리에 pdf명을 키로해서 DB, retriever 저장"""
    pdf_databases = {}
    df = pd.read_csv(osp.join(root, 'test.csv'))
    unique_paths = df['Source_path'].unique()
    # unique_paths = unique_paths[:2]
    
    for path in tqdm(unique_paths, desc="Processing PDFs"):
        # 경로 정규화 및 절대 경로 생성
        normalized_path = normalize_path(path)
        full_path = os.path.normpath(os.path.join(root, normalized_path.lstrip('./'))) if not os.path.isabs(normalized_path) else normalized_path
        
        pdf_title = os.path.splitext(os.path.basename(full_path))[0]
        print(f"Processing {pdf_title}...")
        
        # PDF 처리
        chunks = process_pdf(full_path, **chunk_args)

        # vector_db 생성
        db = create_vector_db(chunks, **vector_db_args)
        
        # Retriever 생성
        retriever = db.as_retriever(**retreiver_args)
        
        # 결과 저장
        pdf_databases[pdf_title] = {
                'db': db,
                'retriever': retriever
        }
    return pdf_databases










