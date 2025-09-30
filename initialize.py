"""
このファイルは、最初の画面読み込み時にのみ実行される初期化処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from uuid import uuid4
import sys
import unicodedata
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import utils
import constants as ct
import requests
import sqlite3


############################################################
# 設定関連
############################################################
load_dotenv()


############################################################
# 関数定義
############################################################

def initialize():
    """
    画面読み込み時に実行する初期化処理
    """
    # 初期化データの用意
    initialize_session_state()
    # ログ出力用にセッションIDを生成
    initialize_session_id()
    # ログ出力の設定
    initialize_logger()
    # RAGのRetrieverを作成
    initialize_retriever()


def initialize_logger():
    """
    ログ出力の設定
    """
    os.makedirs(ct.LOG_DIR_PATH, exist_ok=True)
    
    logger = logging.getLogger(ct.LOGGER_NAME)

    if logger.hasHandlers():
        return

    log_handler = TimedRotatingFileHandler(
        os.path.join(ct.LOG_DIR_PATH, ct.LOG_FILE),
        when="D",
        encoding="utf8"
    )
    formatter = logging.Formatter(
        f"[%(levelname)s] %(asctime)s %(filename)s line %(lineno)s, in %(funcName)s, session_id={st.session_state.session_id}: %(message)s"
    )
    log_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)


def initialize_session_id():
    """
    セッションIDの作成
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid4().hex


def initialize_session_state():
    """
    初期化データの用意
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []


def initialize_retriever():
    """
    Retrieverを作成（whisky.dbからロードし、VectorDBをpersistentで保存）
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    if "retriever" in st.session_state:
        return

    # whisky.dbからデータをロード
    conn = sqlite3.connect("whisky.db")
    c = conn.cursor()
    c.execute("""
        SELECT name, published, image_url, type, country, region, distillery, bottler, age, abv, price,
               nose, palate, finish, conclusion, rating_text, value_for_money
        FROM whisky
    """)
    rows = c.fetchall()
    conn.close()

    docs = []
    for row in rows:
        (name, published, image_url, type_, country, region, distillery, bottler, age, abv, price,
         nose, palate, finish, conclusion, rating_text, value_for_money) = row
        page_content = (
            f"Name: {name}\n"
            f"Published: {published}\n"
            f"Image URL: {image_url}\n"
            f"Type: {type_}\n"
            f"Country: {country}\n"
            f"Region: {region}\n"
            f"Distillery: {distillery}\n"
            f"Bottler: {bottler}\n"
            f"Age: {age}\n"
            f"ABV: {abv}\n"
            f"Price: {price}\n"
            f"Nose: {nose}\n"
            f"Palate: {palate}\n"
            f"Finish: {finish}\n"
            f"Conclusion: {conclusion}\n"
            f"Rating: {rating_text}\n"
            f"Value for Money: {value_for_money}"
        )
        docs.append(page_content)

    # embeddingsとpersistentなvectorstoreの作成
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_texts(
        docs,
        embedding=embeddings,
        persist_directory=ct.CHROMA_DB_DIR  # ← persistent保存先を指定
    )
    db.persist()  # データを永続化

    retriever = db.as_retriever(search_kwargs={"k": ct.TOP_K})

    bm25_retriever = BM25Retriever.from_texts(
        docs,
        preprocess_func=utils.preprocess_func,
        k=ct.TOP_K
    )
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever],
        weights=ct.RETRIEVER_WEIGHTS
    )

    st.session_state.retriever = ensemble_retriever


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s