"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import logging
from typing import List
from sudachipy import tokenizer, dictionary
import constants as ct
import sqlite3
import streamlit as st
import requests
import os

############################################################
# 関数定義
############################################################

def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def preprocess_func(text):
    """
    形態素解析による日本語の単語分割
    Args:
        text: 単語分割対象のテキスト
    
    Returns:
        単語分割を実施後のテキスト
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    tokenizer_obj = dictionary.Dictionary(dict="full").create()
    mode = tokenizer.Tokenizer.SplitMode.A
    tokens = tokenizer_obj.tokenize(text ,mode)
    words = [token.surface() for token in tokens]
    words = list(set(words))

    return words


############################################################
# 関数定義
############################################################
def create_whisky_table():
    conn = sqlite3.connect(ct.DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS whisky (
            id INTEGER PRIMARY KEY,
            name TEXT,
            description TEXT,
            url TEXT,
            lang TEXT,
            published TEXT,
            author TEXT,
            image_url TEXT,
            foto_url TEXT,
            affiliate_url TEXT,
            type TEXT,
            country TEXT,
            region TEXT,
            distillery TEXT,
            bottler TEXT,
            age INTEGER,
            abv REAL,
            price REAL,
            nose TEXT,
            palate TEXT,
            finish TEXT,
            conclusion TEXT,
            rating_marcel INTEGER,
            rating_sascha INTEGER,
            rating_average INTEGER,
            rating_text TEXT,
            value_for_money INTEGER,
            value_for_money_text TEXT
        )
    """)
    conn.commit()
    conn.close()

def fetch_whiskies():
    response = requests.get(ct.API_URL)
    response.raise_for_status()
    return response.json()

def insert_whiskies(whiskies):
    conn = sqlite3.connect(ct.DB_PATH)
    c = conn.cursor()
    for w in whiskies:
        metadata = w.get("metadata", {})
        tasting = w.get("tasting_notes", {})
        rating = w.get("rating", {})
        c.execute("""
            INSERT OR REPLACE INTO whisky VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
        """, (
            w.get("id"),
            w.get("name"),
            w.get("description"),
            w.get("url"),
            w.get("lang"),
            w.get("published"),
            w.get("author"),
            w.get("image_url"),
            w.get("foto_url"),
            w.get("affiliate_url"),
            metadata.get("type"),
            metadata.get("country"),
            metadata.get("region"),
            metadata.get("distillery"),
            metadata.get("bottler"),
            metadata.get("age"),
            metadata.get("abv"),
            metadata.get("price"),
            tasting.get("nose"),
            tasting.get("palate"),
            tasting.get("finish"),
            tasting.get("conclusion"),
            rating.get("marcel"),
            rating.get("sascha"),
            rating.get("average"),
            rating.get("rating_text"),
            rating.get("value_for_money"),
            rating.get("value_for_money_text"),
        ))
    conn.commit()
    conn.close()

def get_all_whiskies():
    conn = sqlite3.connect(ct.DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM whisky")
    rows = c.fetchall()
    conn.close()
    return rows

def db_exists():
    return os.path.exists(ct.DB_PATH)

def table_exists():
    conn = sqlite3.connect(ct.DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='whisky'")
    exists = c.fetchone() is not None
    conn.close()
    return exists

def get_existing_ids():
    conn = sqlite3.connect(ct.DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM whisky")
    ids = set(row[0] for row in c.fetchall())
    conn.close()
    return ids

def insert_new_whiskies(whiskies):
    existing_ids = get_existing_ids()
    new_whiskies = [w for w in whiskies if w.get("id") not in existing_ids]
    if not new_whiskies:
        return
    insert_whiskies(new_whiskies)

def display_whisky_db():
    """Whisky DBの中身を画面に表示する関数"""
    st.markdown("### Whisky DB 一覧")
    whiskies = get_all_whiskies()
    if whiskies:
        for whisky in whiskies:
            st.write(whisky)
    else:
        st.info("DBにデータがありません。")

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

def build_question_answer_prompt(context, user_input):
    """
    ベクタDBから取得したコンテキストとユーザー入力を元に回答生成用プロンプトを作成
    """
    question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
    # context（検索結果）とuser_input（質問）をテンプレートに埋め込む
    prompt_text = question_answer_template.format(context=context, input=user_input)
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_text),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return question_answer_prompt