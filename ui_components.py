"""
このファイルは、画面表示に特化した関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import logging
import streamlit as st
import constants as ct


############################################################
# 関数定義
############################################################

def display_app_title():
    """
    タイトル表示
    """
    st.markdown(f"# {ct.APP_NAME}")
    with st.sidebar:
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            st.image(ct.ICON_FILE_PATH, width=400)
        st.markdown(ct.INITIAL_EXPLANATION_SIDEBAR)


def display_initial_ai_message():
    """
    AIメッセージの初期表示
    """
    with st.chat_message("assistant", avatar=ct.AI_ICON_FILE_PATH):
        st.markdown(ct.INITIAL_EXPLANATION_MAIN)
    st.markdown(ct.RECOMMEND_INPUT_EXAMPLES_TITLE)
    st.info(ct.RECOMMEND_INPUT_EXAMPLES_TEXT)


def display_conversation_log():
    """
    会話ログの一覧表示
    """
    print(st.session_state.messages)
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar=ct.USER_ICON_FILE_PATH):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar=ct.AI_ICON_FILE_PATH):
                st.markdown(message["content"])


