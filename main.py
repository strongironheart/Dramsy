"""
このファイルは、Webアプリのメイン処理が記述されたファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
from dotenv import load_dotenv
import logging
import streamlit as st
import utils
from initialize import initialize
import ui_components as uc
import constants as ct
from langchain.memory import ConversationSummaryBufferMemory
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import Document
import os

############################################################
# 設定関連
############################################################
st.set_page_config(
    page_title=ct.APP_NAME
)

load_dotenv()

logger = logging.getLogger(ct.LOGGER_NAME)


############################################################
# 初期表示
############################################################
# タイトル表示
with st.sidebar:
    uc.display_app_title()

# AIメッセージの初期表示
uc.display_initial_ai_message()

# 初期処理
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.whisky_loaded = False
    st.session_state.openai_obj = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    st.session_state.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
    st.session_state.memory = ConversationSummaryBufferMemory(
        llm=st.session_state.llm,
        max_token_limit=1000,
        return_messages=True
    )      

with st.sidebar:
    st.markdown("## 設定")
    
    # 学習期間選択ボックス
    st.session_state.mode = st.selectbox(
        label=ct.LEARNING_PERIOD_LABEL,
        options=ct.LEARNING_PERIOD_OPTIONS,
        label_visibility="collapsed",
    )
    # 目標学習時間の入力
    st.session_state.target_hours = st.number_input(
        label=ct.TARGET_HOURS_LABEL,
        min_value=1,
        max_value=300,
        value=100,
        step=10,
        help=ct.TARGET_HOURS_HELPER_TEXT,
        format="%d",
        label_visibility="collapsed",
    )
    # 1週間の学習可能時間の入力
    st.session_state.available_hours_per_week = st.number_input(
        label=ct.AVAILABLE_HOURS_PER_WEEK_LABEL,
        min_value=0,
        max_value=168,
        value=10,
        step=1,
        help=ct.AVAILABLE_HOURS_PER_WEEK_HELPER_TEXT,
        format="%d",
        label_visibility="collapsed",
    )
    # 「学習計画作成」ボタン
    if "start_flg" not in st.session_state:
        st.session_state.start_flg = False
    if st.session_state.start_flg:
        st.button(ct.START_BUTTON_LABEL, use_container_width=True, type="primary")
    else:
        st.session_state.start_flg = st.button(ct.START_BUTTON_LABEL, use_container_width=True, type="primary")

############################################################
# 初期化処理
############################################################
try:
    initialize()
except Exception as e:
    logger.error(f"{ct.INITIALIZE_ERROR_MESSAGE}\n{e}")
    st.error(utils.build_error_message(ct.INITIALIZE_ERROR_MESSAGE))
    st.stop()

# アプリ起動時のログ出力
if not "initialized" in st.session_state:
    st.session_state.initialized = True
    logger.info(ct.APP_BOOT_MESSAGE)


############################################################
# 会話ログの表示
############################################################
try:
    uc.display_conversation_log()
except Exception as e:
    logger.error(f"{ct.CONVERSATION_LOG_ERROR_MESSAGE}\n{e}")
    st.error(utils.build_error_message(ct.CONVERSATION_LOG_ERROR_MESSAGE))
    st.stop()


# ############################################################
# # チャット入力の受け付け
# ############################################################
chat_message = st.chat_input(ct.CHAT_INPUT_HELPER_TEXT)


############################################################
# チャット送信時の処理
############################################################
if chat_message:
    # ==========================================
    # 1. ユーザーメッセージの表示
    # ==========================================
    logger.info({"message": chat_message})

    with st.chat_message("user", avatar=ct.USER_ICON_FILE_PATH):
        st.markdown(chat_message)

    # ==========================================
    # 2. LLMからの回答取得
    # ==========================================
    res_box = st.empty()
    with st.spinner(ct.SPINNER_TEXT):
        try:
            # ここでコンテキスト（関連ドキュメント）を取得
            context_docs = st.session_state.retriever.get_relevant_documents(chat_message)
            context = "\n".join([doc.page_content for doc in context_docs])
            # print("==============")
            # print(context_docs)
            # print("==============")
            # print(context)
            # print("==============")

            # ここでプロンプトを作成
            question_answer_prompt = utils.build_question_answer_prompt(context, chat_message)
            # print("///////////////////////")
            # print(question_answer_prompt)
            # print("///////////////////////")
            # プロンプトをレンダリングしてLLMに渡す
            prompt_messages = question_answer_prompt.format_messages(
                input=chat_message,
                context=context,
                chat_history=st.session_state.messages  # ← これを追加
            )

            result = st.session_state.llm.invoke(prompt_messages)
        except Exception as e:
            logger.error(f"{ct.RECOMMEND_ERROR_MESSAGE}\n{e}")
            st.error(utils.build_error_message(ct.RECOMMEND_ERROR_MESSAGE))
            st.stop()
    
    # ==========================================
    # 3. LLMからの回答表示
    # ==========================================
    with st.chat_message("assistant", avatar=ct.AI_ICON_FILE_PATH):
        try:
            st.markdown(result.content, unsafe_allow_html=True)
            logger.info({"message": result})
        except Exception as e:
            logger.error(f"{ct.LLM_RESPONSE_DISP_ERROR_MESSAGE}\n{e}")
            st.error(utils.build_error_message(ct.LLM_RESPONSE_DISP_ERROR_MESSAGE))
            st.stop()

    # ==========================================
    # 4. 会話ログへの追加
    # ==========================================
    st.session_state.messages.append({"role": "user", "content": chat_message})
    st.session_state.messages.append({"role": "assistant", "content": result.content})
    print(st.session_state.messages)

