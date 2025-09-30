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
        st.image(ct.ICON_FILE_PATH, width=200)
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
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar=ct.USER_ICON_FILE_PATH):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar=ct.AI_ICON_FILE_PATH):
                display_product(message["content"])


def display_product(result):
    """
    商品情報の表示

    Args:
        result: LLMからの回答
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # LLMレスポンスのテキストを辞書に変換
    product_lines = result[0].page_content.split("\n")
#    print(product_lines)
#    print("--------------")    
    product = {item.split(": ")[0]: item.split(": ")[1] for item in product_lines}
    if '\ufeffid' in product:
        product['id'] = product.pop('\ufeffid')
#    print(product)
    st.markdown(ct.PROPOSAL_MESSAGE)

    # 「商品名」と「価格」
    st.success(f"""
            {ct.PRODUCT_NAME}: {product['name']}<br>{ct.PRODUCT_ID}: {product['id']}<br>
            {ct.PRICE}: {product['price']}
    """)
    # st.success(f"""
    #         商品名：{product['name']}（商品ID: {product['id']}）\n
    #         価格：{product['price']}
    # """)

    # 在庫状況を表示
    if product.get('stock_status') == ct.STOCK_STATUS_LOW:
        st.warning(ct.STOCK_STATUS_LOW_MESSAGE,icon=ct.WARNING_ICON)
    elif product.get('stock_status') == ct.STOCK_STATUS_OUT_OF_STOCK:
        st.error(ct.STOCK_STATUS_OUT_OF_STOCK_MESSAGE,icon=ct.ERROR_ICON)

    # 「商品カテゴリ」と「メーカー」と「ユーザー評価」
    st.code(f"""
        商品カテゴリ：{product['category']}\n
        メーカー：{product['maker']}\n
        評価：{product['score']}({product['review_number']}件)
    """, language=None, wrap_lines=True)

    # 商品画像
    st.image(f"images/products/{product['file_name']}", width=400)

    # 商品説明
    st.code(product['description'], language=None, wrap_lines=True)

    # おすすめ対象ユーザー
    st.markdown(ct.RECOMMENDED_PEOPLE)
    st.info(product["recommended_people"])

    # 商品ページのリンク
    st.link_button(ct.OPEN_PRODUCTS_PAGE, type="primary", use_container_width=True, url="https://google.com")
