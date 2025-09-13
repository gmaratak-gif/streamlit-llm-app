# app.py
# Streamlit × LangChain × OpenAI 簡易LLMアプリ

import os
from typing import Literal

import streamlit as st
from dotenv import load_dotenv

# === 環境変数の読み込み ===
load_dotenv()

# Streamlit Cloud の Secrets を優先
if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# === LangChain の準備 ===
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 専門家ロール
ExpertKey = Literal["株式投資アナリスト（A）", "生成AIエンジニア（B）"]

EXPERT_SYSTEM_MESSAGES: dict[ExpertKey, str] = {
    "株式投資アナリスト（A）": (
        "あなたは厳格で客観的な株式投資アナリストです。"
        "常に根拠・前提・リスク・代替案を提示し、"
        "専門用語は短く説明を添えてください。"
        "推奨は断定せず、教育目的として述べます。"
    ),
    "生成AIエンジニア（B）": (
        "あなたは実務に強い生成AIエンジニアです。"
        "設計方針→手順→コード例→検証方法→運用上の注意の順で、"
        "具体的かつ再現可能な手順で説明してください。"
    ),
}


def build_chain(expert_key: ExpertKey, model_name: str = "gpt-4o-mini", temperature: float = 0.2):
    """選択した専門家ロールでチェーンを構築"""
    system_msg = EXPERT_SYSTEM_MESSAGES[expert_key]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human", "{user_input}"),
        ]
    )

    llm = ChatOpenAI(model=model_name, temperature=temperature)
    parser = StrOutputParser()
    return prompt | llm | parser


def get_llm_answer(user_text: str, expert_key: ExpertKey, model_name: str = "gpt-4o-mini", temperature: float = 0.2) -> str:
    """入力テキストとロールを基にLLMの回答を返す"""
    chain = build_chain(expert_key=expert_key, model_name=model_name, temperature=temperature)
    return chain.invoke({"user_input": user_text})


# === Streamlit UI ===
st.set_page_config(page_title="Streamlit × LangChain LLM アプリ", page_icon="🤖", layout="centered")

st.title("🤖 Streamlit × LangChain LLM アプリ")
st.caption("Python 3.11 / LangChain / OpenAI API（.env または Secrets）")

with st.expander("ℹ️ 使い方", expanded=True):
    st.markdown(
        """
        - 入力欄に質問を入力し、送信するとLLMが回答します。  
        - ラジオボタンで専門家の役割を選ぶと、回答のスタイルが変わります。  
        """
    )

# APIキー未設定の警告
if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY が見つかりません。.env または Streamlit Secrets に設定してください。", icon="⚠️")

# サイドバー（オプション設定）
with st.sidebar:
    st.header("⚙️ 設定")
    model_name = st.selectbox("モデル", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    temperature = st.slider("Temperature（創造性）", 0.0, 1.0, 0.2, 0.1)
    st.markdown("---")
    st.markdown("**注意**: Streamlit Cloud では Python バージョンを 3.11 に設定してください。")

# 専門家ロール選択
expert_choice: ExpertKey = st.radio(
    "専門家ロールを選択してください：",
    options=list(EXPERT_SYSTEM_MESSAGES.keys()),
    horizontal=True,
)

# 入力フォーム
with st.form(key="llm_form", clear_on_submit=False):
    user_text = st.text_area(
        "質問 / お題",
        placeholder="例）このテーマで投資のリスク要因を整理して、優先順位をつけてください…",
        height=140,
    )
    submitted = st.form_submit_button("送信")

# 送信処理
if submitted:
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY が未設定です。.env もしくは Streamlit Secrets を確認してください。")
    elif not user_text.strip():
        st.warning("テキストを入力してください。")
    else:
        with st.spinner("LLM が考えています…"):
            try:
                answer = get_llm_answer(
                    user_text=user_text.strip(),
                    expert_key=expert_choice,
                    model_name=model_name,
                    temperature=temperature,
                )
                st.success("回答が生成されました。")
                st.markdown("### 📝 回答")
                st.markdown(answer)
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

