__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from openai import OpenAI
import tempfile

def main():
    # 定数定義
    USER_NAME = "user"
    ASSISTANT_NAME = "assistant"
    CHROMA_PERSIST_DIR = "persistent_storage"

    # OpenAI APIキーの設定（サイドバーで入力）
    user_api_key = st.sidebar.text_input(
        label="OpenAI API key",
        placeholder="Paste your OpenAI API key",
        type="password"
    )
    os.environ['OPENAI_API_KEY'] = user_api_key

    # モデルの選択と設定
    select_model = st.sidebar.selectbox("Model", ["gpt-3.5-turbo-1106", "gpt-4-1106-preview"])
    select_temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
    select_chunk_size = st.sidebar.slider("Chunk Size", min_value=0.0, max_value=1000.0, value=300.0, step=10.0)

    # データの読み込みとベクトル化
    def load_and_embed_data(file_path, file_ext):
        print(file_path, file_ext)
        if file_ext == 'csv':
            loader = CSVLoader(file_path=file_path)
        elif file_ext == 'pdf':
            loader = PyMuPDFLoader(file_path)
        elif file_ext == 'md':
            loader = UnstructuredMarkdownLoader(file_path)
        documents = loader.load()
        
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        database = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
        database.add_documents(texts)

        st.success("データがエンベッディングされ、ベクトルデータベースが更新されました。")

    # UI周り
    st.title("QA")
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload a file", type=["csv", "pdf", "md"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                tmp_file_ext = uploaded_file.name.split(".")[-1]

            if st.button("エンベッディングを実行"):
                load_and_embed_data(tmp_file_path, tmp_file_ext)
    
    # クエリの処理
    def query_data(query):
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
        llm = ChatOpenAI(model_name=select_model, temperature=select_temperature)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
        response = qa_chain.run(query)
        return response

    # チャットログを保存したセッション情報を初期化
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    # ユーザーからのメッセージを受け取る
    user_msg = st.chat_input("ここにメッセージを入力")
    if user_msg:
        # 以前のチャットログを表示
        for chat in st.session_state.chat_log:
            with st.chat_message(chat["name"]):
                st.write(chat["msg"])

        # 最新のメッセージを表示
        with st.chat_message(USER_NAME):
            st.write(user_msg)

        # アシスタントのメッセージを表示
        response = query_data(user_msg)
        with st.chat_message(ASSISTANT_NAME):
            st.write(response)

        # セッションにチャットログを追加
        st.session_state.chat_log.append({"name": USER_NAME, "msg": user_msg})
        st.session_state.chat_log.append({"name": ASSISTANT_NAME, "msg": response})

if __name__ == "__main__":
    main()
