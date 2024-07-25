#https://qiita.com/MaTTA_RUNTEQ50B/items/c9545f16bc362237d8a9
# pip install --no-cache-dir -r requirements.txt
# streamlit run main.py

import os
import streamlit as st
from langchain_community.document_loaders import CSVLoader
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from openai import OpenAI
import tempfile # PDFアップロードの際に必要

def main():
    # 定数定義
    USER_NAME = "user"
    ASSISTANT_NAME = "assistant"

    # OpenAI APIキーの設定
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

    # Chromaデータベースの設定
    CHROMA_PERSIST_DIR = "persistent_storage"

    # データの読み込みとベクトル化
    def load_and_embed_data(file_path, file_ext):
        print(file_path, file_ext)
        if file_ext == 'csv':
            loader = CSVLoader(file_path=file_path)
            documents = loader.load()
        elif file_ext == 'pdf':
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
        elif file_ext == 'md':
            loader = UnstructuredMarkdownLoader(file_path)
            documents = loader.load()
        
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
        )

        database = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
        )

        database.add_documents(texts)

        st.success("データがエンベッディングされ、ベクトルデータベースが更新されました。")

    # UI周り
    st.title("QA")
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload a file after paste OpenAI API key", type="pdf")
        # 一時ファイルにPDFを書き込みバスを取得
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                tmp_file_ext = uploaded_file.name.split(".")[-1]

        if st.button("エンベッディングを実行"):
            load_and_embed_data(tmp_file_path, tmp_file_ext)
        
        select_model = st.selectbox("Model", ["gpt-3.5-turbo-1106", "gpt-4-1106-preview",])
        select_temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1,)
        select_chunk_size = st.slider("Chunk", min_value=0.0, max_value=1000.0, value=300.0, step=10.0,)

    # クエリの処理
    def query_data(query):
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
        
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        
        response = qa_chain.run(query)
        return response

    #st.title("LangChain ver.2 RAG APP with CSV")

    #if st.button("エンベッディングを実行"):
        #load_and_embed_data("data/data.csv") 
        load_and_embed_data("data/ut-markdown.md")

    # APIキーの設定
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    def response_chatgpt(
        user_msg: str,
    ):
        """ChatGPTのレスポンスを取得

        Args:
            user_msg (str): ユーザーメッセージ。
        """
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": user_msg},
            ],
            model="gpt-3.5-turbo",
            stream=True,
        )
        return response


    # チャットログを保存したセッション情報を初期化
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []


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
        #response = response_chatgpt(user_msg)
        response = query_data(user_msg)
        #print(response)
        with st.chat_message(ASSISTANT_NAME):
            st.write(response)

        # セッションにチャットログを追加
        st.session_state.chat_log.append({"name": USER_NAME, "msg": user_msg})
        st.session_state.chat_log.append({"name": ASSISTANT_NAME, "msg": response})

if __name__ == "__main__":
    main()
