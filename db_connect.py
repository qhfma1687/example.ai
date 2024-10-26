import streamlit as st
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# MongoDB 연결
client = MongoClient("mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
connection_string = "mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

@st.cache(allow_output_mutation=True)
def get_embeddings():
    return OpenAIEmbeddings()

def quiz_creation_page():
    """
    퀴즈 생성 페이지를 렌더링하는 함수
    """
    st.title("AI 퀴즈 생성기")
    quiz_type = st.radio("생성할 퀴즈 유형을 선택하세요:", ["다중 선택 (객관식)", "주관식", "OX 퀴즈"])
    num_quizzes = st.number_input("생성할 퀴즈의 개수를 입력하세요:", min_value=1, value=5, step=1)
    upload_option = st.radio("입력 유형을 선택하세요", ("PDF 파일", "텍스트 파일", "URL", "토픽 선택"))

    st.header("파일 업로드")
    uploaded_file = None
    text_content = None
    topic = None

    if upload_option == "토픽 선택":
        topic = st.selectbox(
           "토픽을 선택하세요",
           ("수학", "문학", "비문학", "data science", "test", "langchain", "vector_index"),
           index=None,
           placeholder="토픽을 선택하세요",
        )

        if st.button('토픽에 따른 벡터 검색'):
            # 토픽에 따른 벡터 검색 결과 출력
            embeddings = get_embeddings()
            retriever = MongoDBAtlasVectorSearch.from_connection_string(
                connection_string=connection_string,
                collection="db1.PythonDatascienceinterview",
                embeddings=embeddings,
                vector_index="vector_index"
            )

            docs = WikipediaLoader(query=topic, load_max_docs=3).load()
            db_collection = client["db1"]["PythonDatascienceinterview"]
            text_splitter = RecursiveCharacterTextSplitter()
            documents = text_splitter.split_documents(docs)
            vector_search = MongoDBAtlasVectorSearch.from_documents(
                documents=documents,
                embedding=embeddings,
                collection=db_collection,
                index_name="vector_index"
            )
            st.write(vector_search.search_results())

    if st.button('퀴즈 생성'):
        # MongoDB 연결 및 설정
        db_name = "db1"
        collection_name = "PythonDatascienceinterview"
        atlas_collection = client[db_name][collection_name]

        # 토픽을 임베딩합니다.
        embeddings = get_embeddings()
        topic_embedding = embeddings.embed_text(topic)

        # MongoDB에서 벡터 검색을 수행합니다.
        results = search_vectors(collection_name, topic_embedding)

        quiz_questions = []
        for doc in results:
            quiz_questions.append({
                "quiz": doc["quiz"],
                "options1": doc["options1"],
                "options2": doc["options2"],
                "options3": doc["options3"],
                "options4": doc["options4"],
                "correct_answer": doc["correct_answer"]
            })

        st.success('퀴즈 생성이 완료되었습니다!')
        st.write(quiz_questions)
        st.session_state['quiz_created'] = True

def search_vectors(collection_name, query_vector, top_k=10):
    """
    MongoDB에서 벡터 검색을 수행하는 함수
    """
    collection = client.db1[collection_name]
    results = collection.aggregate([
        {
            '$search': {
                'vector': {
                    'query': query_vector,
                    'path': 'vector',
                    'cosineSimilarity': True,
                    'topK': top_k,
                }
            }
        }
    ])

    return list(results)

def quiz_page():
    """
    퀴즈 풀이 페이지를 렌더링하는 함수
    """
    st.title("AI 퀴즈 생성기")
    num_quizzes = st.session_state.quiz_questions
    st.write(f"퀴즈 개수: {len(num_quizzes)}")
    for i, quiz in enumerate(num_quizzes):
        st.write(f"문제 {i + 1}: {quiz['quiz']}")
        if quiz_type == "다중 선택 (객관식)":
            st.write(f"보기 1: {quiz['options1']}")
            st.write(f"보기 2: {quiz['options2']}")
            st.write(f"보기 3: {quiz['options3']}")
            st.write(f"보기 4: {quiz['options4']}")
        st.write(f"정답: {quiz['correct_answer']}")
        st.write("=" * 50)

def main():
    """
    메인 애플리케이션 함수
    """
    st.title("AI 퀴즈 생성기")
    page = st.sidebar.selectbox(
        "페이지를 선택하세요:",
        ["퀴즈 생성", "퀴즈 풀이"]
    )

    if page == "퀴즈 생성":
        quiz_creation_page()
    elif page == "퀴즈 풀이":
        quiz_page()

if __name__ == "__main__":
    main()
