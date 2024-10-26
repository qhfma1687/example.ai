#quiz_creation_page.py

import io
import chardet
import pytesseract
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from pymongo import MongoClient
from langchain import hub
from langchain.chains import create_retrieval_chain
from pymongo.server_api import ServerApi
from pymongo.errors import OperationFailure
from langchain.prompts.prompt import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader, WikipediaLoader
from langchain_community.vectorstores import Chroma, MongoDBAtlasVectorSearch
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain_community.vectorstores import FAISS
from bs4 import BeautifulSoup as Soup
from urllib.parse import urlparse

def is_url(input_string):
    try:
        result = urlparse(input_string)
        return all([result.scheme, result.netloc])
    except:
        return False


#아이디는 코드에 들어가진 않습니다.
#embedings 항목에 array 형식으로 저장된 벡터 값으로 벡터 검색이 되고 atlas vextet index 항목에서 검색기로 등록해주면 검색 가능하다고 합니다. 
#acm41th:vCcYRo8b4hsWJkUj@cluster0 여기까지가 아이디:비밀번호:클러스터 주소라 필수적입니다. 마지막 앱네임도 클러스터명

#Vectorstore
client = MongoClient("mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
#client['your_database_name']이 데베 이름입니다. 데베1은 파이썬 관련 정보가 용량이 적길래 일단 넣어줬습니다.
#임베딩 항목은 따로 처리해서 넣어줘야 할 겁니다.
#랭체인도 데모 데이터로 몽고디비 관련 내용이고 엠플릭스도 영화 관련 데모 데이터입니다.
#콜렉션은 각 디비 안에 있는 데이터셋을 뜻합니다. 디비가 폴더고 얘가 파일 같습니다.
#임베딩값이 들어 있는 콜렉션은 일단 embeded_movies랑 test가 있습니다. 각각 sample_mflix.embedded_movies
#, langchain_db.test처럼 넣어서 쓰면 됩니다.

def connect_db():
    client = MongoClient("mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    return client[langchain_db]

def insert_documents(collection_name, documents):
    db = connect_db()
    collection = db[test]
    collection.insert_many(documents)

def vectorize_and_store(data, collection_name):
    embeddings = OpenAIEmbeddings()
    vector_operations = []

    for document in data:
        text = document['text']
        vector = embeddings.embed_text(text)
        operation = UpdateOne({'_id': document['_id']}, {'$set': {'vector': vector.tolist()}})
        vector_operations.append(operation)

    db = connect_db()
    collection = db[test]
    collection.bulk_write(vector_operations)

def search_vectors(collection_name, query_vector, top_k=10):
    db = connect_db()
    collection = db[test]
    results = collection.aggregate([
        {
            '$search': {
                'vector': {
                    'query': query_vector,
                    'path': 'vector',
                    'cosineSimilarity': True,
                    'topK': top_k
                }
            }
        }
    ])

    #st.write("Question: " + query_vector)
    #st.write("Answer: " + results)
    
    return list(results)

def retrieve_results(user_query):
    # Create MongoDB Atlas Vector Search instance
    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        "mongodb+srv://username:password@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=YourApp",
        "langchain_db.test",
        OpenAIEmbeddings(model="gpt-3.5-turbo-0125"),
        index_name="vector_index"
    )

    # Perform vector search based on user input
    response = vector_search.similarity_search_with_score(
        input=user_query, k=5, pre_filter={"page": {"$eq": 1}}
    )

    st.write("Question: " + user_query)
    st.write("Answer: " + response)

    # Check if any results are found
    if not response:
        return None

    return response


class CreateQuizoub(BaseModel):
    quiz: str = Field(description="The created problem")
    options1: str = Field(description="The first option of the created problem")
    options2: str = Field(description="The second option of the created problem")
    options3: str = Field(description="The third option of the created problem")
    options4: str = Field(description="The fourth option of the created problem")
    correct_answer: str = Field(description="One of the options1 or options2 or options3 or options4")

class CreateQuizsub(BaseModel):
    quiz: str = Field(description="The created problem")
    correct_answer: str = Field(description="The answer to the problem")
    commentary: str = Field(description="The commentary of answer to this problem")

class CreateQuizTF(BaseModel):
    quiz: str = Field(description="The created problem")
    options1: str = Field(description="The true or false option of the created problem")
    options2: str = Field(description="The true or false option of the created problem")
    correct_answer: str = Field(description="One of the options1 or options2")

def make_model(pages):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    embeddings = OpenAIEmbeddings()

    # Rag
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(pages)
    vector = FAISS.from_documents(documents, embeddings)

    # PydanticOutputParser 생성
    parseroub = PydanticOutputParser(pydantic_object=CreateQuizoub)
    parsersub = PydanticOutputParser(pydantic_object=CreateQuizsub)
    parsertf = PydanticOutputParser(pydantic_object=CreateQuizTF)

    prompt = PromptTemplate.from_template(
        "Question: {input}, Please answer in KOREAN."

        "CONTEXT:"
        "{context}."

        "FORMAT:"
        "{format}"
    )
    promptoub = prompt.partial(format=parseroub.get_format_instructions())
    promptsub = prompt.partial(format=parsersub.get_format_instructions())
    prompttf = prompt.partial(format=parsertf.get_format_instructions())

    document_chainoub = create_stuff_documents_chain(llm, promptoub)
    document_chainsub = create_stuff_documents_chain(llm, promptsub)
    document_chaintf = create_stuff_documents_chain(llm, prompttf)

    retriever = vector.as_retriever()

    retrieval_chainoub = create_retrieval_chain(retriever, document_chainoub)
    retrieval_chainsub = create_retrieval_chain(retriever, document_chainsub)
    retrieval_chaintf = create_retrieval_chain(retriever, document_chaintf)

    # chainoub = promptoub | chat_model | parseroub
    # chainsub = promptsub | chat_model | parsersub
    # chaintf = prompttf | chat_model | parsertf
    return 0


def process_text(text_area_content):
    text_content = st.text_area("텍스트를 입력하세요.")

    return text_content

# 파일 처리 함수
def process_file(uploaded_file, upload_option):

    uploaded_file = None
    text_area_content = None
    url_area_content = None
    selected_topic = None
    
    # # 파일 업로드 옵션 선택
    # upload_option = st.radio("입력 유형을 선택하세요", ("이미지 파일", "PDF 파일", "직접 입력", "URL", "토픽 선택"))

    # 선택된 옵션에 따라 입력 방식 제공
    if upload_option == "텍스트 파일":
        uploaded_file = st.file_uploader("텍스트 파일을 업로드하세요.", type=["txt"])
    elif upload_option == "이미지 파일":
        uploaded_file = st.file_uploader("이미지 파일을 업로드하세요.", type=["jpg", "jpeg", "png"])
    elif upload_option == "PDF 파일":
        uploaded_file = st.file_uploader("PDF 파일을 업로드하세요.", type=["pdf"])
    else:
        uploaded_file = None

    # 업로드된 파일 처리
    if uploaded_file is None:
        st.warning("파일을 업로드하세요.")
        return None

    if uploaded_file.type == "text/plain":
        text_content = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type.startswith("image/"):
        image = Image.open(uploaded_file)
        text_content = pytesseract.image_to_string(image)
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text()
    else:
        st.error("지원하지 않는 파일 형식입니다.")
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    if text_area_content is not None:
        text_content = process_file(uploaded_file, text_area_content) #?
    texts = text_splitter.create_documents([text_content])
    
    return texts
    

# 퀴즈 생성 함수
@st.experimental_fragment
def generate_quiz(quiz_type, is_topic, retrieval_chainoub, retrieval_chainsub, retrieval_chaintf):
    try:
        # Generate quiz prompt based on selected quiz type
        if is_topic is None:
            if quiz_type == "다중 선택 (객관식)":
                response = retrieval_chainoub.invoke(
                    {
                        "input": "Create one multiple-choice question focusing on important concepts, following the given format, referring to the following context"
                    }
                )
            elif quiz_type == "주관식":
                response = retrieval_chainsub.invoke(
                    {
                        "input": "Create one open-ended question focusing on important concepts, following the given format, referring to the following context"
                    }
                )
            elif quiz_type == "OX 퀴즈":
                response = retrieval_chaintf.invoke(
                    {
                        "input": "Create one true or false question focusing on important concepts, following the given format, referring to the following context"
                    }
                )
            quiz_questions = response
        else:
            if quiz_type == "다중 선택 (객관식)":
                response = retrieval_chainoub.invoke(
                    {
                        "input": f"Create one {is_topic} multiple-choice question focusing on important concepts, following the given format, referring to the following context"
                    }
                )
            elif quiz_type == "주관식":
                response = retrieval_chainsub.invoke(
                    {
                        "input":  f"Create one {is_topic} open-ended question focusing on important concepts, following the given format, referring to the following context"
                    }
                )
            elif quiz_type == "OX 퀴즈":
                response = retrieval_chaintf.invoke(
                    {
                        "input":  f"Create one {is_topic} true or false question focusing on important concepts, following the given format, referring to the following context"
                    }
                )
            quiz_questions = response

        return quiz_questions
    except Exception as e:
        st.error("유효하지 않은 사용자 입력입니다")
        return None

@st.experimental_fragment
def grade_quiz_answer(user_answer, quiz_answer):
    if user_answer.lower() == quiz_answer.lower():
        grade = "정답"
    else:
        grade = "오답"
    return grade

# 메인 함수
def quiz_creation_page():
    placeholder = st.empty()
    st.session_state.page = 0
    if st.session_state.page == 0:
        with placeholder.container():
            st.title("AI Quiz Generator")
            if 'selected_page' not in st.session_state:
                st.session_state.selected_page = ""

            # 퀴즈 유형 선택
            quiz_type = st.radio("생성할 퀴즈 유형을 선택하세요:", ["다중 선택 (객관식)", "주관식", "OX 퀴즈"],horizontal=True)

            # 퀴즈 개수 선택
            num_quizzes = st.number_input("생성할 퀴즈의 개수를 입력하세요:", min_value=1, value=5, step=1)

            # 파일 업로드 옵션 선택
            upload_option = st.radio("입력 유형을 선택하세요", ("PDF 파일", "텍스트 파일", "URL", "토픽 선택"),horizontal=True)

            # 파일 업로드 옵션
            st.header("파일 업로드")
            uploaded_file = None
            text_content = None
            topic = None
            url_area_content = None
            #uploaded_file = st.file_uploader("텍스트, 이미지, 또는 PDF 파일을 업로드하세요.", type=["txt", "jpg", "jpeg", "png", "pdf"])

            # if upload_option == "직접 입력":               
            #     text_input = st.text_area("텍스트를 입력하세요.")
            #     st.write(text_input)
                # text_content = text_input.load().encoding("utf-8", errors='ignore')
                
                # result = chardet.detect(text_input)
                # encoding = result['encoding']
                # text_content = text_input.decode(encoding)
          
                # try:
                #     text_content = text_input.encoding("utf-8")
                # except UnicodeDecodeError:
                #     # 오류 처리 코드 작성
                #     text_content = text_input.encoding("utf-8")

            
            if upload_option == "토픽 선택":
                topic = st.selectbox(
                   "토픽을 선택하세요",
                   ("수학", "문학", "비문학", "과학", "test", "langchain", "vector_index"),
                   index=None,
                   placeholder="토픽을 선택하세요",
                ) 

            elif upload_option == "URL":
                url_area_content = st.text_area("URL을 입력하세요.")
                # if not url_area_content:  # Check if URL is empty
                #     st.error("URL을 입력해야 합니다.")  # Display error message
                #     return
                
                if not is_url(url_area_content):
                    st.error("URL을 입력해야 합니다.")
                    return

                loader = RecursiveUrlLoader(
                    url=url_area_content, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text
                )
                text_content = loader.load()
                
            else:
                text_content = process_file(uploaded_file, upload_option)
            

            quiz_questions = []

            if text_content is not None:
                if st.button('문제 생성 하기'):
                    with st.spinner('퀴즈를 생성 중입니다...'):
                        llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
                        embeddings = OpenAIEmbeddings()

                        uri = "mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
                        # Create a new client and connect to the server
                        client = MongoClient(uri, server_api=ServerApi('1'))
                        # Send a ping to confirm a successful connection
                        try:
                            client.admin.command('ping')
                            st.write("Pinged your deployment. You successfully connected to MongoDB!")
                        except Exception as e:
                            st.write(e)

                        # Vectorstore
                        # client = MongoClient("mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")

                        # Define collection and index name
                        db_name = "langchain_db"
                        collection_name = "test"
                        atlas_collection = client[db_name][collection_name]
                        vector_search_index = "vector_index"

                        # Rag
                        text_splitter = RecursiveCharacterTextSplitter()
                        # documents = text_content
                        documents = text_splitter.split_documents(text_content)

                        # try:
                        #   connection.test.foo.find_one()
                        # except pymongo.errors.OperationFailure as e:
                        #     st.write(e.code)
                        #     st.write(e.details)

                        vector_search = MongoDBAtlasVectorSearch.from_documents(
                            documents=documents,
                            embedding=embeddings,
                            collection=atlas_collection,
                            index_name=vector_search_index
                        )

                        # Instantiate Atlas Vector Search as a retriever
                        retriever = vector_search.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 5, "score_threshold": 0.75}
                        )

                        # PydanticOutputParser 생성
                        parseroub = PydanticOutputParser(pydantic_object=CreateQuizoub)
                        parsersub = PydanticOutputParser(pydantic_object=CreateQuizsub)
                        parsertf = PydanticOutputParser(pydantic_object=CreateQuizTF)

                        prompt = PromptTemplate.from_template(
                            "{input}, Please answer in KOREAN."

                            "CONTEXT:"
                            "{context}."

                            "FORMAT:"
                            "{format}"
                        )
                        promptoub = prompt.partial(format=parseroub.get_format_instructions())
                        promptsub = prompt.partial(format=parsersub.get_format_instructions())
                        prompttf = prompt.partial(format=parsertf.get_format_instructions())

                        document_chainoub = create_stuff_documents_chain(llm, promptoub)
                        document_chainsub = create_stuff_documents_chain(llm, promptsub)
                        document_chaintf = create_stuff_documents_chain(llm, prompttf)

                        # retriever = vector.as_retriever()

                        retrieval_chainoub = create_retrieval_chain(retriever, document_chainoub)
                        retrieval_chainsub = create_retrieval_chain(retriever, document_chainsub)
                        retrieval_chaintf = create_retrieval_chain(retriever, document_chaintf)

                        is_topic = None

                        for i in range(num_quizzes):
                            try:
                                quiz_questions.append(generate_quiz(quiz_type, text_content, retrieval_chainoub, retrieval_chainsub,retrieval_chaintf))
                                st.session_state['quizs'] = quiz_questions
                            except OperationFailure as e:
                                st.write(f"Failed to fetch documents: {e}")
                        st.session_state.selected_page = "퀴즈 풀이"
                        st.session_state.selected_type = quiz_type
                        st.session_state.selected_num = num_quizzes

                        st.success('퀴즈 생성이 완료되었습니다!')
                        st.write(quiz_questions)
                        st.session_state['quiz_created'] = True

                if st.session_state.get('quiz_created', False):
                    if st.button('퀴즈 풀기'):
                        st.switch_page("pages/quiz_solve_page.py")

            elif topic is not None:
                if st.button('문제 생성 하기'):
                    with st.spinner('퀴즈를 생성 중입니다...'):
                        llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
                        embeddings = OpenAIEmbeddings()
                        
                        if topic == "수학":
                            is_topic = "Mathematics"
                        elif topic == "과학":
                            is_topic = "science"
                        else:
                            is_topic = topic
                        st.write(f"{is_topic}")

                        uri = "mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
                        # Create a new client and connect to the server
                        client = MongoClient(uri, server_api=ServerApi('1'))
                        # Send a ping to confirm a successful connection
                        try:
                            client.admin.command('ping')
                            st.write("Pinged your deployment. You successfully connected to MongoDB!")
                        except Exception as e:
                            st.write(e)

                        # Vectorstore
                        # client = MongoClient("mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")


                        # 데이터베이스 및 컬렉션 설정
                        db_name = "langchain_db"
                        collection_name = "test"
                        atlas_collection = client[db_name][collection_name]
                        vector_search_index = "vector_index"

                        docs = WikipediaLoader(query=is_topic, lang='ko', load_max_docs=20).load()
                        st.write(f"{docs[0].metadata}")
                        

                        # Define a prompt template

                        # Rag
                        text_splitter = RecursiveCharacterTextSplitter()
                        documents = text_splitter.split_documents(docs)

                        # try:
                        #   connection.test.foo.find_one()
                        # except pymongo.errors.OperationFailure as e:
                        #     st.write(e.code)
                        #     st.write(e.details)

                        vector_search = MongoDBAtlasVectorSearch.from_documents(
                            documents=documents,
                            embedding=embeddings,
                            collection=atlas_collection,
                            index_name=vector_search_index
                        )

                        # Instantiate Atlas Vector Search as a retriever
                        retriever = vector_search.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 5, "score_threshold": 0.75}
                        )

                        

                        # PydanticOutputParser 생성
                        parseroub = PydanticOutputParser(pydantic_object=CreateQuizoub)
                        parsersub = PydanticOutputParser(pydantic_object=CreateQuizsub)
                        parsertf = PydanticOutputParser(pydantic_object=CreateQuizTF)

                        prompt = PromptTemplate.from_template(
                            "{input}, Please answer in KOREAN."

                            "CONTEXT:"
                            "{context}."

                            "FORMAT:"
                            "{format}"
                        )
                        promptoub = prompt.partial(format=parseroub.get_format_instructions())
                        promptsub = prompt.partial(format=parsersub.get_format_instructions())
                        prompttf = prompt.partial(format=parsertf.get_format_instructions())

                        document_chainoub = create_stuff_documents_chain(llm, promptoub)
                        document_chainsub = create_stuff_documents_chain(llm, promptsub)
                        document_chaintf = create_stuff_documents_chain(llm, prompttf)

                        retrieval_chainoub = create_retrieval_chain(retriever, document_chainoub)
                        retrieval_chainsub = create_retrieval_chain(retriever, document_chainsub)
                        retrieval_chaintf = create_retrieval_chain(retriever, document_chaintf)

                        for i in range(num_quizzes):
                            try:
                                quiz_questions.append(generate_quiz(quiz_type, is_topic, retrieval_chainoub, retrieval_chainsub,retrieval_chaintf))
                                st.session_state['quizs'] = quiz_questions
                            except OperationFailure as e:
                                st.write(f"Failed to fetch documents: {e}")
                        st.session_state.selected_page = "퀴즈 풀이"
                        st.session_state.selected_type = quiz_type
                        st.session_state.selected_num = num_quizzes

                        st.success('퀴즈 생성이 완료되었습니다!')
                        st.write(quiz_questions)
                        st.session_state['quiz_created'] = True

                if st.session_state.get('quiz_created', False):
                    if st.button('퀴즈 풀기'):
                        st.switch_page("pages/quiz_solve_page.py")


if __name__ == "__main__":
    quiz_creation_page()
