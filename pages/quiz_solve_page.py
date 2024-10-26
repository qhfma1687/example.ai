#quiz_solve_page.py

import streamlit as st
from langchain_core.pydantic_v1 import BaseModel, Field
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
import io
import json

class CreateQuizoub(BaseModel):
    quiz = "만들어진 문제"
    options1 = "만들어진 문제의 첫 번째 보기"
    options2 = "만들어진 문제의 두 번째 보기"
    options3 = "만들어진 문제의 세 번째 보기"
    options4 = "만들어진 문제의 네 번째 보기"
    correct_answer = "options1 or options2 or options3 or options4"


class CreateQuizsub(BaseModel):
    quiz: str = Field(description="만들어진 문제")
    correct_answer: str = Field(description="만들어진 문제의 답")


class CreateQuizTF(BaseModel):
    quiz: str = Field(description="만들어진 문제")
    options1: str = Field(description="만들어진 문제의 참 또는 거짓인 보기")
    options2: str = Field(description="만들어진 문제의 참 또는 거짓인 보기")
    correct_answer: str = Field(description="만들어진 보기중 하나")

# 퀴즈 채점 함수
@st.experimental_fragment
def grade_quiz_answers(user_answers, quiz_answers):
    graded_answers = []
    for user_answer, quiz_answer in zip(user_answers, quiz_answers):
        if user_answer.lower() == quiz_answer.lower():
            graded_answers.append("정답")
        else:
            graded_answers.append("오답")
    st.session_state['ganswer'] = graded_answers
    return graded_answers

# 파일 처리 함수
@st.experimental_fragment
def process_file(uploaded_file):
    if uploaded_file is None:
        st.warning("파일을 업로드하세요.")
        return None

    # 업로드된 파일 처리
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

    return text_content

# 퀴즈 생성 함수
@st.experimental_fragment
def generate_quiz(quiz_type, text_content):
    response = CreateQuizoub
    response.quiz = '어댑터패턴이나 퍼사드패턴은 무엇을 위해 사용되는가?'
    response.options1 = '인터페이스 호환성 때문에 같이 쓸 수 없는 클래스들을 연결해서 사용할 수 있게 함'
    response.options2 = '복잡한 서브시스템을 더 쉽게 사용할 수 있게 해줌'
    response.options3 = '객체의 인터페이스를 다른 인터페이스로 변환할 때 사용함'
    response.options4 = '상속을 사용하여 서브클래스에 대해서 어댑터 역할을 수행함'
    response.correct_answer = '인터페이스 호환성 때문에 같이 쓸 수 없는 클래스들을 연결해서 사용할 수 있게 함'
    quiz_questions = response

    return quiz_questions

@st.experimental_fragment
def grade_quiz_answer(user_answer, quiz_answer):
    if user_answer.lower() == quiz_answer.lower():
        grade = "정답"
    else:
        grade = "오답"
    return grade


def quiz_solve_page():
    st.title("quiz solve page")
    st.markdown("---")
    
    placeholder = st.empty()
    if 'number' not in st.session_state:
        st.session_state.number = 0
    if 'user_selected_answers' not in st.session_state:
        st.session_state.user_answers = []  # 사용자 선택 답변을 저장할 배열 초기화
    if 'correct_answers' not in st.session_state:
        st.session_state.correct_answers = []  # 정답 여부를 저장할 배열 초기화
    if 'canswer' not in st.session_state:
        st.session_state.canswer = ""
    if 'uanswer' not in st.session_state:
        st.session_state.uanswer = ""
        
    for j, question in enumerate(st.session_state.quizs):
        res = json.loads(question["answer"])
        if st.session_state.number == j:
            with placeholder.container():
                st.header(f"문제 {j+1}")
                # st.write(st.session_state.selected_page)
                # st.write(f"문제 번호: {st.session_state.number + 1}")
                # st.markdown("---")
                
                st.write(f"**{res['quiz']}**")
                st.write("\n")
                st.write("\n")
                
                if st.session_state.selected_type == "주관식":
                    st.session_state.canswer = st.text_input(f"질문 {j + 1}에 대한 답변 입력", key=f"{j}1")
                    st.session_state.uanswer = st.session_state.canswer
                elif st.session_state.selected_type == '다중 선택 (객관식)':
                    options = [res.get('options1'), res.get('options2'), res.get('options3'), res.get('options4')]
                    for index, option in enumerate(options):
                        if st.button(f"{index+1}. {option}", key=f"{j}_{index}"):
                            st.session_state.user_answers.append(option)  # 선택한 답변을 배열에 추가
                            st.session_state.uanswer = option
                elif st.session_state.selected_type == 'OX 퀴즈':
                    options = [res.get('options1'), res.get('options2')]
                    for index, option in enumerate(options):
                        if st.button(f"{index+1}. {option}", key=f"{j}_{index}"):
                            st.session_state.user_answers.append(option)  # 선택한 답변을 배열에 추가
                            st.session_state.uanswer = option
                
                st.markdown("---")
                if st.button("다음", key=f"next{j}"):
                    if res['correct_answer'] == st.session_state.canswer or res['correct_answer'] == st.session_state.uanswer:
                        st.success("정답입니다!")
                        st.session_state.correct_answers.append(True)
                    else:
                        st.error("오답입니다.")
                        st.session_state.correct_answers.append(False)
                    st.session_state.number += 1  # 다음 문제로 이동

        j += 1
    
    if st.session_state.number == st.session_state.selected_num:
        st.session_state['total_score'] = sum(st.session_state.correct_answers)  # 정답 개수를 점수로 저장
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button('결과 확인'):
                st.switch_page("pages/quiz_grading_page.py")

        with col2:
            if st.button('점수 확인'):
                if 'total_score' in st.session_state:
                    st.write(f"최종 점수: {st.session_state['total_score']}")
                else:
                    st.write("아직 점수가 계산되지 않았습니다.")

if __name__ == "__main__":
    quiz_solve_page()
