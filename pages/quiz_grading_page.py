import streamlit as st
from langchain_openai import ChatOpenAI
import json
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.docstore.document import Document
from langchain.document_loaders import ApifyDatasetLoader
from langchain.indexes import VectorstoreIndexCreator

def grade_quiz_answers(user_answers, correct_answers):
    graded_answers = []
    for user_answer, correct_answer in zip(user_answers, correct_answers):
        if user_answer == correct_answer:
            graded_answers.append('정답')
        else:
            graded_answers.append('오답')
    return graded_answers

def get_explanation(question, correct_answer):
    user_answers = st.session_state.get('user_selected_answers', [])
    correct_answers = st.session_state.get('correct_answers', [])
    questions = st.session_state.get('quizs', [])
    graded_answers = grade_quiz_answers(user_answers, correct_answers)
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    prompt = (
        f"문제: {question}\n"
        f"정답: {correct_answer}\n"
        "이 문제의 해설을 제공해주세요. "
        "해설은 왜 정답이 맞는지 설명해주세요."
    )
    response = llm(prompt)
    return response

def quiz_grading_page():
    user_answers = st.session_state.get('user_selected_answers', [])
    correct_answers = st.session_state.get('correct_answers', [])
    questions = st.session_state.get('quizs', [])

    # 세션 상태 초기화
    if st.session_state.number >= len(questions):
        st.session_state.number = 0
    if 'number' not in st.session_state:
        st.session_state.number = 0
    if 'quizs' not in st.session_state or not st.session_state.quizs:
        st.warning("퀴즈가 없습니다. 먼저 퀴즈를 풀어주세요.")
        return
    
    graded_answers = grade_quiz_answers(user_answers, correct_answers)
    st.title("quiz review page")
    st.markdown("---")
    total_score = 0

    current_question_index = st.session_state.number
    
    # 질문 리스트의 길이를 확인
    if current_question_index >= len(questions):
        st.warning("유효하지 않은 질문 인덱스입니다.")
        return
    
    question = questions[current_question_index]
    res = json.loads(question["answer"])
    
    st.subheader(f"문제 {current_question_index + 1}")
    st.write("\n")
    st.write(f"{res['quiz']}")
    
    if 'options1' in res:
        st.write(f"1. {res['options1']}")
        st.write(f"2. {res['options2']}")
        st.write(f"3. {res['options3']}")
    if 'options4' in res:
        st.write(f"4. {res['options4']}")
        st.write("\n")
    
        st.write(f"정답 : {res['correct_answer']}")
    
    # explanation = get_explanation(res['quiz'], res['correct_answer'])
    # st.write(f"해설: {explanation}")
    # st.markdown("---")
    
    col1, col2= st.columns(2)
    with col1:
        if st.button("이전 문제"):
            if st.session_state.number > 0:
                st.session_state.number -= 1  # 이전 문제로 이동
            else:
                st.warning("첫 번째 문제입니다.")
    with col2:
        if st.button("다음 문제"):
            if st.session_state.number < len(st.session_state.quizs) - 1:
                st.session_state.number += 1  # 다음 문제로 이동
            else:
                st.warning("마지막 문제입니다.")

if __name__ == "__main__":
    quiz_grading_page()
