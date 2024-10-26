#sign.py

import yaml
import streamlit as st
import bcrypt
import hashlib

def register_user(name, username, email, password):
    """
    새로운 사용자를 등록합니다.

    Args:
        username (str): 원하는 사용자 이름.
        password (str): 원하는 비밀번호.

    Returns:
        str: 성공적인 등록 또는 오류 메시지.
    """
    # 비밀번호 해싱
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    # 기존 데이터 읽기
    with open('config.yaml', 'r') as file:
        existing_data = yaml.safe_load(file)
    
    # 새로운 계정 정보 추가
    if username in existing_data['credentials']['usernames']:
        st.error("이미 사용 중인 사용자 이름입니다.")
        return "계정 생성 실패: 이미 사용 중인 사용자 이름입니다."
    else:
        new_data = {
            "credentials": {
                "usernames": {
                    username: {
                        "email": email,
                        "name": name,
                        "password": hashed_password
                    }
                }
            }
        }

        existing_data['credentials']['usernames'].update(new_data['credentials']['usernames'])
        
        # YAML 파일에 쓰기
        with open('config.yaml', 'w') as file:
            yaml.dump(existing_data, file, default_flow_style=False)
        
        st.success("계정이 성공적으로 생성되었습니다!")
        return "계정이 성공적으로 생성되었습니다!"



def login_user(username, password):
    """
    기존 사용자를 로그인합니다.

    Args:
        username (str): 사용자 이름.
        password (str): 비밀번호.

    Returns:
        str: 성공적인 로그인 또는 오류 메시지.
    """
    # 사용자 정보 확인 및 인증
    with open('config.yaml', 'r') as file:
        existing_data = yaml.safe_load(file)
    
    user_info = existing_data['credentials']['usernames'].get(username)
    if user_info is None:
        st.error("잘못된 사용자 이름 또는 비밀번호입니다.")
        return
    if bcrypt.checkpw(password.encode(), user_info["password"].encode()):
        return f"'{username}' 사용자가 성공적으로 로그인되었습니다."
    else:
        st.error("잘못된 사용자 이름 또는 비밀번호입니다.")

def sign():
    st.title("User Registration & Login")
    
    # User registration
    st.header("Register")
    new_name = st.text_input("Enter your name:")
    new_username = st.text_input("Enter a new username:")
    new_email = st.text_input("Enter your email:")
    new_password = st.text_input("Enter a new password:", type="password")
    if st.button("Register"):
        result = register_user(new_name, new_username, new_email, new_password)
        st.success(result)
        
    # User login
    st.header("Login")
    existing_username = st.text_input("Enter your username:", key="username_input")
    existing_password = st.text_input("Enter your password:", type="password", key="password_input")
    if st.button("Login", key="login_button"):
        result = login_user(existing_username, existing_password)
        if result:
            st.success(result)
