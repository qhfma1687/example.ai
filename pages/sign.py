#sign

import streamlit as st
import boto3

def register_cognito_user(username, password, email):
    region_name = 'us-east-1'
    client_id = '7bdv436rrb0l7nhbsva60t7242'
    user_pool_id = 'us-east-1_pJbggBo44'
    
    cognito_client = boto3.client('cognito-idp', region_name=region_name)

    try:
        response = cognito_client.sign_up(
            ClientId=client_id,
            Username=username,
            Password=password,
            UserAttributes=[
                {
                    'Name': 'email',
                    'Value': email
                },
            ]
        )
        return True
    except cognito_client.exceptions.UsernameExistsException:
        st.error("회원가입 실패: 이미 존재하는 사용자 이름입니다.")
        return False
    except Exception as e:
        st.error(f"회원가입 중 오류 발생: {str(e)}")
        return False

def sign():
    st.title("회원가입")

    new_username = st.text_input("새 사용자 이름을 입력하세요:")
    new_password = st.text_input("새 비밀번호를 입력하세요:", type="password")
    new_email = st.text_input("이메일 주소를 입력하세요:")

    if st.button("회원가입"):
        if register_cognito_user(new_username, new_password, new_email):
            st.success("회원가입이 완료되었습니다.")
            st.write("운영자가 가입을 승인할 때까지 기다려주세요!")
            st.experimental_rerun()
        else:
            st.error("회원가입에 실패했습니다. 다시 시도하세요.")

if __name__ == "__main__":
    sign()
