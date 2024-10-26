import streamlit as st
import boto3

def logout_cognito():
    region_name = 'us-east-1'
    cognito_client = boto3.client('cognito-idp', region_name=region_name)
    access_token = st.session_state.get('access_token')

    try:
        cognito_client.global_sign_out(
            AccessToken=access_token
        )
        st.write("로그아웃되었습니다.")
    except cognito_client.exceptions.NotAuthorizedException:
        st.error("로그아웃 실패: 사용자 인증 토큰이 올바르지 않습니다.")
    except Exception as e:
        st.error(f"로그아웃 중 오류 발생: {str(e)}")

    st.session_state.user = None
    st.session_state.access_token = None

def start():
    placeholder = st.empty()
    if 'user' not in st.session_state:
        st.session_state.user = None

    if st.session_state.user:
        with placeholder.container():
            st.title("퀴즈 이용하러 돌아가기")
            if st.button('퀴즈 생성 바로가기'):
                st.switch_page("pages/quiz_creation_page.py")
            st.title("여기는 로그인한 가입자 전용 서비스입니다.")
            if st.button('로그아웃'):
                logout_cognito()
                st.experimental_rerun()
           
    else:
        with placeholder.container():
            st.title("비회원으로 퀴즈 이용하러 돌아가기")
            if st.button('퀴즈 생성 바로가기'):
                st.switch_page("pages/quiz_creation_page.py")
            
            region_name = 'us-east-1'
            client_id = '7bdv436rrb0l7nhbsva60t7242'
            user_pool_id = 'us-east-1_pJbggBo44'

            st.header("로그인 | ID: admin2 / PW: Admin22!")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                cognito_client = boto3.client('cognito-idp', region_name=region_name)
                try:
                    response = cognito_client.initiate_auth(
                        ClientId=client_id,
                        AuthFlow='USER_PASSWORD_AUTH',
                        AuthParameters={
                            'USERNAME': username,
                            'PASSWORD': password
                        }
                    )
                    st.write(response)  # 디버깅을 위해 전체 응답 출력
                    if 'ChallengeName' in response and response['ChallengeName'] == 'NEW_PASSWORD_REQUIRED':
                        new_password = st.text_input("새 비밀번호를 입력하세요:", type="password")
                        if st.button("비밀번호 변경"):
                            try:
                                challenge_response = cognito_client.respond_to_auth_challenge(
                                    ClientId=client_id,
                                    ChallengeName='NEW_PASSWORD_REQUIRED',
                                    Session=response['Session'],
                                    ChallengeResponses={
                                        'USERNAME': username,
                                        'NEW_PASSWORD': new_password
                                    }
                                )
                                st.write(challenge_response)  # 디버깅을 위해 전체 응답 출력
                                if 'AuthenticationResult' in challenge_response:
                                    authentication_result = challenge_response['AuthenticationResult']
                                    access_token = authentication_result['AccessToken']
                                    st.session_state.user = username
                                    st.session_state.access_token = access_token
                                    st.experimental_rerun()
                                else:
                                    st.error("비밀번호 변경에 실패했습니다.")
                            except Exception as e:
                                st.error(f"비밀번호 변경 중 오류 발생: {str(e)}")
                    elif 'AuthenticationResult' in response:
                        authentication_result = response['AuthenticationResult']
                        access_token = authentication_result['AccessToken']
                        st.session_state.user = username
                        st.session_state.access_token = access_token
                        st.experimental_rerun()
                    else:
                        st.error("인증에 실패했습니다. 사용자 이름 또는 비밀번호를 확인해주세요.")
                except cognito_client.exceptions.NotAuthorizedException:
                    st.error("인증 실패: 사용자 이름 또는 비밀번호가 올바르지 않습니다.")
                except Exception as e:
                    st.error(f"오류 발생: {str(e)}")

            if st.button('회원가입'):
                st.switch_page("pages/sign.py")

if __name__ == "__main__":
    start()
