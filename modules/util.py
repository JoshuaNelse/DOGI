import streamlit as st


def br(x=1):
    for i in range(x):
        st.write('')


def md(x):
    st.markdown('''{}'''.format(x))
