import streamlit as st
import os

st.write("db_username: ", st.secrets["OPENAI_API_KEY"])

st.write("Has environment variables been set: ",
         os.environ["APP_PASSWORD"])

#st.warning('This is a warning', icon="⚠️")