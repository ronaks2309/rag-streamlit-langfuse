# Create virtual Env >> python -m venv .yourname_venv
# Command to activate Virt Env >> yourname_venv/Scripts/activate
# Command to install requirements >> pip install -r requirements.txt
# Command to exit the venv >> deactivate
# Command to launch >> streamlit run app.py

import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os 
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone  
from pinecone import Pinecone as pinecone_Pinecone
from pinecone import ServerlessSpec

#### PREPARATION #### 
def create_vector_index_from_pdf(uploaded_files,password):
    print("get_vector_index_from_pdf called")
    if password != st.secrets.APP_PASSWORD: 
        st.warning("Incorrect Password")
        return
    if check_index() != "Valid":
        st.warning("Agent Name Not Valid")
        return
    data_folder = save_files(uploaded_files)
    text_chunks = get_chunks(data_folder)
    vector_db = get_vector_db(text_chunks)
    return vector_db

def save_files(uploaded_files):
    # delete old files
    for filename in os.listdir("data"):
        if os.path.isfile(os.path.join('data', filename)):
            os.remove(os.path.join('data', filename))
    # save files by looping over the uploaded files
    for file in uploaded_files:
        with open(os.path.join(os.getcwd(),"data", file.name), "wb") as f:
            f.write(file.getbuffer())
        st.success("Saved File to Data: "+file.name)
    return("data")

def get_chunks(data_folder):
    print("get chunks called")
    loader = DirectoryLoader(os.path.join(os.getcwd(),data_folder), loader_cls = PyPDFLoader)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(pages)
    return text_chunks

def get_vector_db(text_chunks):
    print("get vector db called")
    embeddings = OpenAIEmbeddings()
    pc = pinecone_Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    new_index = st.session_state.new_index
    pc.create_index(name=new_index, metric="cosine", dimension=1536,
                    spec=ServerlessSpec(cloud='aws', region='us-west-2') 
                    )
    pinecone_db = Pinecone.from_documents(text_chunks, embeddings, index_name=new_index)
    return pinecone_db

def check_index():
    pc = pinecone_Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index_name = st.session_state.new_index
    if index_name == "":
        st.warning('Agent name cannot be blank', icon="⚠️")
        return "Invalid"
    elif index_name in pc.list_indexes():
        st.warning('Agent already exists', icon="⚠️")
        return "Invalid"
    else: 
        st.success('Agent name available :white_check_mark:')
        return "Valid"

def main():
    print("Main method called")
    load_dotenv()
    st.set_page_config(page_title="Custom Agents Builder", page_icon = ":books:")
    sideb = st.sidebar
    #st.title("Create a vector index from knowledge files :database:")
    st.subheader("Build your Custom Agent")
    
    col1, col2 = st.columns(2)
    col1.text_input("Agent Name", label_visibility='visible', placeholder="Enter name for New Agent", key='new_index')
    #col2.button("Check Availability", on_click= check_index)
    
    #st.text_input("Agent Description", placeholder="Enter a short description of your agent")
    
    #st.text_area("Instructions",placeholder="You are a courteous and helpful customer support agent. Your replies should strictly be based on the contex provided. If you don't know the answer, guide the user to speak with a WMC Associate. Never give financial advice.")

    src = st.radio("Domain Knowledge", ["Web/Confluence", "Database", "Onedrive", "Slack", "JIRA", "Custom (advance)"], horizontal = True)
    
    if src:
        st.info(f":red[Please select Onedrive & upload PDF(s). Multiple files are okay. Other sources are WIP]")
    uploaded_files = st.file_uploader("Provide the path to "+ src, accept_multiple_files=True, key = 1, type = 'pdf')
    
    st.text_input("Enter Password", type = "password", placeholder="Enter Password", key='password')   

    with st.expander("Advance Settings (optional)"):
        st.write(f":red[Advance Settings are currently not functional. Only placeholders]")
        st.text_area("Instructions",placeholder="You are a courteous and helpful customer support agent. Your replies should strictly be based on the contex provided. If you don't know the answer, guide the user to speak with a WMC Associate. Never give financial advice.")
        st.text_area("Conversation Starters",placeholder="How to reset password?,\nHow to setup search campaign?")
        st.radio("Element LLM Model", ["gpt 3.5 turbo", "gpt 4", "Llama 2", "Mistral 7B", "Custom"], horizontal = True, key = 'llm')
        st.checkbox("Enable function calling")
        st.checkbox("Enable External APIs")
 



    st.write('')
    st.write('')
    st.write('')
    st.checkbox("I have read the Walmart  GenAI Security & Legal Policies. <placeholder for URL to Legal policies>")

    if st.button("Create"):
        with st.spinner("Processing"):
            pinecone_index = create_vector_index_from_pdf(uploaded_files,st.session_state.password)
            st.success("Agent created successfully !!")
    
    st.sidebar.write("\n\n\n\n")
    st.sidebar.write("### Caution")
    st.sidebar.write("Experimental prototype may have bugs")
    st.sidebar.write(f":red[NOT SECURE.AVOID PRIVATE DATA]")
    st.sidebar.write("Documentation: coming soon...")
    st.sidebar.write("Demo: https://youtu.be/FYkxdvGPo0k")
    st.sidebar.write("Thank you for testing!")
    st.sidebar.write("Questions/Feedback? Reach out to Ronak")




    # hide_label = (
    #     """
    # <style>
    #     div[data-testid="stFileUploader"]>section[data-testid="stFileUploadDropzone"]>button[data-testid="baseButton-secondary"] {
    #     color:white;
    #     }
    #     div[data-testid="stFileUploader"]>section[data-testid="stFileUploadDropzone"]>button[data-testid="baseButton-secondary"]::after {
    #         content: "BUTTON_TEXT";
    #         color:black;
    #         display: block;
    #         position: absolute;
    #     }
    #     div[data-testid="stFileDropzoneInstructions"]>div>span {
    #     visibility:hidden;
    #     }
    #     div[data-testid="stFileDropzoneInstructions"]>div>span::after {
    #     content:"INSTRUCTIONS_TEXT";
    #     visibility:visible;
    #     display:block;
    #     }
    #     div[data-testid="stFileDropzoneInstructions"]>div>small {
    #     visibility:hidden;
    #     }
    #     div[data-testid="stFileDropzoneInstructions"]>div>small::before {
    #     content:"FILE_LIMITS";
    #     visibility:visible;
    #     display:block;
    #     }
    # </style>
    # """.replace("BUTTON_TEXT", "")
    #     .replace("INSTRUCTIONS_TEXT", "Instr")
    #     .replace("FILE_LIMITS", "Limits")
    # )

    # st.markdown(hide_label, unsafe_allow_html=True)

    # file_uploader = st.file_uploader(label="Upload a file")


if __name__ == '__main__':
    main()


