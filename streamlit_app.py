# Create virtual Env >> python -m venv .yourname_venv
# Command to activate Virt Env >> .stgenai_venv/Scripts/activate
# Command to install requirements >> pip install -r requirements.txt
# Command to exit the venv >> deactivate
# Command to launch >> streamlit run streamlit_app.py

## Next steps
# Create User ID, Session ID - Done
# Update vector store with all FAQ pages
# Check out Streamlit UI re-loading issue, and reduce function calls
# Deploy app on streamlit
# Explore Experiments Datasets in Langfuse
# Clean up the Name of Source docs
# Show the actual chunks in the expander
# Add Version and Release to the Handler
# Add user authentication
#https://blog.streamlit.io/streamlit-authenticator-part-1-adding-an-authentication-component-to-your-app/

import warnings

# Filter out deprecation warnings from langchain
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")


import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os 
#from langchain.document_loaders import DirectoryLoader, PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.callbacks import LLMonitorCallbackHandler
from langchain_community.vectorstores import Pinecone  
from pinecone import Pinecone as pinecone_Pinecone
from pinecone import ServerlessSpec
from streamlit_feedback import streamlit_feedback

from langfuse.callback import CallbackHandler
from langfuse import Langfuse
import uuid
from datetime import datetime


print("Helloworld")

st.session_state.feedback_type = "faces"

def add_feedback_to_trace(feedback):
    print("add_feedback_method_called")
    langfuse = Langfuse()
    # Capture User Feedback with the trace as a score
    #https://langfuse.com/docs/integrations/langchain/example-python
    #https://docs.smith.langchain.com/cookbook/feedback-examples/streamlit
    score_mappings = {
        "thumbs": {"👍": 1, "👎": 0},
        "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
    }
    # Get the score mapping based on the selected feedback option
    scores = score_mappings[st.session_state.feedback_type]
    # Get the score from the selected feedback option's score mapping
    feedback_value = scores.get(feedback["score"])

    #feedback_value = 1 if feedback["score"]=="👍" else 0  
    # For future, add faces instead of thumbs for a more granular feedback
    feedback_comment = feedback["text"]
    #trace_id = "f4e4baa5-5e3c-4083-8b4e-756e240a23b9"
    trace_id = st.session_state.my_trace_id
    print("Trace ID received by add_feedback_to_trace method is: ")
    print(trace_id)
    # Add score, e.g. via the Python SDK
    trace = langfuse.score(
        trace_id=trace_id,
        name="user-feedback",
        value=feedback_value,
        comment=feedback_comment
    )
    st.toast("Thank you! Your feedback has been recorded ✅")




def initialize_session_state():
    print("initialize session state method called")
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me a question 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

    if 'langfuse_handler' not in st.session_state:
        user_id = "Not Assigned"
        session_id = f'{user_id} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}'
        reset_langfuse_handler(user_id = user_id, session_id = session_id)
        print("languse handler declared!")
    
    if 'agent_list' not in st.session_state: 
        st.session_state.agent_list = get_pinecone_index_list()
    


def get_conversation_chain(selected_index):
    print("get conversation chain called")
    #check for password
    if st.session_state.password != st.secrets.APP_PASSWORD:
        st.warning("Incorrect Password")
        return
    if 'user_name' not in st.session_state:
        st.warning("Please enter your name")
        return
    elif st.session_state.user_name.strip() == '':
        st.warning("Please enter your name")
        return
    
    llmonitor_handler = LLMonitorCallbackHandler()
    print("Trace ID: ")
    print(st.session_state.langfuse_handler.get_trace_id())
    embeddings = OpenAIEmbeddings()
    pc = pinecone_Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index_name = selected_index
    vector_db = Pinecone.from_existing_index(index_name, embeddings)
    llm = ChatOpenAI(model = 'gpt-3.5-turbo-0125', temperature = 0.1, callbacks=[st.session_state.langfuse_handler, llmonitor_handler]) 
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.pinecone.Pinecone.html#langchain_community.vectorstores.pinecone.Pinecone.as_retriever
    st.session_state.retriever=vector_db.as_retriever(search_type = "similarity", search_kwargs={"k": 5})
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type='stuff', 
        retriever=st.session_state.retriever,
        memory=memory, 
        callbacks=[st.session_state.langfuse_handler, llmonitor_handler] 
        #return_source_documents = True
        )
    st.success("Custom Agent selected: "+index_name)
    return conversation_chain



#### USER INQUIRY ####
def display_chats():
    print("display chats method called")
    reply_container = st.container()
    container = st.container()
    notes_container = st.container()
    
    with container:
        st.text_input("Question:", placeholder="Ask a question", key='input', on_change = submit)


    # if st.session_state['generated']:
    #     with reply_container:
    #         for i in range(len(st.session_state['generated'])):
    #             message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
    #             message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji") 
    

    if st.session_state['generated']:
         with reply_container:
             for i in range(len(st.session_state['generated'])):
                with st.chat_message("user"):
                    st.write(st.session_state["past"][i])
                with st.chat_message("assistant"):
                    st.write(st.session_state["generated"][i]) 
                    if i == len(st.session_state['generated'])-1 and i>0:
                        feedback_key = f"feedback_key_{i}"
                        print(f"The Feedback key is :" + feedback_key)
                        streamlit_feedback(
                                            key = feedback_key,
                                            feedback_type=st.session_state.feedback_type,
                                            optional_text_label="[Optional] Please provide an explanation",
                                            align="flex-start",
                                            on_submit=add_feedback_to_trace,
                                            )

def submit():
    print ("Submit method called")
    st.session_state.user_question = st.session_state.input
    st.session_state.input = ""
    with st.spinner('Generating Response...'):
        handle_user_question(st.session_state['user_question'])
    return

def handle_user_question(user_question):
    print("handle user question called")
    if 'chain' not in st.session_state:
        st.warning("No Agent is selected. Please select agent, enter user name, password and press go")
        return
    print("Trace ID from handle_user_question method, before invoking chain:")
    print(st.session_state.langfuse_handler.get_trace_id()) 
    mychain = st.session_state.chain
    result = mychain({"question": user_question, "chat_history": st.session_state['history']})
    st.session_state.langfuse_handler.flush()
    print("Trace ID from handle_user_question method after invoking chain:")
    st.session_state.my_trace_id = st.session_state.langfuse_handler.get_trace_id()
    print(st.session_state.langfuse_handler.get_trace_id()) # This is working!
    st.session_state['history'].append((user_question, result["answer"]))
    src_docs = st.session_state.retriever.get_relevant_documents(user_question)
    unique_ref_text = get_unique_references(src_docs)
    st.session_state['past'].append(user_question)
    st.session_state['generated'].append(result["answer"] + "\n\n" + "To Learn more, visit: \n" + unique_ref_text) 
    return

def get_unique_references(src_docs):
    # create list of sources
    src_list = []
    for doc in src_docs:
        src_list.append(doc.metadata.get('source').rpartition("\\")[-1])
    #deduplicate
    unique_src_list = list(dict.fromkeys(src_list))
    unique_ref_text = "\n".join(unique_src_list)
    return unique_ref_text

def get_pinecone_index_list():
   pc = pinecone_Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
   index_names=[]
   active_indexes = pc.list_indexes()
   for indexes in active_indexes:
       print(indexes['name'])
       index_names.append(indexes['name']) 
   return index_names
    

def reset_langfuse_handler (user_id, session_id):
    print("Reset Langfuse handler method called")
    if 'langfuse_handler' in st.session_state:
        del st.session_state.langfuse_handler
    st.session_state['langfuse_handler'] = CallbackHandler(user_id = user_id, session_id = session_id)
    print("Langfuse Handler reset with user: " + user_id + "session Id: " + session_id)


def main():
    print("Main method called")
    load_dotenv()
    initialize_session_state()
    #get_conversation_chain('wmc-faq')
    st.set_page_config(page_title="WMC GenAI Playground", page_icon = ":seedling:")
    st.subheader("WMC Chat-Agents Playground :seedling:")
    sideb = st.sidebar
    #st.sidebar.title("Select Pinecone Index")
    selected_index = sideb.selectbox(
        "Select an agent",
        # Need to fetch this list from the Pinecone Index 
        options = st.session_state.agent_list , #["wmc-faq", "wmc-data-gov", "wmc-sales-presentations","wmc-creatives-builder", "wmc-test1"],
        index=None,
        placeholder="Choose agent")
    sideb.text_input("Your Name", placeholder="Enter your Name", key='user_name')
    sideb.text_input("Password", type = "password", placeholder="Enter Password", key='password')
    if st.sidebar.button("Go"):
        with st.spinner("Connecting to vector dB"):
            reset_langfuse_handler(user_id = st.session_state.user_name, 
                session_id = f'{st.session_state.user_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}') 
            st.session_state.chain = get_conversation_chain(selected_index)
    if st.sidebar.button("Clear History"):
        st.session_state['history'] = []
        st.session_state['generated'] = ["Hello! Ask me a question 🤗"]
        st.session_state['past'] = ["Hey! 👋"]
    st.sidebar.write("\n\n\n\n")
    st.sidebar.write("### Caution")
    st.sidebar.write("Experimental prototype may have bugs")
    st.sidebar.write(f":red[NOT SECURE.AVOID PRIVATE DATA]")
    st.sidebar.write("Documentation: coming soon...")
    st.sidebar.write("Demo: https://youtu.be/FYkxdvGPo0k")
    st.sidebar.write("Thank you for testing!")
    st.sidebar.write("Questions? Slack channel: TBD")
    


    

    display_chats()


if __name__ == '__main__':
    main()


