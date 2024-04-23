# Create virtual Env >> python -m venv .yourname_venv
# Command to activate Virt Env >> .stgenai_venv/Scripts/activate
# Command to install requirements >> pip install -r requirements.txt
# Command to exit the venv >> deactivate
# Command to launch >> streamlit run streamlit_app.py

## Next steps
# Create User ID, Session ID - Done
# Update vector store with all FAQ pages - Done
# Check out Streamlit UI re-loading issue, and reduce function calls
# Deploy app on streamlit - Done
# Explore Experiments Datasets in Langfuse, and auto run
# Clean up the Name of Source docs - Done
# Show the actual chunks in the expander - Done
# Add Version and Release to the Handler - Done
# Add user authentication
#https://blog.streamlit.io/streamlit-authenticator-part-1-adding-an-authentication-component-to-your-app/
# Automatic question prompts/suggested questions
# Auto Version / Release numbering
# Response evaluation for scores, and take appropriate action - 
# Re-generate response if negative response
# Router chain with multiple agents
# Try A/B test with MMR type retrival
# Offer Custom Agent builder interface
# Add capability to parse embedded tables, images, videos
# Add capability to input webpages, slack, confluence
# Microservice architecture - break the app front end, conversation serivce, chat histry (database), LLM, Vector store 
#https://www.youtube.com/watch?v=I_4jEnDwGwI&t=334s

# Add Routing, Fallback, Self Correction: https://www.youtube.com/watch?v=-ROS6gfYIts
# Advance retrieval chains such as multiple versions of the question
# Eplore what is a Dense Passage Retrieval DPR



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
from urllib.parse import quote


import subprocess
git_version = subprocess.check_output(["git","describe"]).strip()

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
    feedback_comment = feedback["text"]
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
    return
    


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
        session_id = f"{user_id} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
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
    st.session_state.llm = ChatOpenAI(model = 'gpt-3.5-turbo-0125', temperature = 0.1, callbacks=[st.session_state.langfuse_handler, llmonitor_handler]) 
    llm = st.session_state.llm
    memory = ConversationBufferMemory(memory_key="chat_history", output_key = 'answer',return_messages=True)
    # https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.pinecone.Pinecone.html#langchain_community.vectorstores.pinecone.Pinecone.as_retriever
    st.session_state.retriever=vector_db.as_retriever(search_type = "similarity", search_kwargs={"k": 5})
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type='stuff', 
        retriever=st.session_state.retriever,
        memory=memory, 
        callbacks=[st.session_state.langfuse_handler, llmonitor_handler],
        return_source_documents = True
        )
    st.success("Custom Agent selected: "+index_name)
    return conversation_chain



#### USER INQUIRY ####
def display_chats():
    print("display chats method called: Length of generated is " + str(len(st.session_state['generated'])))
    reply_container = st.container()
    container = st.container()
    notes_container = st.container()
    
    with container:
        st.text_input("Question:", placeholder="Ask a question", key='input', on_change = submit)


    if st.session_state['generated']:
         with reply_container:
             for i in range(len(st.session_state['generated'])):
                with st.chat_message("user"):
                    st.write(st.session_state["past"][i])
                with st.chat_message("assistant"):
                    st.write(st.session_state["generated"][i]) 
                    # If this is the last message, then ask for feedback and print Citations
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
                        st.markdown("***Disclaimer:*** *My responses may be inaccurate. Verify citations. Feel free to ask follow-up questions For eg. 'Tell me more', 'Are you sure?'*")

                        with st.expander("View Citations"):
                            references = get_references()
                            # This needs to be dynamic in future. Right now 5 tabs are hardcoded
                            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Citation 1", "Citation 2", "Citation 3", "Citation 4", "Citation 5"])
                            with tab1:
                                st.write(references[0].get('url'))
                                st.write(references[0].get('page_content'))
                            with tab2:
                                st.write(references[1].get('url'))
                                st.write(references[1].get('page_content'))
                            with tab3:
                                st.write(references[2].get('url'))
                                st.write(references[2].get('page_content'))
                            with tab4:
                                st.write(references[3].get('url'))
                                st.write(references[3].get('page_content'))
                            with tab5:
                                st.write(references[4].get('url'))
                                st.write(references[4].get('page_content'))
    return
     
                            



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
    st.session_state.original_docs = result.get('source_documents')
    st.session_state['past'].append(user_question)
    st.session_state['generated'].append(result["answer"])# + "\n\n" + "References: \n" + unique_ref_text) 
    return

def get_unique_references(src_docs):
    # create list of sources
    src_list = []
    print('\n\n\n ---------------------Source Docs -------------------------------------')
    print(src_docs)
    for doc in src_docs:
        file_name = doc.metadata.get('source').rpartition("\\")[-1].rpartition("/")[-1]
        src_list.append(file_name)
    #deduplicate
    st.session_state.unique_src_list = list(dict.fromkeys(src_list))
    unique_ref_text = "\n".join(st.session_state.unique_src_list)
    return unique_ref_text

def get_references():
    pdf_urls = {
        "Guide - Create Ad Groups Step-4.pdf": "https://advertisinghelp.walmart.com/s/guides?channel=Sponsored&article=000008380",
        "Guide - Suggested Bids.pdf" : "https://advertisinghelp.walmart.com/s/guides?channel=Sponsored&article=000009825",
        "Guide - Custom Reports.pdf" : "https://advertisinghelp.walmart.com/s/guides?channel=Sponsored&article=000009797",
        "Guide - On Demand Reports.pdf" : "https://advertisinghelp.walmart.com/s/guides?channel=Sponsored&article=000009366",
        "Guide - Item Health Reports.pdf" : "https://advertisinghelp.walmart.com/s/guides?channel=Sponsored&article=000009389",
        "Guide - All Campaigns.pdf" : "https://advertisinghelp.walmart.com/s/guides?channel=Sponsored&article=000008410",
        "Guide - Add and Select Keywords - Step 5.pdf" : "https://advertisinghelp.walmart.com/s/guides?channel=Sponsored&article=000008383",
        "Guide - Set Placement Inclusion - Step 2.pdf" : "https://advertisinghelp.walmart.com/s/guides?channel=Sponsored&article=000009457",
        "Guide - Create a campaign - Step 1.pdf" : "https://advertisinghelp.walmart.com/s/guides?channel=Sponsored&article=000008379",
        "Guide - All Keywords.pdf" : "https://advertisinghelp.walmart.com/s/guides?channel=Sponsored&article=000011134",
        "Guide - Add Bid Multipliers - Step 3.pdf" : "https://advertisinghelp.walmart.com/s/guides?channel=Sponsored&article=000008409",
        "Guide - Add Items to Ad Group Step 4 (continued).pdf" : "https://advertisinghelp.walmart.com/s/guides?channel=Sponsored&article=000008382",
        "ALL FAQS - Sponsored Vidoes.pdf" : "https://advertisinghelp.walmart.com/s/faqs?channel=SponsoredVideos&article=Introduction_to_Sponsored_Videos_SponsoredVideos_FAQs",
        "ALL FAQS - Sponsored Brands.pdf" : "https://advertisinghelp.walmart.com/s/faqs?channel=Search%20Brand%20Amplifier&article=Advertising_basics_SBA_FAQs",
        "ALL FAQS - Shop Builder.pdf" : "https://advertisinghelp.walmart.com/s/faqs?channel=ShopBuilder&article=Advertising_basics_ShopBuilder_FAQs",
        "ALL FAQS - Display.pdf" : "https://advertisinghelp.walmart.com/s/faqs?channel=Display&article=Performance_Dashboard_Display_FAQ",
        "ALL FAQS - Sponsored Products.pdf" : "https://advertisinghelp.walmart.com/s/faqs?channel=Sponsored%20Products&article=Item_Health_Report",
        "Guide - Overview of Sponsored Products.pdf" : "https://advertisinghelp.walmart.com/s/guides?channel=Sponsored%20Products&article=000008449",
        "Guide - Advertiser Reports.pdf" : "https://advertisinghelp.walmart.com/s/guides?channel=Sponsored&article=000008390"
    }
    references = []
    for doc in st.session_state.original_docs:
        page_content = doc.dict()["page_content"]
        #src_file = doc.dict()["metadata"]["source"]
        src_file = doc.metadata.get('source').rpartition("\\")[-1].rpartition("/")[-1]
        url = pdf_urls.get(src_file)
        chunk = {"url":url, "page_content": page_content}
        references.append(chunk)
    return references



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
    st.session_state['langfuse_handler'] = CallbackHandler(user_id = user_id, session_id = session_id, version=git_version)
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
                session_id = f"{st.session_state.user_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 
            st.session_state.chain = get_conversation_chain(selected_index)
    if st.sidebar.button("Clear History"):
        st.session_state['history'] = []
        st.session_state['generated'] = ["Hello! Ask me a question 🤗"]
        st.session_state['past'] = ["Hey! 👋"]
    st.sidebar.write("\n\n\n\n")
    st.sidebar.write("### Caution")
    st.sidebar.write(f":red[Not privacy/security approved. Avoid proprietary info]")
    
    st.sidebar.write("This MVP is our starting point 🌱, knowing it won't be flawless. Together, we'll keep improving until it's perfect🚀.")
    slack_url = "https://walmart.enterprise.slack.com/archives/C070HND53Q8"
    st.sidebar.write("Questions or Feedback? Slack: [#wmc-internal-chatbot-mvp](%s)" % slack_url)
    knowledge_scope_url= "https://my.wal-mart.com/:i:/g/personal/r0s01vi_homeoffice_wal-mart_com/EXhNiIc90eBBkdcr5_mW0C0B8mf-BHAyvb4M9gPl1LatXQ?e=AWOI9G"    
    st.sidebar.write("Trained only on [Subset of Help Pages](%s)" % knowledge_scope_url)
    url = "https://my.wal-mart.com/:p:/g/personal/s0m0o96_homeoffice_wal-mart_com/EcexNGllYxxCpDKL1vf2FAEBTbn2sEdodF3w84lo6zPIkQ?e=TkNU2F"
    st.sidebar.write("More details in [Kickoff slide deck](%s)" % url)
    what_is_rag = 'https://www.databricks.com/glossary/retrieval-augmented-generation-rag'
    st.sidebar.write("Built using [RAG architecture](%s)" %what_is_rag)
    st.sidebar.write("Thank you!")
    
    

    display_chats()


if __name__ == '__main__':
    main()


