import streamlit as st
from lida import Manager, TextGenerationConfig, llm
from lida.datamodel import Goal
import os
import pandas as pd
from langchain.callbacks import LLMonitorCallbackHandler
from PIL import Image
from io import BytesIO
import base64
from streamlit_feedback import streamlit_feedback

def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)
    
    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))

# make data dir if it doesn't exist
os.makedirs("data", exist_ok=True)

st.set_page_config(
    page_title="WMC Viz Agents Playground :seedling:",
    page_icon="📊",
)

st.subheader("WMC Data-Insight-Agents Playground :seedling:")


# Step 1 - Get OpenAI API key
openai_key = os.getenv("OPENAI_API_KEY")

selected_model = "gpt-3.5-turbo"
temperature = 0.2
use_cache = True

#st.sidebar.write("### Choose a dataset")
selected_dataset = None

datasets = [
    {"label": "Select an Agent", "url": None},
    {"label": "Cars (example)", "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv"},
    {"label": "Weather (example)", "url": "https://raw.githubusercontent.com/uwdata/draco/master/data/weather.json"},
    {"label": "SP Perf Insights","url": "https://raw.githubusercontent.com/ronaks2309/st-hello-world/main/synthetic.csv"}
]

selected_dataset_label = st.sidebar.selectbox(
    'Choose a Dataset',
    options=[dataset["label"] for dataset in datasets],
    index=0
)




upload_own_data = st.sidebar.checkbox("Upload your own data")

if upload_own_data:
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or JSON file", type=["csv", "json"])

    if uploaded_file is not None:
        # Get the original file name and extension
        file_name, file_extension = os.path.splitext(uploaded_file.name)

        # Load the data depending on the file type
        if file_extension.lower() == ".csv":
            data = pd.read_csv(uploaded_file)
        elif file_extension.lower() == ".json":
            data = pd.read_json(uploaded_file)

        # Save the data using the original file name in the data dir
        uploaded_file_path = os.path.join("data", uploaded_file.name)
        data.to_csv(uploaded_file_path, index=False)

        selected_dataset = uploaded_file_path

        datasets.append({"label": file_name, "url": uploaded_file_path})

        # st.sidebar.write("Uploaded file path: ", uploaded_file_path)
else:
    selected_dataset = datasets[[dataset["label"]
                                    for dataset in datasets].index(selected_dataset_label)]["url"]

if not selected_dataset:
    st.info("To continue, select a dataset from the sidebar on the left or upload your own.")


selected_method = 'default'

secret_password = st.sidebar.text_input("Password", type = "password", placeholder="Enter Password", key='password')
st.sidebar.button("Go")
st.sidebar.write("\n\n\n\n")
st.sidebar.write("### Caution")
st.sidebar.write("Experimental prototype may have bugs")
st.sidebar.write(f":red[NOT SECURE.AVOID PRIVATE DATA]")
st.sidebar.write("Documentation: coming soon...")
st.sidebar.write("Demo: https://youtu.be/FYkxdvGPo0k")
st.sidebar.write("Thank you for testing!")
st.sidebar.write("Questions/Feedback? Reach out to Ronak")

if secret_password != st.secrets.APP_PASSWORD:
    st.warning("Password missing or incorrect")
# Step 3 - Generate data summary
if openai_key and selected_dataset and selected_method and secret_password == st.secrets.APP_PASSWORD:
    handler = LLMonitorCallbackHandler()
    lida = Manager(text_gen=llm("openai", api_key=openai_key))
    textgen_config = TextGenerationConfig(
        n=1,
        temperature=temperature,
        model=selected_model,
        use_cache=use_cache)
            
    st.write("#### Summary")
    # **** lida.summarize *****
    summary = lida.summarize(
        selected_dataset,
        summary_method=selected_method,
        textgen_config=textgen_config)

    if "dataset_description" in summary:
        st.write(summary["dataset_description"])

    if "fields" in summary:
        fields = summary["fields"]
        nfields = []
        for field in fields:
            flatted_fields = {}
            flatted_fields["column"] = field["column"]
            # flatted_fields["dtype"] = field["dtype"]
            for row in field["properties"].keys():
                if row != "samples":
                    flatted_fields[row] = field["properties"][row]
                else:
                    flatted_fields[row] = str(field["properties"][row])
            # flatted_fields = {**flatted_fields, **field["properties"]}
            nfields.append(flatted_fields)
        nfields_df = pd.DataFrame(nfields)
        st.write(nfields_df)
    else:
        st.write(str(summary))

    goals = lida.goals(summary, n=4, textgen_config=textgen_config)

    st.write ("#### Suggested Questions") 
    goal_questions = [goal.visualization for goal in goals]
    st.write(goal_questions)
    #goal_questions.append("----Ask my own question----")
    # selected_goal = st.selectbox('Choose a generated question or ask own', 
    #                     options=goal_questions, 
    #                     index=None,
    #                     placeholder="Choose one"
    #                     )
    #st.text_input("Own Question", label_visibility="collapsed", placeholder= 'Paste a suggested question or ask your own', key = 'user_query')
    
    text_area = st.text_input("Query your Data to Generate Graph")
    if st.button("Generate Graph"):
        if len(text_area) > 0:
            st.info("Your Query: " + text_area)
          
            user_query = text_area
            charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config)  
            with st.expander("See Code"):
                st.code(charts[0].code)
            image_base64 = charts[0].raster
            img = base64_to_image(image_base64)
            st.image(img)
        streamlit_feedback(feedback_type="thumbs",optional_text_label="[Optional] Please provide an explanation",align="flex-start")
        st.download_button("Export Graph", data='''dummy image''', use_container_width=True)
