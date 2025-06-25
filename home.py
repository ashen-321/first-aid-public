import streamlit as st
import streamlit.components.v1 as components
from audio_recorder_streamlit import audio_recorder
import sys
import os
import io
import asyncio
import re
import json
import random
import magic
from PIL import Image
from io import BytesIO
import base64
import time
import hmac
import numpy as np
import logging
from streamlit_pdf_viewer import pdf_viewer
from concurrent.futures import ThreadPoolExecutor

module_paths = ["./", "./configs", "./backend"]
file_path = os.path.dirname(__file__)
input_file_path = os.path.join(file_path, "input-multimedia")
internal_file_path = os.path.join(file_path, "internal-multimedia")
voice_prompt = ''
tokens = 0
audio_extensions = [".mp3", ".wav"]
image_extensions = [".jpg", ".png", ".webp"]
video_extensions = [".mp4", ".mov"]
document_extensions = [".pdf", ".doc", ".csv", ".json", ".txt", ".xml"]

os.chdir(file_path)

for module_path in module_paths:
    full_path = os.path.normpath(os.path.join(file_path, module_path))
    sys.path.append(full_path)

from utility import *
from utils import *

os.environ["OPENAI_API_KEY"] = api_key = os.getenv("bedrock_api_token")
os.environ["OPENAI_BASE_URL"] = base_url = os.getenv("bedrock_api_url")
os.environ["TAVILY_API_KEY"] = os.getenv("tavily_api_token")
os.environ["GMAPS_API_KEY"] = os.getenv("gmaps_api_token")
os.environ["PERPLEXITY_API_KEY"] = os.getenv('openperplex_api_token')

from agents import *
from tools import *
from orchestration import *


# --------------------------------------------------------------------------------------------
# Webpage Setup ------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------

# Variables
aoss_host = read_key_value(".aoss_config.txt", "AOSS_host_name")
aoss_index = read_key_value(".aoss_config.txt", "AOSS_index_name")
input_image_file = "input_image"
input_audio_file = "input_audio"
input_video_file = "input_video"
input_document_file = "input_document"
last_uploaded_files = None

st.set_page_config(page_title="First Aid",page_icon="🩺",layout="wide")
st.title("Personal assistant")

# Logger
class StreamlitLogHandler(logging.Handler):
    # Initializes a custom log handler with a Streamlit container for displaying logs
    def __init__(self, container):
        super().__init__()
        # Store the Streamlit container for log output
        self.container = container
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])') # Regex to remove ANSI codes
        self.log_area = self.container.empty() # Prepare an empty conatiner for log output

    def emit(self, record):
        msg = self.format(record)
        clean_msg = self.ansi_escape.sub('', msg)  # Strip ANSI codes
        self.log_area.markdown(clean_msg)

    def clear_logs(self):
        self.log_area.empty()  # Clear previous logs

# Set up logging to capture all info level logs from the root logger
def setup_logging():
    root_logger = logging.getLogger() # Get the root logger
    log_container = st.container() # Create a container within which we display logs
    handler = StreamlitLogHandler(log_container)
    handler.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    return handler

# Encapsulate another web page
def vto_encap_web():
    iframe_src = "https://agent.cavatar.info:7861"
    components.iframe(iframe_src)

# Display Non_English charaters
def print_text():
    return st.session_state.user_input.encode('utf-8').decode('utf-8')


# --------------------------------------------------------------------------------------------
# Sidebar ------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


with st.sidebar:
    st.header(':green[Settings]')

    # Model selector
    model_id = st.selectbox('Choose Model',(
        "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        'us.anthropic.claude-3-5-haiku-20241022-v1:0', 
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        'us.amazon.nova-lite-v1:0',
        'us.deepseek.r1-v1:0'
    ))

    # File upload box
    upload_file = st.file_uploader("Upload your image/video/pdfs/docs here:", accept_multiple_files=True, type=["jpg", "png", "webp", "mp4", "mov", "mp3", "wav", "pdf", "doc", "csv", "json", "txt", "xml"])
    file_url = st.text_input("Or input a valid file URL:", key="file_url", type="default")

    # Only update input file directory if something has changed
    if upload_file != last_uploaded_files:
        # File saving
        audio_file_indexes = []
        image_file_indexes = []
        video_file_indexes = []
        document_file_indexes = []
        document_file_names = []
    
        # Clear input file directory
        empty_directory(input_file_path)
    
        # Save file type indexes
        if upload_file is not None:
            for i in range(len(upload_file)):
                _, upload_file_extension = os.path.splitext(upload_file[i].name)
                
                if upload_file_extension in audio_extensions:
                    audio_file_indexes.append(i)
        
                elif upload_file_extension in image_extensions:
                    image_file_indexes.append(i)
        
                elif upload_file_extension in video_extensions:
                    video_file_indexes.append(i)
                
                elif upload_file_extension in document_extensions:
                    document_file_indexes.append(i)
    
        # Read file indexes and save accordingly
        # Audio upload
        for i in range(len(audio_file_indexes)):
            index = audio_file_indexes[i]
            audio_bytes = upload_file[index].read()
            st.audio(audio_bytes, format="audio/wav")
            
            input_file = os.path.join(input_file_path, input_audio_file + f"_{i}" + upload_file_extension)
            with open(input_file, 'wb') as audio_file:
                audio_file.write(audio_bytes)
    
        # Image upload
        for i in range(len(image_file_indexes)):
            index = image_file_indexes[i]
            image_bytes = upload_file[index].read()
            st.image(io.BytesIO(image_bytes))
            
            input_file = os.path.join(input_file_path, input_image_file + f"_{i}" + upload_file_extension)
            with open(input_file, 'wb') as image_file:
                image_file.write(image_bytes)
    
        # Video upload
        for i in range(len(video_file_indexes)):
            index = video_file_indexes[i]
            video_bytes = upload_file[index].getvalue()
            st.video(video_bytes)
            
            input_file = os.path.join(input_file_path, input_video_file + f"_{i}" + upload_file_extension)
            with open(input_file, 'wb') as video_file:
                video_file.write(video_bytes)
    
        # Document upload
        for index in document_file_indexes:
            document = upload_file[index]
            doc_bytes = document.getvalue()
            document_file_names.append(document.name)
            full_file_name = os.path.join(input_file_path, document.name)
            
            if not os.path.exists(full_file_name):
                with open(full_file_name, 'wb') as f:
                    f.write(doc_bytes)
        
            if is_pdf(full_file_name):
                pdf_viewer(input=doc_bytes, width=1200)
            
            elif 'json' in document.name and isinstance(doc_bytes, bytes):
                string_data = doc_bytes.decode('utf-8')
                json_data = json.loads(string_data)
                st.json(json_data)
            
            else:
                st.write(doc_bytes[:1000]+"......".encode())
    
        rag_data = ''
        for file_name in document_file_names:
            full_file_name = os.path.join(input_file_path, file_name)
            
            if is_pdf(full_file_name):    
                texts, tables = parser_pdf(file_path, file_name)
                xml_text = parse_pdf_to_xml(full_file_name)
                rag_data += xml_text
            else:
                with open(full_file_name, 'rb') as file:
                    file_content = file.read()
                rag_data += file_content.decode()
    
        last_uploaded_files = upload_file


# --------------------------------------------------------------------------------------------
# GUI ----------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------


start_time = time.time()

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "I am your assistant. How can I help today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

def sync_generator_from_async(async_gen):
    yield from asyncio.run(anext(async_gen))


if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Send config information
    set_agent_config({
        "model_id": model_id,
        "internal_file_path": internal_file_path
    })

    # Initiate orchestration
    events = Orchestration().invoke(prompt)

    # Parse text
    st.write_stream(events)
    # sync_events = sync_generator_from_async(events)
    # write_data = []
    # for event in sync_events:
    #     write_data.append(event)
    # response = write_data[-1].popitem()[1]['messages']
    # if type(response) is list:
    #     response = response[-1].content

    # Display multimedia
    for file in os.listdir(internal_file_path):
        _, file_extension = os.path.splitext(file)
        file = os.path.join(internal_file_path, file)
        
        if file_extension in image_extensions:
            st.image(Image.open(file), output_format="png", use_container_width=True)
    
    # Display text
    response_formatted = f"{response}\n\n ✒︎***Content created by using:*** {model_id}, Latency: {(time.time() - start_time) * 1000:.2f} ms, Tokens In: {estimate_tokens(prompt, method='max')}, Out: {estimate_tokens(response, method='max')}"
    st.session_state.messages.append({"role": "assistant", "content": response_formatted})
    st.chat_message("ai", avatar='🤵').write(response_formatted)
