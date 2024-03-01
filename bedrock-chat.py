import boto3
from anthropic import Anthropic
from botocore.config import Config
import shutil
import os
import pandas as pd
import json
import io
from langchain.llms.bedrock import Bedrock
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.llms.bedrock import Bedrock
from langchain.callbacks.base import BaseCallbackHandler
import time
config = Config(
    read_timeout=600,
    retries = dict(
        max_attempts = 5 ## Handle retries
    )
)
import re
import numpy as np
import openpyxl
import inflect
p = inflect.engine()
from openpyxl.cell import Cell
from openpyxl.worksheet.cell_range import CellRange
from textractor import Textractor
from textractor.visualizers.entitylist import EntityList
from textractor.data.constants import TextractFeatures
from textractor.data.text_linearization_config import TextLinearizationConfig
from boto3.dynamodb.conditions import Key 

# Read credentials
with open('config.json') as f:
    config_file = json.load(f)
DYNAMODB_TABLE=config_file["DynamodbTable"]
DYNAMODB_USER= config_file["UserId"]
BUCKET=config_file["Bucket_Name"]
OUTPUT_TOKEN=config_file["max-output-token"]
S3 = boto3.client('s3')
DYNAMODB  = boto3.resource('dynamodb')
st.set_page_config(initial_sidebar_state="collapsed")


if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'token' not in st.session_state:
    st.session_state['token'] = 0
if 'chat_hist' not in st.session_state:
    st.session_state['chat_hist'] = []
if 'user_sess' not in st.session_state:
    st.session_state['user_sess'] = f"{DYNAMODB_USER}-{str(time.time()).split('.')[0]}"
if 'chat_session_list' not in st.session_state:
    st.session_state['chat_session_list'] = []
if 'count' not in st.session_state:
    st.session_state['count'] = 0


def handle_doc_upload_or_s3(file):
    """
    Handle parsing of documents from local file system or S3.
    Supports PDF, PNG, JPG, CSV, XLSX, JSON and Text files.
    Parameters:
        file (str): Path or S3 URI of document to parse
    Returns:
        content: Parsed contents of the file in appropriate format
    """
    
    dir_name, ext = os.path.splitext(file)
    if  ext in [".pdf", ".png", ".jpg"]:   
        content=exract_pdf_text_aws(file)
    elif "csv"  in ext:
        content= pd.read_csv(file)
    elif ext in [".xlsx", ".xlx"]:
        content=table_parser_utills(file)
    elif "json" in ext and "s3://" not in dir_name:
        # with open(file) as json_file:       
        content = json.load(file)
    elif  "json" in ext and "s3://" in dir_name:
        s3 = boto3.client('s3')
        match = re.match("s3://(.+?)/(.+)", file)
        if match:
            bucket_name = match.group(1)
            key = match.group(2)    
            obj = s3.get_object(Bucket=bucket_name, Key=key)        
            content = json.loads(obj['Body'].read())
    elif ext in [".txt",".py",".yml"]  and "s3://" not in dir_name:
        with open(file, "r") as txt_file:       
            content = txt_file.read()
    elif  ext in [".txt",".py",".yml"]  and "s3://" in dir_name:
        s3 = boto3.client('s3')
        match = re.match("s3://(.+?)/(.+)", file)
        if match:
            bucket_name = match.group(1)
            key = match.group(2)    
            obj = s3.get_object(Bucket=bucket_name, Key=key)        
            content = obj['Body'].read()
    # Implement any of file extension logic 
    return content


def exract_pdf_text_aws(file):
    """
    Extract text from PDF/image files using Amazon Textract service.
    Supports PDFs/Images stored locally or in S3.    
    Parameters:
        file (str): Path or S3 URI of PDF file
    Returns:
        text (str): Extracted text from PDF
    """     
    file_base_name=os.path.basename(file)
    # Checking if extracted doc content is in S3
    if [x for x in get_s3_keys("extracted_output/") if file_base_name in x]:      
        response = S3.get_object(Bucket=BUCKET, Key=f"extracted_output/{file_base_name}.txt")
        text = response['Body'].read()
        return text
    
    else:
        dir_name, ext = os.path.splitext(file)
        extractor = Textractor(region_name="us-east-1")
        # Asynchronous call, you will experience some wait time. Try caching results for better experience
        if "s3://" in file and "pdf" in ext:
            logga=st.empty()
            logga.write("Asynchronous Textract call, you may experience some wait time.")
            document = extractor.start_document_analysis(
            file_source=file,
            features=[TextractFeatures.LAYOUT,TextractFeatures.TABLES],       
            save_image=False,   
            s3_output_path=f"s3://{BUCKET}/textract_output/" #S3 location to store textract json response
        )    
            logga.write("")
        #Synchronous call
        else:
            document = extractor.analyze_document(
            file_source=file,
            features=[TextractFeatures.LAYOUT,TextractFeatures.TABLES],  
            save_image=False,
        )

        config = TextLinearizationConfig(
        hide_figure_layout=True,   
        hide_header_layout=False,    
        table_prefix="<table>",
        table_suffix="</table>",
        )    
        # Store extracted text in S3 
        S3.put_object(Body=document.get_text(config=config), Bucket=BUCKET, Key=f"extracted_output/{file_base_name}.txt")      
        return document.get_text(config=config)

def strip_newline(cell):
    """
    A utility function to strip newline characters from a cell.
    Parameters:
    cell (str): The cell value.
    Returns:
    str: The cell value with newline characters removed.
    """
    return str(cell).strip()

def table_parser_utills(file):    
    """
    Converts an Excel table to a csv string, 
    handling duplicated values across merged cells.
    Args:
        file: Excel table  
    Returns: 
        Pandas DataFrame representation of the Excel table
    """
    # Read from S3 or local
    if "s3://" in file:
        s3 = boto3.client('s3')
        match = re.match("s3://(.+?)/(.+)", file)
        if match:
            bucket_name = match.group(1)
            key = match.group(2)    
            obj = s3.get_object(Bucket=bucket_name, Key=key)  
        # Read Excel file from S3 into a buffer
        xlsx_buffer = io.BytesIO(obj['Body'].read())
        xlsx_buffer.seek(0) 
        # Load workbook, get active worksheet
        wb = openpyxl.load_workbook(xlsx_buffer)
        worksheet = wb.active
    else:
        # Load workbook, get active worksheet
        wb = openpyxl.load_workbook(file)
        worksheet = wb.active
    # Unmerge cells, duplicate merged values to individual cells
    all_merged_cell_ranges: list[CellRange] = list(
            worksheet.merged_cells.ranges
        )
    for merged_cell_range in all_merged_cell_ranges:
        merged_cell: Cell = merged_cell_range.start_cell
        worksheet.unmerge_cells(range_string=merged_cell_range.coord)
        for row_index, col_index in merged_cell_range.cells:
            cell: Cell = worksheet.cell(row=row_index, column=col_index)
            cell.value = merged_cell.value
    # determine table header index
    df = pd.DataFrame(worksheet.values)
    df=df.map(strip_newline)  
    return df.to_csv(sep="|", index=False, header=0)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:  
        self.text+=token+""        
        self.container.code(self.text)

def put_db(messages, params):
    """Store long term chat history in DynamoDB"""    
    chat_item = {
        "UserId": DYNAMODB_USER, # user id
        "SessionId": params["chat_id"], # User session id
        "messages": [messages]  # 'messages' is a list of dictionaries
    }
    existing_item = DYNAMODB.Table(DYNAMODB_TABLE).get_item(Key={"UserId": DYNAMODB_USER, "SessionId":params["chat_id"]})
    if "Item" in existing_item:
        existing_messages = existing_item["Item"]["messages"]
        chat_item["messages"] = existing_messages + [messages]

    response = DYNAMODB.Table(DYNAMODB_TABLE).put_item(
        Item=chat_item
    ) 
    
def get_s3_keys(prefix):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    keys=""
    if "Contents" in response:
        keys = []
        for obj in response['Contents']:
            key = obj['Key']
            name = key[len(prefix):]
            keys.append(name)
    return keys

def get_chat_historie(params):
    """
    This function retrieves chat history stored in a dynamoDB table partitioned by a userID and sorted by a SessionID
    """
    current_chat=""
    if DYNAMODB_TABLE:
        chat_histories = DYNAMODB.Table(DYNAMODB_TABLE).get_item(Key={"UserId": DYNAMODB_USER, "SessionId":params["chat_id"]})
        if "Item" in chat_histories:
            #Only return the last 10 chat conversations
            st.session_state['chat_hist']=chat_histories['Item']['messages'][-10:]
            for chat in chat_histories['Item']['messages'][-10:]:
                for k, v in chat.items():
                    current_chat+=v
 
    elif st.session_state['chat_hist']:
        st.session_state['chat_hist']=st.session_state['chat_hist'][-10:]
        for chat in st.session_state['chat_hist']:
            for k, v in chat.items():
                current_chat+=v
    return current_chat

def get_session_ids_by_user(table_name, user_id):
    """
    Get Session Ids and corresponding top message for a user to populate the chat history drop down on the front end
    """
    table = DYNAMODB.Table(table_name)
    message_list={}
    session_ids = []
    args = {
        'KeyConditionExpression': Key('UserId').eq(user_id)
    }
    while True:
        response = table.query(**args)
        session_ids.extend([item['SessionId'] for item in response['Items']])
        if 'LastEvaluatedKey' not in response:
            break
        args['ExclusiveStartKey'] = response['LastEvaluatedKey']
        
    for session_id in session_ids:
        try:
            message_list[session_id]=DYNAMODB.Table(table_name).get_item(Key={"UserId": user_id, "SessionId":session_id})['Item']['messages'][0]['user']
        except Exception as e:
            print(e)
            pass
    return message_list

def query_llm(params, handler):
    import json       
    bedrock_runtime = boto3.client(service_name='bedrock-runtime',region_name='us-east-1',config=config)
    # Get chat history from DynamoDB Table
    current_chat=get_chat_historie(params) 
    # Store uploaded file in S3
    if params['file']:
        doc=''
        for docs in params['file']:
            file_name=str(docs.name)
            file_path=f"file_store/{file_name}"
            S3.put_object(Body=docs.read(),Bucket= BUCKET, Key=file_path)
            params['file_name']=f"s3://{BUCKET}/{file_path}"
            text=handle_doc_upload_or_s3(params['file_name'])
            doc_name=os.path.basename(params['file_name'])
            doc+=f"<{doc_name}>\n{text}\n</{doc_name}>\n"    
        # Chat template for document query
        with open("prompt/doc_chat.txt","r") as f:
            chat_template=f.read()
        values = {
        "doc": doc,
        "prompt": params['prompt'],
        "current_chat": current_chat,
        }
        prompt=f"\n\nHuman: {chat_template.format(**values)}\n\nAssistant:"
    else:
        # Chat template for open ended query
        with open("prompt/chat.txt","r") as f:
            chat_template=f.read()
        values = {
        "prompt": params['prompt'],
        "current_chat": current_chat,
        }
        prompt=f"\n\nHuman: {chat_template.format(**values)}\n\nAssistant:"
    
    inference_modifier = {'max_tokens_to_sample':OUTPUT_TOKEN, 
                      "temperature":0.5,
                      # "top_k":250,
                      # "top_p":1,    
                      "stop_sequences": ["Human:"]
                     }
    llm = Bedrock(model_id=f"anthropic.{params['model']}", client=bedrock_runtime, model_kwargs = inference_modifier,
              streaming=True,  # Toggle this to turn streaming on or off
              callbacks=handler)

    response = llm(prompt)
    claude = Anthropic()
    input_token=claude.count_tokens(chat_template)
    output_token=claude.count_tokens(response)
    tokens=input_token+output_token
    st.session_state['token']+=tokens
    # Update Dynamo DB table with current chat
    chat_history={"user": f"{params['prompt']}",
    "assiatant":f"\n\nAssistant: {response}\n\nHuman: "}           
    if DYNAMODB_TABLE:
        put_db(chat_history, params)
    else:
        st.session_state['chat_hist'].append(chat_history)
    return response

def db_message_2_streamlit_chat(current_chat):
    """
    Convert Chat history to fit streamlits syntax, when prepopulating previous chat in the UI.
    """
    import re
    parsed_data = []
    string=current_chat.replace("\n\nAssistant:","<<<Assistance>>>").replace("\n\nHuman:","<<</Assistance>>>")
    pattern = re.compile(r'<<<[^>]+>>>')
    segments = re.split(pattern, string)
    for ids,message in enumerate(segments):
        if ids%2==0:
            parsed_data.append({"role":"user","content":message.replace("Question:","")})
        else:
            parsed_data.append({"role":"assistant","content":message})
    return parsed_data

def get_key_from_value(dictionary, value):
    return next((key for key, val in dictionary.items() if val == value), None)
    
def chat_bedrock_(params):
    if params["chat_item"].strip():
        restored_chat=get_chat_historie(params)
        curr_message=db_message_2_streamlit_chat(restored_chat)
        curr_message.pop()
        st.session_state.messages=curr_message
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):   
            if "```" in message["content"]:
                st.markdown(message["content"],unsafe_allow_html=True )
            else:
                st.markdown(message["content"].replace("$", "\$"),unsafe_allow_html=True )
    if prompt := st.chat_input("Whats up?"):        
        st.session_state.messages.append({"role": "user", "content": prompt})        
        with st.chat_message("user"):             
            if "```" in prompt:
                st.markdown(prompt,unsafe_allow_html=True )
            else:
                st.markdown(prompt.replace("$", "\$"),unsafe_allow_html=True )
        with st.chat_message("assistant"): 
            message_placeholder = st.empty()
            stream_handler = StreamHandler(message_placeholder)
            time_now=time.time()            
            params["prompt"]=prompt
            answer=query_llm(params, [stream_handler])
            if "```" in answer:
                message_placeholder.markdown(answer,unsafe_allow_html=True )
            else:
                message_placeholder.markdown(answer.replace("$", "\$"),unsafe_allow_html=True )
            st.session_state.messages.append({"role": "assistant", "content": answer}) 
        
def app_sidebar():
    with st.sidebar:          
        holder=st.empty()
        button=st.button("New Chat", type ="primary")
        models=[ 'claude-instant-v1','claude-v2:1', 'claude-v2']
        model=st.selectbox('Model', models, index=2)
        params={"model":model}       
        user_chat_id=get_session_ids_by_user(DYNAMODB_TABLE, DYNAMODB_USER)
        dict_items = list(user_chat_id.items())
        dict_items.insert(0, (st.session_state['user_sess'],"New Chat")) 
        if button:
            st.session_state['user_sess'] = f"{DYNAMODB_USER}-{str(time.time()).split('.')[0]}"
            dict_items.insert(0, (st.session_state['user_sess'],"New Chat"))      
        st.session_state['chat_session_list'] = dict(dict_items)
        chat_items=st.selectbox("previous chat sessions",st.session_state['chat_session_list'].values(),key="chat_sessions")
        session_id=get_key_from_value(st.session_state['chat_session_list'], chat_items)   
        file = st.file_uploader('Upload a document', accept_multiple_files=True) 
        params={"model":model, "chat_id":session_id, "chat_item":chat_items, "file":file }  
        st.session_state['count']=1
        return params

    
    
def main():    
    params=app_sidebar()
    chat_bedrock_(params)
   
    
if __name__ == '__main__':
    main()   
