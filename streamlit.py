import streamlit as st

from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_community.document_loaders import PyPDFLoader

from transformers import AutoTokenizer
import transformers
import torch

from apis import keys


model_id = 'ericzzz/falcon-rw-1b-instruct-openorca'

if 'pipeline' not in st.session_state :

    st.session_state.tokenizer = AutoTokenizer.from_pretrained(model_id)

    st.session_state.pipeline = transformers.pipeline(
    'text-generation',
    model=model_id,
    tokenizer=st.session_state.tokenizer,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    )

st.title("PDF Q&A")
    
st.write('PDF')
    

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

if uploaded_file and 'uploaded' not in st.session_state:
    
    st.session_state.uploaded = True
    filename = 'temp.pdf'
    with open(filename, 'wb') as f: 
        f.write(uploaded_file.getvalue())

    data = PyPDFLoader('temp.pdf')
    pdf = data.load()

    # Step 2: Transform (Split)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[
                                                "\n\n", "\n", "(?<=\. )", " "], length_function=len)
    docs = text_splitter.split_documents(pdf)
    #print('Split into ' + str(len(docs)) + ' docs')

    # Step 3: Embed
    # https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.openai.OpenAIEmbeddings.html

    gpt4all_embd = GPT4AllEmbeddings()

    # Step 4: Store
    # Initialize MongoDB python client
    client = MongoClient(keys['MONGO_STR'], server_api=ServerApi('1'))
    collection = client['try']['vec']

    # # Reset w/out deleting the Search Index 
    collection.delete_many({})

    docsearch = MongoDBAtlasVectorSearch.from_documents(
        docs, gpt4all_embd, collection=collection, index_name = "vector_index"
    )


query = st.text_input('Enter your question here!')

if query and st.session_state.uploaded: 

    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        keys['MONGO_STR'],
        "try" + "." + "vec",
        GPT4AllEmbeddings(),
        index_name="vector_index",
    )


    results = vector_search.similarity_search_with_score(
        query=query, k=2
    )


    context = ''

    for result in results:
        context += result[0].page_content


    system_message = 'You are a helpful assistant. Give answers only if the information is present in the context, if information is not present answer with "Information is not present."'
    prompt = f'<SYS> {system_message} <CONTEXT> {context} <INST> {query} <RESP> '

    response = st.session_state.pipeline(
    prompt, 
    max_length=512,
    repetition_penalty=1.05
    )

    response = response[0]['generated_text'].split("<RESP>")[-1]

    st.write(response)
