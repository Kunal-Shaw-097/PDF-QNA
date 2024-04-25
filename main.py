import torch
from pypdf import PdfReader 
import re
import transformers
from apis import keys

# # Function to clean text
# def clean_text(text):
#     # Remove non-ASCII characters
#     text = re.sub(r'[^\x00-\x7F]+', ' ', text)
#     # Remove newlines and tabs
#     text = re.sub(r'[\r|\n|\r\n]+', ' ', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# pdf = PdfReader('hazard-and-risk.pdf')

# print(len(pdf.pages)) 
  
# # creating a page object 
# page = pdf.pages[0] 

# text = page.extract_text().strip()

# text = clean_text(text)

from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_community.document_loaders import PyPDFLoader


data = PyPDFLoader('hazard-and-risk.pdf')
pdf = data.load()


# Step 2: Transform (Split)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[
                                               "\n\n", "\n", "(?<=\. )", " "], length_function=len)
docs = text_splitter.split_documents(pdf)
print('Split into ' + str(len(docs)) + ' docs')

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






