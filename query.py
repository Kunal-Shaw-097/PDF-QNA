from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings import GPT4AllEmbeddings
from apis import keys

vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    keys['MONGO_STR'],
    "try" + "." + "vec",
    GPT4AllEmbeddings(),
    index_name="vector_index",
)

query = "What is hazard?"

results = vector_search.similarity_search_with_score(
    query=query, k=2
)


context = ''

# Display results
for result in results:
    # print(result[0].page_content)
    # print(" \n")
    context += result[0].page_content

print(context)



from transformers import AutoTokenizer
import transformers
import torch

model = 'ericzzz/falcon-rw-1b-instruct-openorca'

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
   'text-generation',
   model=model,
   tokenizer=tokenizer,
   torch_dtype=torch.bfloat16,
   device_map='auto',
)

system_message = 'You are a helpful assistant. Give answers only if the information is present in the context, if information is not present answer with "Information is not present."'
prompt = f'<SYS> {system_message} <CONTEXT> {context} <INST> {query} <RESP> '

response = pipeline(
   prompt, 
   max_length=512,
   repetition_penalty=1.05
)
response = response[0]['generated_text'].split("<RESP>")[-1]
print(response)