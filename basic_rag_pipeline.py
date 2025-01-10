
import os 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama


from llama_index.core import SimpleDirectoryReader

documents=SimpleDirectoryReader(input_files=[
    "/home/guest/basicrag/eBook-How-to-Build-a-Career-in-AI.pdf"
]).load_data()

from llama_index.core import Document

document=Document(text="\n\n".join([doc.text for doc in documents]))

from llama_index.core import VectorStoreIndex
from llama_index.core import Settings

Settings.llm=ChatOllama(model="llama3.2",temperature=0.1)

Settings.embed_model="local:BAAI/bge-small-en-v1.5"   ###embedding model by huggingface to convert the documents into numerical 

index = VectorStoreIndex.from_documents([document],
                                        service_context=Settings.embed_model)

query_engine = index.as_query_engine()

response = query_engine.query(input("enter the query ?")
)
print(str(response))
