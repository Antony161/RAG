import os

from llama_index.core import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document





documents=SimpleDirectoryReader(input_files=[
    "/home/guest/basicrag/eBook-How-to-Build-a-Career-in-AI.pdf"
]).load_data()



document=Document(text="\n\n".join([doc.text for doc in documents]))


def build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index",
    chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    
    llm=ChatOllama(model="llama3.2",temperature=0.1)
    Settings.llm=llm
    Settings.embed_model=embed_model
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context, service_context=Settings.embed_model
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=Settings.embed_model,
        )
    return automerging_index


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=6,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )
    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )
    return auto_merging_engine

index = build_automerging_index(
    [document],
    llm=ChatOllama(model="llama3.2", temperature=0.1),
    save_dir="./merging_index",
)
query_engine = get_automerging_query_engine(index, similarity_top_k=6)
response = query_engine.query(input("enter the query ?"))
print(response)
