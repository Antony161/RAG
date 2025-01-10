import os
from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core import load_index_from_storage
from llama_index.core import Settings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document





documents=SimpleDirectoryReader(input_files=[
    "/home/guest/basicrag/eBook-How-to-Build-a-Career-in-AI.pdf"
]).load_data()



document=Document(text="\n\n".join([doc.text for doc in documents]))


def build_sentence_window_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    sentence_window_size=3,
    save_dir="sentence_index",
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    llm=ChatOllama(model="llama3.2",temperature=0.1)
    Settings.llm=llm
    Settings.embed_model=embed_model
    Settings.node_parser=node_parser


    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            documents, service_context=Settings.embed_model
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=Settings.embed_model,
        )

    return sentence_index


def get_sentence_window_query_engine(
    sentence_index, similarity_top_k=6, rerank_top_n=2
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine


index = build_sentence_window_index(
    [document],
    llm=ChatOllama(model="llama3.2", temperature=0.1),
    save_dir="./sentence_index",
)
query_engine = get_sentence_window_query_engine(index, similarity_top_k=6)
response = query_engine.query(input("enter the query ?"))
print(response)
