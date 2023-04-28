from langchain.chains.question_answering import load_qa_chain
from qdrant_client import QdrantClient
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Qdrant
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from flask import Flask, request
from flask_cors import CORS
import json
import os
from configparser import ConfigParser
from pathlib import Path
# import openai

# fpath = Path.cwd() / Path('.env')
# cfg_reader = ConfigParser()
# cfg_reader.read(str(fpath))
# cfg_reader.get('API_KEYS', 'OPENAI_API_KEY')

# os.environ['OPENAI_API_KEY'] = cfg_reader.get('API_KEYS', 'OPENAI_API_KEY')
# os.environ['COHERE_API_KEY'] = cfg_reader.get('API_KEYS', 'COHERE_API_KEY')
# os.environ['QDRANT_API_KEY'] = cfg_reader.get('API_KEYS', 'QDRANT_API_KEY')
# os.environ['QDRANT_URL'] = cfg_reader.get('API_KEYS', 'QDRANT_URL')

openai_api_key = os.getenv('OPENAI_API_KEY')
cohere_api_key = os.getenv('COHERE_API_KEY')
qdrant_api_key = os.getenv('QDRANT_API_KEY')
qdrant_url = os.getenv('QDRANT_URL')

# print(openai_api_key)
# print(cohere_api_key)
# print(qdrant_api_key)
app = Flask(__name__)
CORS(app)
print("Exceuted1")


@app.route('/')
def hello_world():
    return {"PDFQA": "ChatBot"}


# @app.post('/embed')
@app.route('/embed', methods=['POST'])
def embed_pdf():
    collection_name = request.json.get('collection_name')
    file_url = request.json.get('file_url')
    try:
        loader = PyPDFLoader(file_url)
        docs = loader.load_and_split()
    except Exception as ex:
        return {"Error": "Unsupported file, Only Text PDFs are supported.."}

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)

    embed_model = CohereEmbeddings(
        model='multilingual-22-12', cohere_api_key=cohere_api_key)
    qdrant = Qdrant.from_documents(texts, embed_model, url=qdrant_url,
                                   collection_name=collection_name, prefer_grpc=True, api_key=qdrant_api_key)
    return {"collection_name": qdrant.collection_name}


@app.route('/retriever', methods=['POST'])
def retrieve_info():
    collection_name = request.json.get('collection_name')
    query = request.json.get('query')
    chat_llm = ChatOpenAI(temperature=0.1, openai_api_key=openai_api_key)

    client = QdrantClient(url=qdrant_url, prefer_grpc=True,
                          api_key=qdrant_api_key)
    embed_model = CohereEmbeddings(
        model='multilingual-22-12', cohere_api_key=cohere_api_key)
    qdrant = Qdrant(client=client, collection_name=collection_name,
                    embedding_function=embed_model.embed_query)
    search_results = qdrant.similarity_search(query, k=3)
    chain = load_qa_chain(chat_llm, chain_type='stuff')
    results = chain({'input_documents': search_results,
                    'question': query}, return_only_outputs=True)
    return {'results': results['output_text']}
