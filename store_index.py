from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Replace OpenAI key with Groq key
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# If needed, set base URL for Groq
os.environ["OPENAI_API_KEY"] = GROQ_API_KEY  # only if using OpenAI-compatible clients
os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"  # important for LangChain or OpenAI SDK

# Load and process PDF documents
extracted_data = load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

# Use Hugging Face for embedding
embeddings = download_hugging_face_embeddings()

# Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"  # change if desired

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # this should match your embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# Upload documents to Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)
