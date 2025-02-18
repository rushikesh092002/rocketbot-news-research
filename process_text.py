import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def process_text_and_store(data, urls):
    """Process text, create embeddings, and store them in a FAISS vector store."""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )

    # Splitting text into smaller chunks
    docs = []
    for i, text in enumerate(data):
        split_docs = text_splitter.create_documents([text])
        for doc in split_docs:
            doc.metadata["source"] = urls[i]
        docs.extend(split_docs)

    if not docs:
        return None
    
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore
