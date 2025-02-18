import os
import pickle
import streamlit as st
import time
from fetch_content import fetch_text_from_url
from process_text import process_text_and_store
from query_handler import handle_query
from langchain_community.llms import Ollama

st.title("🚀 RocketBot: News Research Tool")
st.sidebar.title("🔗 Enter News Article URLs")

# Sidebar input for URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("🔍 Process URLs")
file_path = "faiss_store_llama.pkl"

# Placeholder for status messages
main_placeholder = st.empty()

# Load the LLM (Llama3)
llm = Ollama(model="llama3", temperature=0.6)

if process_url_clicked:
    main_placeholder.text("📥 Fetching and processing articles...")
    
    # Fetch content from each URL
    data = [fetch_text_from_url(url) for url in urls if url.strip()]
    print(f"Data loaded: {data}")  # Debugging output

    if all(len(text) < 10 for text in data):
        st.error("⚠️ No valid content extracted. Check URLs or site restrictions.")
    else:
        vectorstore = process_text_and_store(data, urls)
        if vectorstore:
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore, f)
            main_placeholder.text("✅ Processing Complete! You can now ask questions.")

# Query input
query = st.text_input("💬 Ask a question about the articles:")
if query:
    handle_query(query, file_path, llm)
