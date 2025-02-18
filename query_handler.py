import os
import pickle
import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain

def handle_query(query, file_path, llm):
    """Handle the user's query, retrieve relevant documents and provide answers."""
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        retrieved_docs = retriever.get_relevant_documents(query)

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

        result = chain({"question": query}, return_only_outputs=True)

        st.header("📝 Answer")
        st.write(result["answer"])

        # Extract sources properly
        sources = list(set([doc.metadata.get("source", "Unknown") for doc in retrieved_docs]))

        if sources:
            st.subheader("🔗 Sources:")
            for source in sources:
                st.write(f"- {source}")
        else:
            st.write("⚠ No specific sources found.")
