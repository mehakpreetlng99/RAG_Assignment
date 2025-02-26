import gradio as gr
import chromadb
import openai
import os
import itertools
import re
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Load environment variables
load_dotenv()
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ChromaDB Client Configuration
CHROMA_DB_DIR = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

# Set up embedding function for ChromaDB
embedding_function = SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Dictionary to store user query history per collection
user_query_history = {}

def sanitize_collection_name(filename):
    # Sanitize collection name to meet ChromaDB requirements.
    return re.sub(r"[^\w-]", "_", filename)[:63].strip("_")

def process_and_store_pdf(pdf_path, collection_name):
    # Process PDF, extract text, and store in ChromaDB.
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Create or get collection
    collection = client.get_or_create_collection(name=collection_name, embedding_function=embedding_function)

    # Store chunks in ChromaDB
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk.page_content],
            metadatas=[{"page": chunk.metadata.get("page", "N/A")}],
            ids=[f"{collection_name}_chunk_{i}"]
        )

    # Initialize query history for this collection
    user_query_history[collection_name] = []

    return len(chunks)  # Return processed chunk count

def query_pdf_database(query, collection_name):
    # Query the ChromaDB collection, retrieve relevant context, and use it to generate a response.
    try:
        collection = client.get_collection(name=collection_name, embedding_function=embedding_function)
    except Exception:
        return f"Error: Collection '{collection_name}' not found. Please upload and process the PDF first."

    # Retrieve relevant chunks from ChromaDB
    results = collection.query(
        query_texts=[query],
        n_results=3  # Fetch top 3 relevant results
    )

    if results and results['documents']:
        retrieved_chunks = list(itertools.chain(*results['documents']))  
        context = "\n".join(retrieved_chunks)  

        # Use GPT to generate a response with context
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for analyzing uploaded PDFs."},
                    {"role": "assistant", "content": f"Here is some relevant context:\n{context}"},
                    {"role": "user", "content": f"Based on this context, answer the following question: {query}"}
                ],
                max_tokens=200
            )
            final_response = response.choices[0].message.content

            # Store query and response in history
            if collection_name in user_query_history:
                user_query_history[collection_name].append({"query": query, "response": final_response})

            return final_response

        except Exception as e:
            return f"Error connecting to OpenAI API: {e}"
    else:
        return "No relevant data found for the query."


def get_query_history(collection_name):
    # Retrieve user query history for a specific collection.
    history = user_query_history.get(collection_name, [])
    if not history:
        return "No query history available for this document."
    
    formatted_history = "\n\n".join([f"**Q:** {entry['query']}\n**A:** {entry['response']}" for entry in history])
    return formatted_history

def process_pdf(files):
    # Process multiple PDFs and create collections.
    if not files:
        return "No files uploaded."

    results = []
    for file in files:
        pdf_path = file.name
        filename = os.path.splitext(os.path.basename(pdf_path))[0]
        collection_name = sanitize_collection_name(filename)

        if len(collection_name) < 3:
            results.append(f"Error: Collection name for '{filename}' is too short. Please rename the file.")
            continue

        try:
            num_chunks = process_and_store_pdf(pdf_path, collection_name)
            results.append(f"Processed '{filename}': {num_chunks} chunks stored in '{collection_name}'.")
        except Exception as e:
            results.append(f" Error processing '{filename}': {e}")

    return "\n".join(results)

def query_uploaded_pdf(query, files):
    # Query the latest uploaded PDF collection.
    if not files:
        return "No file uploaded for query."

    latest_file = files[-1]  # Query the last uploaded file
    filename = os.path.splitext(os.path.basename(latest_file.name))[0]
    collection_name = sanitize_collection_name(filename)

    return query_pdf_database(query, collection_name)

def view_query_history(files):
    # Retrieve the query history for the last uploaded PDF.
    if not files:
        return "No file uploaded."

    latest_file = files[-1]  
    filename = os.path.splitext(os.path.basename(latest_file.name))[0]
    collection_name = sanitize_collection_name(filename)

    return get_query_history(collection_name)

def build_gradio_app():
    with gr.Blocks() as app:
        gr.Markdown("### Upload PDFs for Analysis")
        
        pdf_input = gr.File(label="Upload PDFs", file_types=[".pdf"], file_count="multiple")
        process_button = gr.Button("Process PDFs")
        process_output = gr.Textbox(label="Processing Result", interactive=False)

        query_input = gr.Textbox(label="Ask a question about your PDF")
        query_button = gr.Button("Submit Query")
        query_output = gr.Textbox(label="Query Result", interactive=False)

        history_button = gr.Button("View Query History")
        history_output = gr.Textbox(label="Query History", interactive=False)

        # Button click events
        process_button.click(process_pdf, inputs=pdf_input, outputs=process_output)
        query_button.click(query_uploaded_pdf, inputs=[query_input, pdf_input], outputs=query_output)
        history_button.click(view_query_history, inputs=pdf_input, outputs=history_output)

        return app

if __name__ == "__main__":
    app = build_gradio_app()
    app.launch()


