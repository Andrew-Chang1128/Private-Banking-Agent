import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
import numpy as np
import os

# Import LangChain components for PDF processing
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def setup_model_and_tokenizer():
    """Initialize the Llama model and tokenizer."""
    model_name = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def create_vector_database(pdf_directory="./"):
    """Create and populate the vector database with content from PDF documents using LangChain.

    Args:
        pdf_directory: Directory containing PDF files to process

    Returns:
        ChromaDB collection with document contents
    """
    # Initialize ChromaDB client
    client = chromadb.Client()

    # Delete existing collection if it exists
    try:
        client.delete_collection("multilingual_docs")
    except:
        pass  # Collection doesn't exist, which is fine

    # Create a collection with multilingual embedding function
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-mpnet-base-v2"
    )
    collection = client.create_collection("multilingual_docs", embedding_function=embedding_function)

    # Process PDF files from the directory using LangChain
    print(f"Processing PDF files from {pdf_directory} using LangChain...")

    try:
        # Use LangChain DirectoryLoader to load all PDFs in the directory
        loader = DirectoryLoader(
            pdf_directory,
            glob="*.pdf",  # Load all PDFs, including in subdirectories
            loader_cls=PyPDFLoader
        )

        # Load documents
        documents = loader.load()
        print(f"Loaded {len(documents)} document pages")

        if not documents:
            print("No documents were loaded from the PDF files")
            return collection

        # Use text splitter to create smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")

        # Prepare documents and IDs for ChromaDB
        doc_texts = [chunk.page_content for chunk in chunks]
        doc_ids = [f"doc_{i}" for i in range(len(doc_texts))]

        # Add documents to the collection
        print(f"Adding {len(doc_texts)} document chunks to vector database")
        collection.add(
            documents=doc_texts,
            ids=doc_ids
        )

    except Exception as e:
        print(f"Error processing PDF files: {e}")

    return collection

class RAGChatbot:
    def __init__(self, model, tokenizer, collection):
        """Initialize the RAG chatbot with model, tokenizer, and document collection."""
        self.model = model
        self.tokenizer = tokenizer
        self.collection = collection

    def retrieve_context(self, question: str, n_results: int = 2) -> str:
        """Retrieve relevant context from the vector database."""
        # Limit the number of tokens to retrieve to prevent exceeding model limits
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )

        # Get all contexts
        contexts = results['documents'][0]

        # Join contexts but ensure we're not creating something too long
        # Estimate roughly 4 chars per token as a conservative measure
        max_context_chars = 1500  # ~375 tokens, leaving room for question and overhead

        combined_context = ""
        for ctx in contexts:
            # If adding this context would make the combined context too long, stop
            if len(combined_context) + len(ctx) > max_context_chars:
                combined_context += "... (additional context truncated to fit model limits)"
                break
            combined_context += ctx + " "

        context = combined_context.strip()

        print("\n=== Retrieved Context ===")
        print(context)
        print(f"Context length (characters): {len(context)}")
        print("=====================\n")
        return context

    def generate_response(self, question: str, context: str) -> str:
        """Generate a response using the LLM based on the question and context."""
        prompt = f"""Context: {context}

        Question: {question}

        Answer: """

        print("=== Generated Prompt ===")
        print(prompt)
        print("=====================\n")

        try:
            # Tokenize input with better truncation - ensure we're well below the max
            max_input_length = 450  # Reduced from 512 to provide buffer
            print(f"Tokenizing input (max length: {max_input_length})...")
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=max_input_length,
                truncation=True
            ).to(self.model.device)

            input_length = len(inputs.input_ids[0])
            print(f"Input length after tokenization: {input_length} tokens")

            if input_length > max_input_length:
                print(f"⚠️ Warning: Input was truncated from {input_length} to {max_input_length} tokens")

            # Generate response with max_new_tokens instead of max_length as recommended
            print("Generating response...")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,  # Use max_new_tokens instead of max_length
                min_new_tokens=30,   # Ensure some meaningful content
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,  # Use sampling for faster generation
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Decode and return response - use skip_special_tokens
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract the answer from the full response
            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip()
            else:
                answer = response.strip()

            print("=== Generated Answer ===")
            print(answer)
            print("=====================\n")

            return answer

        except Exception as e:
            print(f"=== Generation Error ===")
            print(f"Error: {str(e)}")
            print("=====================\n")
            return f"I encountered an error while generating a response: {str(e)}. Try asking a shorter question."

    def chat(self, question: str) -> str:
        """Process a single chat interaction."""
        print("\n=== User Question ===")
        print(question)
        print("=====================\n")

        context = self.retrieve_context(question)
        response = self.generate_response(question, context)
        return response

class SimpleRAGRetriever:
    """A simpler RAG implementation that just returns relevant context without an LLM."""

    def __init__(self, collection):
        """Initialize with just the vector database collection."""
        self.collection = collection

    def chat(self, question: str) -> str:
        """Process a chat interaction by just retrieving relevant passages."""
        print("\n=== User Question ===")
        print(question)
        print("=====================\n")

        # Retrieve relevant context
        results = self.collection.query(
            query_texts=[question],
            n_results=3  # Get more results since we're not using an LLM
        )

        contexts = results['documents'][0]

        print("\n=== Retrieved Contexts ===")
        for i, ctx in enumerate(contexts, 1):
            print(f"Context {i}: {ctx[:100]}...")  # Show preview of each context
        print("=====================\n")

        # Format and return the relevant passages as the response
        response = "Based on the documents I've found:\n\n"

        # Make sure we don't exceed reasonable length
        max_context_display_chars = 500  # Limit each context display

        for i, context in enumerate(contexts, 1):
            if len(context) > max_context_display_chars:
                truncated = context[:max_context_display_chars] + "... (truncated)"
                response += f"Passage {i}: {truncated}\n\n"
            else:
                response += f"Passage {i}: {context.strip()}\n\n"

        response += "These are the most relevant passages from your documents that might help answer your question."

        return response

def main():
    """Main function to run the chatbot."""
    print("Initializing Multilingual RAG Chatbot...")

    # Setup components
    model, tokenizer = setup_model_and_tokenizer()
    collection = create_vector_database()
    chatbot = RAGChatbot(model, tokenizer, collection)

    print("Chatbot initialized! Type 'exit' to end the conversation.\n")

    # Chat loop
    while True:
        question = input("You: ")

        if question.lower() == 'exit':
            print("\nGoodbye!")
            break

        response = chatbot.chat(question)
        print(f"\nChatbot: {response}\n")

if __name__ == "__main__":
    main()