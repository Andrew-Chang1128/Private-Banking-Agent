import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
import numpy as np
import os
import yfinance as yf
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
        client.delete_collection("financial_docs")
    except:
        pass  # Collection doesn't exist, which is fine

    # Create a collection with finance-optimized embedding function
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-mpnet-base-v2"  # High-quality model optimized for financial text
    )
    collection = client.create_collection("financial_docs", embedding_function=embedding_function)

    # Process PDF files from the directory using LangChain
    print(f"Processing financial documents from {pdf_directory} using LangChain...")

    try:
        # Use LangChain DirectoryLoader to load all PDFs in the directory
        loader = DirectoryLoader(
            pdf_directory,
            glob="*.pdf",  # Load all PDFs, including in subdirectories
            loader_cls=PyPDFLoader
        )

        # Load documents
        documents = loader.load()
        print(f"Loaded {len(documents)} financial document pages")

        if not documents:
            print("No financial documents were loaded from the PDF files")
            return collection

        # Use text splitter to create smaller chunks optimized for financial content
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} financial content chunks")

        # Prepare documents and IDs for ChromaDB
        doc_texts = [chunk.page_content for chunk in chunks]
        doc_ids = [f"fin_doc_{i}" for i in range(len(doc_texts))]

        # Add documents to the collection
        print(f"Adding {len(doc_texts)} financial document chunks to vector database")
        collection.add(
            documents=doc_texts,
            ids=doc_ids
        )

    except Exception as e:
        print(f"Error processing financial PDF files: {e}")

    return collection

class RAGChatbot:
    def __init__(self, model, tokenizer, collection):
        """Initialize the RAG chatbot with model, tokenizer, and document collection."""
        self.model = model
        self.tokenizer = tokenizer
        self.collection = collection
        self.is_first_message = True
        self.portfolio_tickers = []
        self.stock_insights = []

    def extract_tickers(self, portfolio_text: str) -> list:
        """Extract ticker symbols from the user's portfolio text using the LLM."""
        # Shorter, more direct prompt for faster processing
        prompt = f"""
        Extract ONLY stock ticker symbols from this portfolio text.
        Return ONLY comma-separated uppercase tickers (e.g., AAPL, MSFT).
        NO explanations, NO text after tickers.
        
        Example: " I have investments in Apple, Google, Palantir, Netflix, Amazon, and I also own some shares of Intel and Tesla that I bought last year
        Output: AAPL, GOOG, PLTR, NFLX, AMZN,INTC, TSLA
        
        Portfolio: {portfolio_text}
        Tickers:"""

        print("Extracting tickers...")
        
        # Tokenize with lower max_length to process faster
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=256,  # Limit input length for faster processing
            truncation=True
        ).to(self.model.device)
        
        # fast generation with minimal parameters
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=15,    # Further reduced - we only need a few tickers
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                # No sampling parameters, no beam search - pure greedy decoding
            )
            
            # Process result - focusing on everything after "Tickers:"
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the tickers from the result
            if "Tickers:" in result:
                tickers_part = result.split("Tickers:")[-1].strip()
            else:
                tickers_part = result.strip()
                
            # Clean up and extract tickers - handling any unwanted text
            if "\n" in tickers_part:
                tickers_part = tickers_part.split("\n")[0]
                
            # Process the tickers
            tickers = [ticker.strip().upper().replace('"', '') for ticker in tickers_part.split(',') if ticker.strip()]
            print(f"Extracted tickers: {tickers}")
            return tickers
            
        except Exception as e:
            print(f"Error extracting tickers: {str(e)}")
            # Return empty list in case of error
            return []
    
    def get_stock_insights(self, tickers: list) -> str:
        """Fetch real-time stock prices and fundamental data."""
        insights = "Investment Insights:\n"
        print(f"Fetching insights for {tickers}")

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                # Get price, P/E ratio, market cap, revenue, and earnings
                current_price = info.get("currentPrice", "N/A")
                pe_ratio = info.get("trailingPE", "N/A")
                market_cap = info.get("marketCap", "N/A")
                revenue = info.get("totalRevenue", "N/A")
                net_income = info.get("netIncomeToCommon", "N/A")
                debt_to_equity = info.get("debtToEquity", "N/A")

                insights += f"""
                - {ticker}:
                    Price: ${current_price}
                    P/E Ratio: {pe_ratio}
                    Market Cap: {market_cap}
                    Revenue: {revenue}
                    Net Income: {net_income}
                    Debt-to-Equity Ratio: {debt_to_equity}\n
                """

            except Exception as e:
                insights += f"- {ticker}: Unable to retrieve data (Error: {str(e)})\n"

        return insights
    
    def FinBERT_FLS_specifier(self, content: str) -> str:
        """Input a piece of financial content and return a statement generated by FinBERT."""
        # setup model
        from transformers import BertTokenizer, BertForSequenceClassification, pipeline
        finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-fls', num_labels=3)
        tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-fls')
        nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer)
        print("Initializing FinBERT model...\n")

        # usage
        results = nlp([content])
        label, score = results[0]['label'], results[0]['score']
        financial_interpretation = {
            'Specific FLS': 'Specific Forward-Looking Statement',
            'Not FLS': 'Non-Forward-Looking Statement',
            'Non-specific FLS': 'General Forward-Looking Statement'
        }
        statement = f"FinBERT Analysis:\n Classification - {financial_interpretation[label]}\n Confidence score: {score:.4f}"

        return statement

    def retrieve_context(self, question: str, n_results: int = 5) -> str:
        """Retrieve relevant context from the vector database."""
        # Fetch relevant documents
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )

        # Extract context from results
        contexts = results['documents'][0]

        # Define max character count
        max_context_chars = 1500  # Limit to ~375 tokens

        # Prioritize relevant contexts containing tickers from the user's portfolio
        relevant_contexts = [ctx for ctx in contexts if any(ticker in ctx for ticker in self.portfolio_tickers)]
        
        # If no directly relevant contexts, use any retrieved ones
        if not relevant_contexts:
            relevant_contexts = contexts  

        # Build combined context with max character limit
        combined_context = ""
        for ctx in relevant_contexts:
            if len(combined_context) + len(ctx) > max_context_chars:
                break  # Stop before exceeding max length
            combined_context += ctx + " "

        # Ensure clean formatting and trimming
        context = " ".join(combined_context.strip().split())  # Remove extra whitespace

        print("\n=== Retrieved Context ===")
        print(context)  # Display cleaned context
        print(f"Context length (characters): {len(context)}")
        print("=====================\n")

        return context


    def generate_response(self, question: str, context: str) -> str:
        """Generate a response using the LLM based on the question and context."""
        if "stress test" in question.lower() or "simulate scenarios" in question.lower():
            prompt = f"""As a financial analyst, perform a stress test on the portfolio based on the provided company data.
            
            Context: {context}

            Identify potential risks based on historical data and current market conditions. Consider scenarios such as:
            - Interest rate hikes or cuts
            - Sector-specific downturns
            - Economic recessions
            - Supply chain disruptions
            - Regulatory changes

            Use finance equations to estimate potential percentage changes in stock prices under these conditions.

            Answer:"""
        else:
            prompt = f"""As a financial advisor, answer the question based on the Context: {context}

            Question: {question}

            Answer: """

        print("=== Generated Prompt ===")
        print(" ".join(prompt.split()))
        print("=====================\n")
        try:
            max_input_length = 1024 
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

            print("Generating response...")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=700, 
                min_new_tokens=30, 
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True, 
                pad_token_id=self.tokenizer.eos_token_id
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip()
            else:
                answer = response.strip()

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
        
        if self.is_first_message:
            self.is_first_message = False
            self.portfolio_tickers = self.extract_tickers(question)
            
            if self.portfolio_tickers:
                stock_insights = self.get_stock_insights(self.portfolio_tickers)
                self.stock_insights = stock_insights
                context = self.retrieve_context(question)
                response = self.generate_response(question, context)
                return f"{stock_insights}\n\n{response}"
        
        context = self.retrieve_context(question)
        response = self.generate_response(question, context + self.stock_insights)
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

        results = self.collection.query(
            query_texts=[question],
            n_results=3 
        )

        contexts = results['documents'][0]

        print("\n=== Retrieved Contexts ===")
        for i, ctx in enumerate(contexts, 1):
            print(f"Context {i}: {ctx[:100]}...")
        print("=====================\n")

        response = "Based on the documents I've found:\n\n"

        max_context_display_chars = 500

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
    
    first_message = True
    # Chat loop
    while True:
        if first_message:
            question = input("Input your portfolio: ")
            first_message = False
        else:
            question = input("You: ")

        if question.lower() == 'exit':
            print("\nGoodbye!")
            break

        response = chatbot.chat(question)
        print(f"\nChatbot: {response}\n")

if __name__ == "__main__":
    main()
# I have investments in Apple, Google, Palantir, Netflix, Amazon and I also own some shares of Intel and Tesla that I bought last year
