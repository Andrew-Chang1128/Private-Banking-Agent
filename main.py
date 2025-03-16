import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
import numpy as np
import os
import yfinance as yf
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import BertTokenizer, BertForSequenceClassification, pipeline


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
    client = chromadb.Client()

    try:
        client.delete_collection("news_docs")
    except:
        pass

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.create_collection("news_docs", embedding_function=embedding_function)

    print(f"Processing PDF files from {pdf_directory} using LangChain...")

    try:
        loader = DirectoryLoader(
            pdf_directory,
            glob="*.pdf",
            loader_cls=PyPDFLoader
        )

        documents = loader.load()
        print(f"Loaded {len(documents)} document pages")

        if not documents:
            print("No documents were loaded from the PDF files")
            return collection

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")

        doc_texts = [chunk.page_content for chunk in chunks]
        doc_ids = [f"doc_{i}" for i in range(len(doc_texts))]

        print(f"Adding {len(doc_texts)} document chunks to vector database")
        collection.add(
            documents=doc_texts,
            ids=doc_ids
        )

    except Exception as e:
        print(f"Error processing PDF files: {e}")

    return collection

def setup_finbert():
    """Load FinBERT model for sentiment analysis."""
    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    nlp = pipeline("text-classification", model=finbert, tokenizer=tokenizer)
    return nlp

class RAGChatbot:
    def __init__(self, model, tokenizer, collection, finbert_nlp):
        """Initialize the RAG chatbot with model, tokenizer, and document collection."""
        self.model = model
        self.tokenizer = tokenizer
        self.collection = collection
        self.finbert_nlp = finbert_nlp
        self.is_first_message = True
        self.portfolio_tickers = []
        self.stock_insights = []

    def extract_tickers(self, portfolio_text: str) -> list:
        """Extract ticker symbols from the user's portfolio text using the LLM."""
        prompt = f"""
        Extract ONLY stock ticker symbols from this portfolio text.
        Return ONLY comma-separated uppercase tickers (e.g., AAPL, MSFT).
        NO explanations, NO text after tickers.
        
        Example: " I have investments in Apple, Google, Palantir, Netflix, Amazon, and I also own some shares of Intel and Tesla that I bought last year
        Output: AAPL, GOOG, PLTR, NFLX, AMZN,INTC, TSLA
        
        Portfolio: {portfolio_text}
        Tickers:"""

        print("Extracting tickers...")
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).to(self.model.device)
        
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=15,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "Tickers:" in result:
                tickers_part = result.split("Tickers:")[-1].strip()
            else:
                tickers_part = result.strip()
                
            if "\n" in tickers_part:
                tickers_part = tickers_part.split("\n")[0]
                
            tickers = [ticker.strip().upper().replace('"', '') for ticker in tickers_part.split(',') if ticker.strip()]
            print(f"Extracted tickers: {tickers}")
            return tickers
            
        except Exception as e:
            print(f"Error extracting tickers: {str(e)}")
            return []
    
    def get_stock_insights(self, tickers: list) -> str:
        """Fetch real-time stock prices and fundamental data."""
        insights = "Investment Insights:\n"
        print(f"Fetching insights for {tickers}")

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

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
    
    def analyze_sentiment(self, texts: List[str]) -> Dict[str, str]:
        """Analyze sentiment of retrieved document contexts using FinBERT."""
        sentiments = {}
        for text in texts:
            result = self.finbert_nlp(text[:512])
            label = result[0]['label']
            sentiments[text] = label
        return sentiments

    def retrieve_context(self, question: str, n_results: int = 5) -> str:
        """Retrieve relevant context from the vector database."""

        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )

        contexts = results['documents'][0]

        sentiment_results = self.analyze_sentiment(contexts)
        
        sentiment_summary = "\nSentiment Analysis of Retrieved Documents:\n"
        for context, sentiment in sentiment_results.items():
            summary_text = " ".join(context.split())
            sentiment_summary += f"- [{sentiment}] {summary_text[:100]}...\n"

        max_context_chars = 1500

        relevant_contexts = [ctx for ctx in contexts if any(ticker in ctx for ticker in self.portfolio_tickers)]
        
        if not relevant_contexts:
            relevant_contexts = contexts  

        combined_context = ""
        for ctx in relevant_contexts:
            if len(combined_context) + len(ctx) > max_context_chars:
                break
            combined_context += ctx + " "

        context = " ".join(combined_context.strip().split())

        print("\n=== Retrieved Context with Sentiment ===")
        print(sentiment_summary)
        print("=====================")
        
        return context, sentiment_summary


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
        print(prompt.replace("\\n", "\n")) 
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

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).replace("\\n", "\n")

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

        context, sentiment_summary = self.retrieve_context(question)
        response = self.generate_response(question, context + sentiment_summary.replace("\\n", "\n") + self.stock_insights)
        
        return f"{sentiment_summary}\n\n{response}"
    

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
    print("Initializing RAG Chatbot...")

    model, tokenizer = setup_model_and_tokenizer()
    collection = create_vector_database()
    finbert_nlp = setup_finbert()
    chatbot = RAGChatbot(model, tokenizer, collection, finbert_nlp)

    print("Chatbot initialized! Type 'exit' to end the conversation.\n")
    
    first_message = True
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
