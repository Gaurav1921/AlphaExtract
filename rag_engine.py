"""
RAG Query Engine with Gemini
-----------------------------
Ask questions about 10-K filings and get AI-generated answers with citations.

Example:
  Q: "What battery supply chain risks did Tesla mention?"
  A: "Tesla mentions three key battery risks:
      1. Lithium price volatility (Item 1A, chunk 12)
      2. Dependency on Panasonic (Item 1A, chunk 15)
      3. Cobalt sourcing (Item 1A, chunk 18)"
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import logging
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

from opensearch_setup import OpenSearchManager

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlphaRAG:
    """
    RAG (Retrieval-Augmented Generation) system for 10-K Q&A.
    """
    
    def __init__(
        self,
        gemini_api_key: str = None,
        model_name: str = "all-MiniLM-L6-v2",
        opensearch_host: str = "localhost",
        opensearch_port: int = 9200,
        index_name: str = "alphaextract-docs"
    ):
        """
        Initialize RAG system.
        
        Args:
            gemini_api_key: Google Gemini API key
            model_name: Sentence transformer model
            opensearch_host: OpenSearch host
            opensearch_port: OpenSearch port
            index_name: Index name
        """
        logger.info("Initializing AlphaRAG...")
        
        # Load embedding model (same as used for indexing)
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Connect to OpenSearch
        self.opensearch = OpenSearchManager(
            host=opensearch_host,
            port=opensearch_port,
            index_name=index_name
        )
        
        # Initialize Gemini
        if gemini_api_key is None:
            gemini_api_key = os.environ.get('GEMINI_API_KEY')
        
        if not gemini_api_key:
            logger.warning("No Gemini API key provided. Set GEMINI_API_KEY env var.")
            self.gemini = None
        else:
            genai.configure(api_key=gemini_api_key)
            self.gemini = genai.GenerativeModel('gemini-2.5-flash-lite')
            logger.info("✓ Gemini API configured (gemini-2.5-flash-lite)")
        
        logger.info("✓ AlphaRAG ready!")
    
    def embed_query(self, query: str) -> List[float]:
        """
        Convert query text to embedding vector.
        
        Args:
            query: Query string
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        ticker: Optional[str] = None,
        section: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant document chunks.
        
        Args:
            query: Query string
            k: Number of results
            ticker: Filter by ticker (optional)
            section: Filter by section (optional)
            
        Returns:
            List of relevant chunks with scores
        """
        # Convert query to embedding
        query_vector = self.embed_query(query)
        
        # Build filters
        filters = {}
        if ticker:
            filters['ticker'] = ticker.upper()
        if section:
            filters['section'] = section
        
        # Search OpenSearch
        results = self.opensearch.search(
            query_vector=query_vector,
            k=k,
            filters=filters if filters else None
        )
        
        return results
    
    def format_context(self, results: List[Dict]) -> str:
        """
        Format retrieved chunks into context for LLM.
        
        Args:
            results: Search results
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Document {i}]\n"
                f"Source: {result['ticker']} - {result['section']} (Chunk {result['chunk_id']})\n"
                f"Relevance Score: {result['score']:.3f}\n"
                f"Content: {result['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def generate_answer(
        self,
        query: str,
        context: str,
        results: List[Dict]
    ) -> Dict:
        """
        Generate answer using Gemini.
        
        Args:
            query: User query
            context: Retrieved context
            results: Search results (for citations)
            
        Returns:
            Dict with answer and metadata
        """
        if not self.gemini:
            return {
                'answer': "Gemini API not configured. Set GEMINI_API_KEY environment variable.",
                'sources': [],
                'error': True
            }
        
        # Build prompt
        prompt = f"""You are a financial analyst assistant. Answer the user's question based ONLY on the provided 10-K filing excerpts.

Context from 10-K filings:
{context}

User Question: {query}

Instructions:
1. Answer the question using ONLY information from the provided context
2. Be specific and cite which document(s) you're referencing (e.g., "According to Document 1...")
3. If the context doesn't contain relevant information, say "I don't have enough information in the provided excerpts to answer this question."
4. Be concise but comprehensive
5. Use bullet points for multiple items when appropriate

Answer:"""
        
        try:
            # Generate response
            response = self.gemini.generate_content(prompt)
            answer = response.text
            
            # Extract sources
            sources = [
                {
                    'ticker': r['ticker'],
                    'section': r['section'],
                    'chunk_id': r['chunk_id'],
                    'score': r['score'],
                    'preview': r['text'][:150] + "..."
                }
                for r in results
            ]
            
            return {
                'answer': answer,
                'sources': sources,
                'num_sources': len(sources),
                'error': False
            }
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': [],
                'error': True
            }
    
    def query(
        self,
        question: str,
        k: int = 5,
        ticker: Optional[str] = None,
        section: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Complete RAG pipeline: Retrieve → Generate.
        
        Args:
            question: User question
            k: Number of chunks to retrieve
            ticker: Filter by ticker (optional)
            section: Filter by section (optional)
            verbose: Print detailed output
            
        Returns:
            Dict with answer, sources, and metadata
        """
        if verbose:
            logger.info(f"\n{'='*80}")
            logger.info(f"QUERY: {question}")
            logger.info(f"{'='*80}")
        
        # Step 1: Retrieve relevant chunks
        if verbose:
            logger.info(f"\n[1] Retrieving relevant chunks (k={k})...")
        
        results = self.retrieve(
            query=question,
            k=k,
            ticker=ticker,
            section=section
        )
        
        if not results:
            return {
                'answer': "No relevant information found in the indexed documents.",
                'sources': [],
                'num_sources': 0,
                'error': False
            }
        
        if verbose:
            logger.info(f"✓ Found {len(results)} relevant chunks")
            for i, r in enumerate(results, 1):
                logger.info(f"  {i}. {r['ticker']} - {r['section']} (score: {r['score']:.3f})")
        
        # Step 2: Format context
        context = self.format_context(results)
        
        # Step 3: Generate answer
        if verbose:
            logger.info(f"\n[2] Generating answer with Gemini...")
        
        response = self.generate_answer(question, context, results)
        
        if verbose and not response['error']:
            logger.info(f"✓ Answer generated\n")
        
        return response
    
    def print_response(self, response: Dict):
        """
        Pretty print RAG response.
        
        Args:
            response: Response dict from query()
        """
        print(f"\n{'='*80}")
        print("ANSWER")
        print(f"{'='*80}")
        print(response['answer'])
        
        if response['sources']:
            print(f"\n{'='*80}")
            print(f"SOURCES ({len(response['sources'])} documents)")
            print(f"{'='*80}")
            
            for i, source in enumerate(response['sources'], 1):
                print(f"\n[{i}] {source['ticker']} - {source['section']} (Chunk {source['chunk_id']})")
                print(f"    Relevance: {source['score']:.3f}")
                print(f"    Preview: {source['preview']}")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ALPHARAG - Interactive Q&A System")
    print("=" * 80)
    
    # Check for API key
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("\n⚠️  GEMINI_API_KEY not set!")
        print("Set it with: export GEMINI_API_KEY='your-key-here'")
        api_key = input("\nEnter your Gemini API key (or press Enter to skip): ").strip()
        
        if api_key:
            os.environ['GEMINI_API_KEY'] = api_key
    
    # Initialize RAG
    print("\n[1] Initializing RAG system...")
    rag = AlphaRAG(gemini_api_key=api_key if api_key else None)
    
    # Test queries
    test_queries = [
        "What are the main risks Apple faces?",
        "What did Tesla say about battery supply chains?",
        "How is Google investing in AI?",
        "What are Microsoft's revenue trends?"
    ]
    
    print("\n[2] Running test queries...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST QUERY {i}/{len(test_queries)}")
        print(f"{'='*80}")
        print(f"Q: {query}\n")
        
        response = rag.query(query, k=3, verbose=False)
        rag.print_response(response)
        
        if i < len(test_queries):
            input("\nPress Enter for next query...")
    
    # Interactive mode
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("Type 'quit' to exit, 'help' for options")
    
    while True:
        print("\n" + "-"*80)
        query = input("\nYour question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if query.lower() == 'help':
            print("\nOptions:")
            print("  - Ask any question about the 10-K filings")
            print("  - Add 'ticker:AAPL' to filter by company")
            print("  - Add 'section:item_1a' to filter by section")
            print("  - Type 'quit' to exit")
            continue
        
        if not query:
            continue
        
        # Parse filters
        ticker = None
        section = None
        
        if 'ticker:' in query.lower():
            parts = query.split()
            for part in parts:
                if part.lower().startswith('ticker:'):
                    ticker = part.split(':')[1].upper()
                    query = query.replace(part, '').strip()
        
        if 'section:' in query.lower():
            parts = query.split()
            for part in parts:
                if part.lower().startswith('section:'):
                    section = part.split(':')[1].lower()
                    query = query.replace(part, '').strip()
        
        # Query
        response = rag.query(query, k=5, ticker=ticker, section=section, verbose=False)
        rag.print_response(response)