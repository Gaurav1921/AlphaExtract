"""
Enhanced RAG Engine - Production Grade
--------------------------------------
Improvements over basic RAG:
1. Conversation history for multi-turn chat
2. Cross-encoder re-ranking for better retrieval
3. Query expansion for improved recall
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai
from dotenv import load_dotenv
import re

from opensearch_setup import OpenSearchManager

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationHistory:
    """
    Manages conversation context for multi-turn chat.
    """
    
    def __init__(self, max_history: int = 5):
        """
        Initialize conversation history.
        
        Args:
            max_history: Maximum number of Q&A pairs to remember
        """
        self.history = []
        self.max_history = max_history
    
    def add(self, question: str, answer: str, sources: List[Dict] = None):
        """Add Q&A pair to history."""
        self.history.append({
            'question': question,
            'answer': answer,
            'sources': sources or []
        })
        
        # Keep only last N exchanges
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context(self) -> str:
        """Format history as context string."""
        if not self.history:
            return ""
        
        context_parts = ["Previous conversation:"]
        for i, exchange in enumerate(self.history, 1):
            context_parts.append(f"\nQ{i}: {exchange['question']}")
            context_parts.append(f"A{i}: {exchange['answer'][:200]}...")
        
        return "\n".join(context_parts)
    
    def clear(self):
        """Clear conversation history."""
        self.history = []
    
    def is_follow_up(self, query: str) -> bool:
        """
        Detect if query is a follow-up question.
        
        Args:
            query: Current query
            
        Returns:
            True if query references previous context
        """
        if not self.history:
            return False
        
        follow_up_indicators = [
            'that', 'this', 'it', 'them', 'they',
            'more', 'also', 'what about', 'how about',
            'compared to', 'versus', 'similar'
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in follow_up_indicators)
    
    def expand_follow_up(self, query: str) -> str:
        """
        Expand follow-up question with context from history.
        
        Args:
            query: Current query
            
        Returns:
            Expanded query with context
        """
        if not self.is_follow_up(query):
            return query
        
        # Get last question's context
        last_q = self.history[-1]['question']
        
        # Simple expansion: prepend previous topic
        expanded = f"{last_q} {query}"
        
        logger.info(f"Expanded follow-up: '{query}' → '{expanded}'")
        return expanded


class QueryExpander:
    """
    Expands queries with synonyms and related terms.
    """
    
    # Domain-specific synonym map
    SYNONYMS = {
        'ai': ['artificial intelligence', 'machine learning', 'ML', 'deep learning'],
        'risk': ['threat', 'vulnerability', 'challenge', 'concern'],
        'revenue': ['sales', 'income', 'earnings'],
        'profit': ['earnings', 'income', 'margin'],
        'supply chain': ['logistics', 'manufacturing', 'suppliers'],
        'battery': ['cell', 'energy storage', 'lithium-ion'],
        'regulation': ['regulatory', 'compliance', 'legal'],
        'competition': ['competitive', 'competitor', 'rival'],
        'growth': ['expansion', 'increase', 'scaling'],
        'customer': ['consumer', 'user', 'client']
    }
    
    def expand(self, query: str) -> List[str]:
        """
        Generate query variations with synonyms.
        
        Args:
            query: Original query
            
        Returns:
            List of query variations (original + expansions)
        """
        queries = [query]  # Always include original
        
        query_lower = query.lower()
        
        # Find matching terms
        for term, synonyms in self.SYNONYMS.items():
            if term in query_lower:
                # Generate variations
                for syn in synonyms[:2]:  # Limit to 2 synonyms per term
                    # Replace term with synonym
                    expanded = re.sub(
                        r'\b' + term + r'\b',
                        syn,
                        query_lower,
                        flags=re.IGNORECASE
                    )
                    if expanded != query_lower:
                        queries.append(expanded)
        
        # Deduplicate
        queries = list(dict.fromkeys(queries))
        
        if len(queries) > 1:
            logger.info(f"Query expansion: {len(queries)} variations")
        
        return queries[:3]  # Max 3 variations


class EnhancedRAG:
    """
    Production-grade RAG with conversation history, re-ranking, and query expansion.
    """
    
    def __init__(
        self,
        gemini_api_key: str = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        opensearch_host: str = "localhost",
        opensearch_port: int = 9200,
        index_name: str = "alphaextract-docs",
        enable_reranking: bool = True,
        enable_query_expansion: bool = True
    ):
        """
        Initialize enhanced RAG system.
        
        Args:
            gemini_api_key: Google Gemini API key
            embedding_model: Sentence transformer model
            reranker_model: Cross-encoder model for re-ranking
            opensearch_host: OpenSearch host
            opensearch_port: OpenSearch port
            index_name: Index name
            enable_reranking: Enable re-ranking
            enable_query_expansion: Enable query expansion
        """
        logger.info("Initializing Enhanced RAG...")
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Load re-ranker (optional)
        self.enable_reranking = enable_reranking
        if enable_reranking:
            logger.info(f"Loading re-ranker: {reranker_model}")
            self.reranker = CrossEncoder(reranker_model)
            logger.info("✓ Re-ranker loaded")
        else:
            self.reranker = None
        
        # Initialize query expander
        self.enable_query_expansion = enable_query_expansion
        self.query_expander = QueryExpander()
        
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
            logger.warning("No Gemini API key provided")
            self.gemini = None
        else:
            genai.configure(api_key=gemini_api_key)
            self.gemini = genai.GenerativeModel('gemini-2.5-flash-lite')
            logger.info("✓ Gemini configured")
        
        # Conversation history
        self.conversation = ConversationHistory(max_history=5)
        
        logger.info("✓ Enhanced RAG ready!")
    
    def embed_query(self, query: str) -> List[float]:
        """Convert query to embedding."""
        embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        return embedding.tolist()
    
    def retrieve_multi_query(
        self,
        queries: List[str],
        k: int = 10,
        ticker: Optional[str] = None,
        section: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve documents for multiple query variations.
        
        Args:
            queries: List of query variations
            k: Number of results per query
            ticker: Filter by ticker
            section: Filter by section
            
        Returns:
            Deduplicated and merged results
        """
        all_results = {}  # Use dict to deduplicate by chunk_id
        
        for query in queries:
            query_vector = self.embed_query(query)
            
            filters = {}
            if ticker:
                filters['ticker'] = ticker.upper()
            if section:
                filters['section'] = section
            
            results = self.opensearch.search(
                query_vector=query_vector,
                k=k,
                filters=filters if filters else None
            )
            
            # Merge results (keep highest score for each chunk)
            for result in results:
                key = f"{result['ticker']}_{result['section']}_{result['chunk_id']}"
                
                if key not in all_results or result['score'] > all_results[key]['score']:
                    all_results[key] = result
        
        # Sort by score
        merged_results = list(all_results.values())
        merged_results.sort(key=lambda x: x['score'], reverse=True)
        
        return merged_results
    
    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Re-rank results using cross-encoder.
        
        Args:
            query: Original query
            results: Retrieved results
            top_k: Number of results to return
            
        Returns:
            Re-ranked results
        """
        if not self.enable_reranking or not self.reranker:
            return results[:top_k]
        
        if not results:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [(query, result['text']) for result in results]
        
        # Get re-ranking scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Add rerank scores to results
        for result, score in zip(results, rerank_scores):
            result['rerank_score'] = float(score)
        
        # Sort by rerank score
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        logger.info(f"Re-ranked {len(results)} results → top {top_k}")
        
        return results[:top_k]
    
    def format_context(self, results: List[Dict]) -> str:
        """Format results into context for LLM."""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            score_info = f"Relevance: {result['score']:.3f}"
            if 'rerank_score' in result:
                score_info += f" | Rerank: {result['rerank_score']:.3f}"
            
            context_parts.append(
                f"[Document {i}]\n"
                f"Source: {result['ticker']} - {result['section']} (Chunk {result['chunk_id']})\n"
                f"{score_info}\n"
                f"Content: {result['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def generate_answer(
        self,
        query: str,
        context: str,
        results: List[Dict],
        conversation_context: str = ""
    ) -> Dict:
        """Generate answer with Gemini."""
        if not self.gemini:
            return {
                'answer': "Gemini API not configured.",
                'sources': [],
                'error': True
            }
        
        # Build prompt with conversation context
        prompt = f"""You are a financial analyst assistant analyzing 10-K filings.

{conversation_context}

Context from 10-K filings:
{context}

Current Question: {query}

Instructions:
1. Answer based ONLY on the provided context
2. Reference specific documents (e.g., "According to Document 1...")
3. If context is insufficient, say so
4. Be concise but comprehensive
5. Use bullet points for multiple items

Answer:"""
        
        try:
            response = self.gemini.generate_content(prompt)
            answer = response.text
            
            sources = [
                {
                    'ticker': r['ticker'],
                    'section': r['section'],
                    'chunk_id': r['chunk_id'],
                    'score': r['score'],
                    'rerank_score': r.get('rerank_score'),
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
                'answer': f"Error: {str(e)}",
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
        Complete enhanced RAG pipeline.
        
        Args:
            question: User question
            k: Number of results to return
            ticker: Filter by ticker
            section: Filter by section
            verbose: Print detailed logs
            
        Returns:
            Response dict with answer and sources
        """
        if verbose:
            logger.info(f"\n{'='*80}")
            logger.info(f"QUERY: {question}")
            logger.info(f"{'='*80}")
        
        # Step 1: Check if follow-up question
        if self.conversation.is_follow_up(question):
            expanded_question = self.conversation.expand_follow_up(question)
            if verbose:
                logger.info(f"[Follow-up detected] Expanded query")
        else:
            expanded_question = question
        
        # Step 2: Query expansion
        if self.enable_query_expansion:
            query_variations = self.query_expander.expand(expanded_question)
            if verbose and len(query_variations) > 1:
                logger.info(f"[Query expansion] {len(query_variations)} variations")
        else:
            query_variations = [expanded_question]
        
        # Step 3: Retrieve
        if verbose:
            logger.info(f"\n[1] Retrieving documents (k={k*2} before rerank)...")
        
        results = self.retrieve_multi_query(
            queries=query_variations,
            k=k*2,  # Retrieve more, then rerank
            ticker=ticker,
            section=section
        )
        
        if not results:
            return {
                'answer': "No relevant information found.",
                'sources': [],
                'num_sources': 0,
                'error': False
            }
        
        if verbose:
            logger.info(f"✓ Retrieved {len(results)} candidates")
        
        # Step 4: Re-rank
        if self.enable_reranking:
            if verbose:
                logger.info(f"\n[2] Re-ranking with cross-encoder...")
            results = self.rerank(question, results, top_k=k)
        else:
            results = results[:k]
        
        if verbose:
            logger.info(f"✓ Top {len(results)} after re-ranking:")
            for i, r in enumerate(results, 1):
                score_str = f"{r['score']:.3f}"
                if 'rerank_score' in r:
                    score_str += f" → {r['rerank_score']:.3f}"
                logger.info(f"  {i}. {r['ticker']} (score: {score_str})")
        
        # Step 5: Generate answer
        context = self.format_context(results)
        conversation_context = self.conversation.get_context()
        
        if verbose:
            logger.info(f"\n[3] Generating answer...")
        
        response = self.generate_answer(
            question,
            context,
            results,
            conversation_context
        )
        
        # Step 6: Update conversation history
        if not response['error']:
            self.conversation.add(question, response['answer'], response['sources'])
        
        return response
    
    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation.clear()
        logger.info("Conversation history cleared")
    
    def print_response(self, response: Dict):
        """Pretty print response."""
        print(f"\n{'='*80}")
        print("ANSWER")
        print(f"{'='*80}")
        print(response['answer'])
        
        if response['sources']:
            print(f"\n{'='*80}")
            print(f"SOURCES ({len(response['sources'])} documents)")
            print(f"{'='*80}")
            
            for i, source in enumerate(response['sources'], 1):
                score_str = f"{source['score']:.3f}"
                if source.get('rerank_score'):
                    score_str += f" → {source['rerank_score']:.3f}"
                
                print(f"\n[{i}] {source['ticker']} - {source['section']} (Chunk {source['chunk_id']})")
                print(f"    Score: {score_str}")
                print(f"    Preview: {source['preview']}")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ENHANCED RAG - Multi-Turn Chat Demo")
    print("=" * 80)
    
    # Initialize
    print("\n[1] Initializing Enhanced RAG...")
    rag = EnhancedRAG()
    
    # Demo: Multi-turn conversation
    print("\n[2] Multi-turn conversation demo:\n")
    
    conversation = [
        "What are Apple's main risks?",
        "How do they compare to Tesla's risks?",  # Follow-up
        "What about their revenue trends?",         # Follow-up
    ]
    
    for i, query in enumerate(conversation, 1):
        print(f"\n{'='*80}")
        print(f"TURN {i}")
        print(f"{'='*80}")
        print(f"Q: {query}\n")
        
        response = rag.query(query, k=3, verbose=False)
        rag.print_response(response)
        
        if i < len(conversation):
            input("\n[Press Enter for next turn...]")
    
    # Interactive mode
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("Commands: 'quit', 'reset' (clear history), 'help'")
    
    while True:
        print("\n" + "-"*80)
        query = input("\nYour question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if query.lower() == 'reset':
            rag.reset_conversation()
            print("✓ Conversation history cleared")
            continue
        
        if query.lower() == 'help':
            print("\nFeatures:")
            print("  • Ask follow-up questions (e.g., 'What about X?')")
            print("  • Query expansion (automatic synonyms)")
            print("  • Re-ranking (better relevance)")
            print("  • Conversation memory (last 5 turns)")
            continue
        
        if not query:
            continue
        
        response = rag.query(query, k=5, verbose=False)
        rag.print_response(response)