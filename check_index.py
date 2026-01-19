"""Quick check of OpenSearch index"""
from opensearch_setup import OpenSearchManager

manager = OpenSearchManager()
stats = manager.get_stats()

print(f"Documents in index: {stats['document_count']}")
print(f"Index size: {stats['size_mb']} MB")

# Try a simple search
print("\nTesting search...")
dummy_vector = [0.1] * 384
results = manager.search(dummy_vector, k=5)
print(f"Found {len(results)} results")

if results:
    print("\nSample result:")
    print(f"  Ticker: {results[0]['ticker']}")
    print(f"  Section: {results[0]['section']}")
    print(f"  Text: {results[0]['text'][:100]}...")