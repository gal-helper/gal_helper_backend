import asyncio
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from ai_service import ai_service
from database import db_service
from rag_processor import rag_processor

async def debug_database_connection():
    print("\n[1/7] Debugging database connection...")
    
    if not await db_service.connect():
        print("FAILED: Database connection failed")
        return False
    
    print("SUCCESS: Database connected")
    return True

async def debug_table_structure():
    print("\n[2/7] Debugging table structure...")
    
    try:
        async with db_service.pool.acquire() as conn:
            tables = await conn.fetch("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            
            print(f"Found tables: {[t['table_name'] for t in tables]}")
            
            if 'documents' not in [t['table_name'] for t in tables]:
                print("ERROR: 'documents' table missing")
                return False
            
            columns = await conn.fetch("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = 'documents'
                ORDER BY ordinal_position
            """)
            
            print("Documents table columns:")
            for col in columns:
                print(f"  - {col['column_name']}: {col['data_type']} ({'NULL' if col['is_nullable'] == 'YES' else 'NOT NULL'})")
            
            vector_column = [c for c in columns if c['column_name'] == 'content_vector']
            if not vector_column:
                print("ERROR: content_vector column missing")
                return False
            
            print("SUCCESS: Table structure looks correct")
            return True
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

async def debug_vector_extension():
    print("\n[3/7] Debugging pgvector extension...")
    
    try:
        async with db_service.pool.acquire() as conn:
            extensions = await conn.fetch("""
                SELECT extname, extversion 
                FROM pg_extension 
                WHERE extname = 'vector'
            """)
            
            if not extensions:
                print("ERROR: pgvector extension not installed")
                print("Run: CREATE EXTENSION IF NOT EXISTS vector;")
                return False
            
            print(f"SUCCESS: pgvector extension installed (version: {extensions[0]['extversion']})")
            
            vectorized_count = await conn.fetchval(
                "SELECT COUNT(*) FROM documents WHERE content_vector IS NOT NULL"
            )
            print(f"Vectorized documents: {vectorized_count}")
            
            return True
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

async def debug_embedding_generation():
    print("\n[4/7] Debugging embedding generation...")
    
    test_texts = [
        "Test text for embedding generation",
        "Artificial intelligence and machine learning",
        "Test query for vector search debugging"
    ]
    
    try:
        for i, text in enumerate(test_texts, 1):
            print(f"\n  Test {i}: '{text[:30]}...'")
            embedding = await ai_service.get_embedding(text)
            print(f"    Dimensions: {len(embedding)}")
            print(f"    First 3 values: {embedding[:3]}")
            print(f"    Last 3 values: {embedding[-3:]}")
        
        print("\nSUCCESS: Embedding generation working")
        return True
        
    except Exception as e:
        print(f"ERROR: Embedding generation failed: {e}")
        return False

async def debug_search_functionality():
    print("\n[5/7] Debugging search functionality...")
    
    test_query = "machine learning artificial intelligence"
    
    try:
        query_embedding = await ai_service.get_embedding(test_query)
        print(f"Query embedding generated: {len(query_embedding)} dimensions")
        
        vector_results = await db_service.search_similar_documents(query_embedding, limit=3)
        print(f"Vector search results: {len(vector_results)} documents")
        
        keyword_results = await db_service.keyword_search("machine", limit=3)
        print(f"Keyword search results: {len(keyword_results)} documents")
        
        if not vector_results and not keyword_results:
            print("WARNING: No documents found in database")
            
        return True
        
    except Exception as e:
        print(f"ERROR: Search functionality failed: {e}")
        return False

async def debug_rag_processor_logic():
    print("\n[6/7] Debugging RAG processor logic...")
    
    try:
        print("Initializing RAG processor...")
        await rag_processor.initialize()
        
        test_question = "What is machine learning?"
        
        print(f"\nTesting with question: '{test_question}'")
        print("Using RAG mode...")
        
        start_time = time.time()
        result = await rag_processor.ask_question(test_question, use_rag=True)
        elapsed = time.time() - start_time
        
        print(f"\nRAG Process Results:")
        print(f"  Success: {result['success']}")
        print(f"  RAG Used: {result['rag_used']}")
        print(f"  Response time: {elapsed:.2f}s")
        print(f"  Sources found: {len(result.get('sources', []))}")
        
        if result.get('sources'):
            print("\n  Sources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"    {i}. {source['filename']}")
                print(f"       Similarity: {source.get('similarity', 0):.3f}")
                print(f"       Method: {source.get('search_method', 'unknown')}")
        
        print("\nTesting without RAG mode...")
        result_no_rag = await rag_processor.ask_question(test_question, use_rag=False)
        print(f"  Success: {result_no_rag['success']}")
        print(f"  RAG Used: {result_no_rag['rag_used']}")
        
        return result['success']
        
    except Exception as e:
        print(f"ERROR: RAG processor debug failed: {e}")
        return False

async def debug_detailed_vector_search():
    print("\n[7/7] Detailed vector search debugging...")
    
    try:
        async with db_service.pool.acquire() as conn:
            stats = await db_service.get_statistics()
            total_docs = stats.get('documents', 0)
            vectorized_docs = stats.get('vectorized_documents', 0)
            
            print(f"Total documents: {total_docs}")
            print(f"Vectorized documents: {vectorized_docs}")
            print(f"Similarity threshold: {config.SIMILARITY_THRESHOLD}")
            
            if vectorized_docs > 0:
                sample_vector_doc = await conn.fetchrow(
                    "SELECT id, filename, content FROM documents WHERE content_vector IS NOT NULL LIMIT 1"
                )
                
                if sample_vector_doc:
                    doc_id = sample_vector_doc['id']
                    doc_content = sample_vector_doc['content'][:100] + "..." if len(sample_vector_doc['content']) > 100 else sample_vector_doc['content']
                    
                    print(f"\nSample vectorized document:")
                    print(f"  ID: {doc_id}")
                    print(f"  Filename: {sample_vector_doc['filename']}")
                    print(f"  Content preview: {doc_content}")
                    
                    doc_embedding = await ai_service.get_embedding(sample_vector_doc['content'])
                    
                    raw_results = await conn.fetch(f"""
                        SELECT 
                            id,
                            filename,
                            1 - (content_vector <=> $1::vector) as similarity
                        FROM documents
                        WHERE content_vector IS NOT NULL
                        ORDER BY content_vector <=> $1::vector
                        LIMIT 5
                    """, '[' + ','.join(map(str, doc_embedding)) + ']')
                    
                    print(f"\nRaw similarity scores (self-search):")
                    for row in raw_results:
                        print(f"  {row['filename']}: {row['similarity']:.4f}")
                        
                        if row['similarity'] >= config.SIMILARITY_THRESHOLD:
                            print(f"    ✓ Above threshold ({config.SIMILARITY_THRESHOLD})")
                        else:
                            print(f"    ✗ Below threshold ({config.SIMILARITY_THRESHOLD})")
            
            test_query = "technology artificial intelligence"
            query_embedding = await ai_service.get_embedding(test_query)
            
            raw_query_results = await conn.fetch(f"""
                SELECT 
                    id,
                    filename,
                    content,
                    1 - (content_vector <=> $1::vector) as similarity
                FROM documents
                WHERE content_vector IS NOT NULL
                ORDER BY content_vector <=> $1::vector
                LIMIT 5
            """, '[' + ','.join(map(str, query_embedding)) + ']')
            
            print(f"\nQuery: '{test_query}'")
            print("Raw vector search results:")
            for row in raw_query_results:
                content_preview = row['content'][:50] + "..." if row['content'] and len(row['content']) > 50 else row['content']
                print(f"  {row['filename']}: {row['similarity']:.4f}")
                print(f"    Preview: {content_preview}")
                
                if row['similarity'] >= config.SIMILARITY_THRESHOLD:
                    print(f"    ✓ Would be included")
                else:
                    print(f"    ✗ Would be excluded (below threshold)")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Detailed vector debug failed: {e}")
        return False

async def create_test_document():
    print("\n[+] Creating test document for debugging...")
    
    test_content = """
    Machine Learning Basics
    Machine learning is a subset of artificial intelligence that enables computers to learn from data.
    
    Deep Learning
    Deep learning uses neural networks with multiple layers to learn complex patterns.
    
    Natural Language Processing
    NLP focuses on enabling computers to understand and generate human language.
    
    Computer Vision
    Computer vision allows machines to interpret and understand visual information.
    """
    
    test_file = "debug_test_document.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_content)
    
    try:
        print(f"Processing test document: {test_file}")
        result = await rag_processor.process_document(test_file)
        
        print(f"Result: {result['success']}")
        print(f"Message: {result['message']}")
        print(f"Documents processed: {result['documents_processed']}")
        
        if result['success']:
            print("Waiting for embedding generation...")
            await asyncio.sleep(3)
            
            stats = await db_service.get_statistics()
            print(f"Vectorized documents after test: {stats.get('vectorized_documents', 0)}")
        
        if os.path.exists(test_file):
            os.remove(test_file)
            
        return result['success']
        
    except Exception as e:
        print(f"ERROR: Test document creation failed: {e}")
        if os.path.exists(test_file):
            os.remove(test_file)
        return False

async def main():
    print("=" * 60)
    print("VECTOR SEARCH DEBUGGING SCRIPT")
    print("=" * 60)
    
    tests = [
        ("Database Connection", debug_database_connection),
        ("Table Structure", debug_table_structure),
        ("Vector Extension", debug_vector_extension),
        ("Embedding Generation", debug_embedding_generation),
        ("Search Functionality", debug_search_functionality),
        ("RAG Processor Logic", debug_rag_processor_logic),
        ("Detailed Vector Search", debug_detailed_vector_search),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Running: {test_name}")
        print('='*40)
        
        try:
            result = await test_func()
            results[test_name] = result
            status = "PASS" if result else "FAIL"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            print(f"\n{test_name}: ERROR - {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("DEBUG SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    print("\n" + "=" * 60)
    print("RECOMMENDED ACTIONS")
    print("=" * 60)
    
    if not results.get("Vector Extension"):
        print("1. Install pgvector extension:")
        print("   Run in PostgreSQL: CREATE EXTENSION IF NOT EXISTS vector;")
    
    if results.get("Table Structure") and not results.get("Vector Extension"):
        print("2. Check if vector column exists in documents table")
    
    if not results.get("Embedding Generation"):
        print("3. Check AI service API key and connectivity")
    
    if results.get("Embedding Generation") and not results.get("Search Functionality"):
        print("4. Check if documents are being vectorized")
        print("   Run: SELECT COUNT(*) FROM documents WHERE content_vector IS NOT NULL;")
    
    if not any(results.values()):
        print("5. Create test document to verify functionality:")
        await create_test_document()
    
    print("\n" + "=" * 60)
    if passed == total:
        print("ALL TESTS PASSED ✓")
    else:
        print(f"{total - passed} TESTS FAILED - Check logs above")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())