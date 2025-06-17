import random
import time
from typing import List, Dict


class RandomRAG:
    # simply returns random chunks regardless of query
    def __init__(self):
        self.documents = []
        self.chunks = []
        
        print(f"Random RAG initialised. Returns random chunks for any query")
    
    def build_index(self, documents: List[Dict]):
        # only need to store chunks
        self.documents = documents
        self.chunks = self.chunk_documents(documents)
        
        print(f"Random index built with {len(self.chunks)} chunks")
    
    def chunk_documents(self, documents: List[Dict], chunk_size: int = 100) -> List[Dict]:
        # Split documents into chunks
        chunks = []
        
        for doc in documents:
            try:
                content = doc.get('content', '')
                if not content:
                    continue
                    
                words = content.split()
                
                # overlapping chunks consistent with the other RAG systems
                for i in range(0, len(words), chunk_size // 2):
                    chunk_words = words[i:i + chunk_size]
                    # skip small chunks
                    if len(chunk_words) < 10:
                        continue
                    
                    chunk_text = ' '.join(chunk_words)
                    
                    chunk = {
                        'doc_id': doc.get('id', f'doc_{len(chunks)}'),
                        'chunk_id': len(chunks),
                        'text': chunk_text,
                        'category': doc.get('category', 'unknown'),
                        'magic_phrase': doc.get('magic_phrase', None)
                    }
                    chunks.append(chunk)
                    
            except Exception as e:
                print(f"Error processing document {doc.get('id', 'unknown')}: {e}")
                continue
        
        return chunks
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        if not self.chunks:
            return []
        
        if not query.strip():
            return []
        
        # randomly select k chunks
        num_chunks_to_return = min(k, len(self.chunks))
        selected_chunks = random.sample(self.chunks, num_chunks_to_return)
        
        # add random scores for consistency with other systems
        results = []
        for chunk in selected_chunks:
            chunk_copy = chunk.copy()
            chunk_copy['score'] = random.random()
            results.append(chunk_copy)
        
        # sort by highest first score for consistency
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
    def query(self, question: str, k: int = 5) -> Dict:
        start_time = time.time()
        
        try:
            retrieved_chunks = self.retrieve(question, k)
            retrieval_time = time.time() - start_time
            
            # Simple answer generation
            if retrieved_chunks:
                answer = retrieved_chunks[0]['text'][:200] + "..."
            else:
                answer = "No relevant information found."
            
            return {
                'question': question,
                'answer': answer,
                'retrieved_chunks': retrieved_chunks,
                'retrieval_time': retrieval_time,
                'generation_time': 0.001,
                'total_time': retrieval_time + 0.001
            }
            
        except Exception as e:
            print(f"Error processing query '{question}': {e}")
            return {
                'question': question,
                'answer': "Error processing query.",
                'retrieved_chunks': [],
                'retrieval_time': 0.0,
                'generation_time': 0.0,
                'total_time': time.time() - start_time
            }