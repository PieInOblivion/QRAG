import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from sentence_transformers import SentenceTransformer


class ClassicalRAG:
    def __init__(self, embed_dim, device):
        self.device = device
        self.embed_dim = embed_dim
        
        # use pretrained sentence transformer
        print("Loading pre-trained sentence transformer")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.encoder.to(device)
        
        # get actual embedding dimension from the model
        self.actual_embed_dim = self.encoder.get_sentence_embedding_dimension()
        print(f"Pre-trained embedding dimension: {self.actual_embed_dim}")
        
        # add projection layer if dimensions don't match
        if embed_dim != self.actual_embed_dim:
            self.projection = nn.Linear(self.actual_embed_dim, embed_dim).to(device)
            nn.init.xavier_uniform_(self.projection.weight)
        else:
            self.projection = None
        
        self.documents = []
        self.doc_embeddings = None
        self.chunks = []
        
        print(f"Classical RAG initialised with pre-trained embeddings on {self.device}")
    
    def build_index(self, documents):
        print("Building classical index with pre-trained embeddings")
        
        self.documents = documents
        self.chunks = self.chunk_documents(documents)
        
        # extract
        all_texts = [chunk['text'] for chunk in self.chunks]
        
        # encode with model
        self.doc_embeddings = self.encode_texts_pretrained(all_texts)
        
        print(f"Classical index built with {len(self.chunks)} chunks")
    
    def chunk_documents(self, documents, chunk_size: int = 100):
        chunks = []
        
        for doc in documents:
            try:
                content = doc.get('content', '')
                if not content:
                    continue
                    
                words = content.split()
                
                # overlapping chunks consistent with other RAGs
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
    
    def encode_texts_pretrained(self, texts, batch_size: int = 32):
        # encode texts with the pre-trained sentence transformer
        if not texts:
            return torch.empty(0, self.embed_dim, device=self.device)
            
        self.encoder.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # get embeddings
                embeddings = self.encoder.encode(
                    batch_texts, 
                    convert_to_tensor=True,
                    device=self.device,
                    show_progress_bar=False
                )
                
                # apply projection if needed
                if self.projection is not None:
                    embeddings = self.projection(embeddings)
                
                # normalise
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings)
        
        return torch.cat(all_embeddings, dim=0)
    
    def retrieve(self, query: str, k: int = 5):
        if self.doc_embeddings is None:
            raise ValueError("Index not built")
        
        if not query.strip():
            return []
        
        # encode query with pre-trained model
        query_embedding = self.encode_texts_pretrained([query])
        
        if query_embedding.size(0) == 0:
            return []
        
        similarities = torch.mm(query_embedding, self.doc_embeddings.T).squeeze()
        
        # handle single chunk case
        if similarities.dim() == 0:
            similarities = similarities.unsqueeze(0)
        
        top_k_values, top_k_indices = torch.topk(similarities, k=min(k, len(similarities)))
        
        results = []
        for score, idx in zip(top_k_values, top_k_indices):
            chunk = self.chunks[idx.item()].copy()
            chunk['score'] = score.item()
            results.append(chunk)
        
        return results
    
    def query(self, question: str, k: int = 5):
        start_time = time.time()
        
        try:
            retrieved_chunks = self.retrieve(question, k)
            retrieval_time = time.time() - start_time
            
            # simple answer generation
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