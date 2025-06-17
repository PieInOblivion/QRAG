import time
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    # results from RAG evaluation
    system_name: str
    total_queries: int
    correct_retrievals: int
    accuracy: float
    avg_response_time: float
    category_breakdown: Dict[str, Dict[str, Any]]
    
class RAGEvaluationRunner:
    # uses pregenerated evaluation statements
    def __init__(self):
        self.results = []
    
    def evaluate_rag_system(self, rag_system, documents: List[Dict], queries: List[Dict], system_name: str, k: int = 5) -> EvaluationResult:
        # can evaluate all three RAG systems, since they all implement build_index() and query()
        print(f"Evaluating {system_name}...")
        print(f"- Documents: {len(documents)}")
        print(f"- Queries: {len(queries)}")
        print(f"- Retrieval k: {k}")
        
        # build index
        start_time = time.time()
        rag_system.build_index(documents)
        index_time = time.time() - start_time
        print(f"- Index build time: {index_time:.3f}s")
        
        correct_retrievals = 0
        total_response_time = 0
        category_stats = {}
        query_results = []
        
        for query in queries:
            # execute query
            start_time = time.time()
            response = rag_system.query(query['question'], k=k)
            response_time = time.time() - start_time
            total_response_time += response_time
            
            # was target retrieved
            target_word = query['target_word']
            retrieved_correctly = self.check_retrieval_success(response['retrieved_chunks'], target_word)
            
            if retrieved_correctly:
                correct_retrievals += 1
            
            # per category performance
            category = query.get('complexity', 'unknown')
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'correct': 0}
            
            category_stats[category]['total'] += 1
            if retrieved_correctly:
                category_stats[category]['correct'] += 1
            
            # individual results
            query_results.append({
                'query_id': query['id'],
                'question': query['question'],
                'target_word': target_word,
                'retrieved_correctly': retrieved_correctly,
                'response_time': response_time,
                'top_score': response['retrieved_chunks'][0]['score'] if response['retrieved_chunks'] else 0.0
            })
        
        # final metrics
        accuracy = correct_retrievals / len(queries) if queries else 0.0
        avg_response_time = total_response_time / len(queries) if queries else 0.0
        
        # category breakdown
        category_breakdown = {}
        for category, stats in category_stats.items():
            category_breakdown[category] = {
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0,
                'total_queries': stats['total'],
                'correct_retrievals': stats['correct']
            }
        
        result = EvaluationResult(
            system_name=system_name,
            total_queries=len(queries),
            correct_retrievals=correct_retrievals,
            accuracy=accuracy,
            avg_response_time=avg_response_time,
            category_breakdown=category_breakdown
        )
        
        # store for comparison
        self.results.append({
            'result': result,
            'query_results': query_results,
            'index_time': index_time
        })
        
        return result
    
    def check_retrieval_success(self, retrieved_chunks: List[Dict], target_word: str) -> bool:
        # checks if the target word is in the top k retrieved chunks, true if found
        if not retrieved_chunks:
            return False
        
        target_lower = target_word.lower()
        
        for chunk in retrieved_chunks:
            chunk_text = chunk.get('text', '').lower()
            if target_lower in chunk_text:
                return True
        
        return False
    
    def print_results(self, result: EvaluationResult):
        print(f"{result.system_name} Results")
        print(f"Overall Performance:")
        print(f"  Correct retrievals: {result.correct_retrievals}/{result.total_queries}")
        print(f"  Accuracy: {result.accuracy:.1%}")
        print(f"  Avg response time: {result.avg_response_time:.3f}s")
        
        if result.category_breakdown:
            print(f"Performance by Complexity:")
            for category, metrics in result.category_breakdown.items():
                print(f"  {category.title()}: {metrics['accuracy']:.1%} ({metrics['correct_retrievals']}/{metrics['total_queries']})")