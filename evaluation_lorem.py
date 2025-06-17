import random
import numpy as np
from typing import List, Dict, Tuple

class LoremEvaluationTest:
    def __init__(self, evaluation_statements: List[Dict], seed: int):
        # init with evaluation statements from TrainingDataGenerator
        self.evaluation_statements = evaluation_statements
        self.seed = seed
        
        # lorem ipsum vocabulary
        self.lorem_words = [
            "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", 
            "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore",
            "dolore", "magna", "aliqua", "enim", "ad", "minim", "veniam", "quis",
            "nostrud", "exercitation", "ullamco", "laboris", "nisi", "aliquip"
        ]
        
        print(f"Lorem Evaluation Test initialised with {len(evaluation_statements)} evaluation statements")
    
    def generate_evaluation_dataset(self, num_lorem_chunks: int = 1000, chunk_size: int = 100) -> Tuple[List[Dict], List[Dict]]:
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        print(f"Generating lorem ipsum evaluation dataset")
        print(f"- Lorem chunks: {num_lorem_chunks}")
        print(f"- Evaluation statements to inject: {len(self.evaluation_statements)}")
        print(f"- Chunk size: {chunk_size} words")
        
        lorem_chunks = self.generate_lorem_chunks(num_lorem_chunks, chunk_size)
        
        documents = self.inject_evaluation_statements(lorem_chunks)

        queries = self.create_evaluation_queries()
        
        print(f"Lorem evaluation dataset ready:")
        print(f"- Total documents: {len(documents)}")
        print(f"- Documents with injected statements: {sum(1 for d in documents if d.get('has_injection', False))}")
        print(f"- Evaluation queries: {len(queries)}")
        
        return documents, queries
    
    def generate_lorem_chunks(self, num_chunks: int, chunk_size: int) -> List[Dict]:
        chunks = []
        
        for i in range(num_chunks):
            # random lorem text
            words = random.choices(self.lorem_words, k=chunk_size)
            content = ' '.join(words)
            
            chunk = {
                'id': f"lorem_eval_{i}",
                'content': content,
                'source_type': 'lorem_ipsum',
                'length': len(words),
                'category': 'lorem_baseline',
                'magic_phrase': None,
                'has_injection': False
            }
            chunks.append(chunk)
        
        return chunks
    
    def inject_evaluation_statements(self, lorem_chunks: List[Dict]) -> List[Dict]:
        # inject pre-generated evaluation statements into random lorem chunks
        if len(lorem_chunks) < len(self.evaluation_statements):
            raise ValueError(f"Need at least {len(self.evaluation_statements)} chunks for injection, got {len(lorem_chunks)}")

        # one statement per chunk
        injection_chunks = random.sample(lorem_chunks, len(self.evaluation_statements))

        # inject statements
        for i, statement_info in enumerate(self.evaluation_statements):
            chunk = injection_chunks[i]
            
            # replace middle section
            words = chunk['content'].split()
            if len(words) > 20:
                # middle third with statement
                start_idx = len(words) // 3
                end_idx = 2 * len(words) // 3
                
                # insert statement
                if statement_info['complexity'] == 'simple':
                    injection_text = f"Our experimental analysis demonstrates that {statement_info['statement']}, which represents a significant finding in our research methodology"
                else:
                    injection_text = f"Through comprehensive investigation, our research confirms that {statement_info['statement']}, establishing this as a fundamental principle in the field"
                
                new_words = words[:start_idx] + injection_text.split() + words[end_idx:]
                chunk['content'] = ' '.join(new_words)
            else:
                # for short chunks, append
                if statement_info['complexity'] == 'simple':
                    chunk['content'] += f" Our analysis reveals that {statement_info['statement']}."
                else:
                    chunk['content'] += f" Research demonstrates that {statement_info['statement']}."
            
            # mark chunk
            chunk['injected_statement'] = statement_info
            chunk['has_injection'] = True
            chunk['target_word'] = statement_info['target_word']
            chunk['magic_phrase'] = statement_info['target_word']
        
        print(f"Injected {len(self.evaluation_statements)} evaluation statements into lorem chunks")
        return lorem_chunks
    
    def create_evaluation_queries(self) -> List[Dict]:
        """Create evaluation queries from pre-generated statements"""
        queries = []
        
        for i, statement_info in enumerate(self.evaluation_statements):
            query = {
                'id': f"lorem_eval_query_{i}",
                'question': statement_info['query'],
                'type': statement_info['type'],
                'complexity': statement_info['complexity'],
                'target_word': statement_info['target_word'],
                'expected_statement': statement_info['statement'],
                'expected_phrase': statement_info['target_word'],
                'category': 'lorem_evaluation',
                'relevant_docs': []
            }
            queries.append(query)
        
        return queries