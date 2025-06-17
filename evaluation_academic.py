import random
import numpy as np
import PyPDF2
from typing import List, Dict, Tuple
from pathlib import Path
import re

class AcademicPDFEvaluationTest:
    # academic PDF evaluation test using pre-generated evaluation statements from TrainingDataGenerator.
    # this creates a challenging test using real academic papers with complex vocabulary
    
    def __init__(self, evaluation_statements: List[Dict], pdfs_folder: str, seed: int):
        # evaluation statements from TrainingDataGenerator
        self.evaluation_statements = evaluation_statements
        self.pdfs_folder = Path(pdfs_folder)
        self.seed = seed
        
        print(f"Academic PDF Evaluation Test initialised:")
        print(f"- PDF folder: {pdfs_folder}")
        print(f"- Evaluation statements: {len(evaluation_statements)}")
    
    def generate_evaluation_dataset(self, chunk_size: int = 100, max_chunks_per_pdf: int = 50) -> Tuple[List[Dict], List[Dict]]:
        # generate academic PDF evaluation dataset with injected evaluation statements
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        print(f"Generating academic PDF evaluation dataset")
        print(f"- PDF folder: {self.pdfs_folder}")
        print(f"- Evaluation statements to inject: {len(self.evaluation_statements)}")
        print(f"- Chunk size: {chunk_size} words")
        print(f"- Max chunks per PDF: {max_chunks_per_pdf}")
        
        # incase folder doesn't exist
        if not self.pdfs_folder.exists():
            print(f"Creating {self.pdfs_folder}/ directory")
            self.pdfs_folder.mkdir(parents=True, exist_ok=True)
            print("Add PDF files to directory and run again.")
            return [], []
        
        pdf_chunks = self.extract_and_chunk_pdfs(chunk_size, max_chunks_per_pdf)
        
        if len(pdf_chunks) < len(self.evaluation_statements):
            raise ValueError(f"Need at least {len(self.evaluation_statements)} PDF chunks for injection, got {len(pdf_chunks)}")
        
        documents = self.inject_evaluation_statements(pdf_chunks)
        
        queries = self._create_evaluation_queries()
        
        print(f"Academic PDF evaluation dataset ready:")
        print(f"- Total documents: {len(documents)}")
        print(f"- Documents with injected statements: {sum(1 for d in documents if d.get('has_injection', False))}")
        print(f"- Evaluation queries: {len(queries)}")
        
        return documents, queries
    
    def extract_and_chunk_pdfs(self, chunk_size: int, max_chunks_per_pdf: int) -> List[Dict]:
        # extract text from PDFs and create chunks
        chunks = []
        
        # get all pdfs
        pdf_files = list(self.pdfs_folder.glob("*.pdf"))
        
        if not pdf_files:
            print(f"Warning: No PDFs found in {self.pdfs_folder}")
            return chunks
        
        print(f"Processing {len(pdf_files)} PDF files")
        
        for pdf_path in pdf_files:
            try:
                print(f"Processing: {pdf_path.name}")
                
                full_text = self.extract_pdf_content(pdf_path)
                
                if len(full_text.split()) < 50:
                    print(f"Skipping; too short ({len(full_text.split())} words)")
                    continue
                
                pdf_chunks = self.chunk_academic_text(full_text, pdf_path.name, chunk_size, max_chunks_per_pdf)
                chunks.extend(pdf_chunks)
                
                print(f"Extracted {len(pdf_chunks)} chunks ({len(full_text.split())} total words)")
                
            except Exception as e:
                print(f"Error processing {pdf_path.name}: {e}")
                continue
        
        print(f"Extracted {len(chunks)} total chunks from {len(pdf_files)} PDFs")
        return chunks
    
    def extract_pdf_content(self, pdf_path: Path) -> str:
        full_text = ""
        
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + " "
        
        except Exception as e:
            print(f"      PDF extraction error: {e}")
            return ""
        
        # remove excessive whitespace
        full_text = re.sub(r'\s+', ' ', full_text)
        
        # remove common PDF artifacts
        full_text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]]', ' ', full_text)
        
        # remove likely extraction errors
        words = full_text.split()
        cleaned_words = [word for word in words if 2 <= len(word) <= 25]
        
        return ' '.join(cleaned_words)
    
    def chunk_academic_text(self, text: str, source_file: str, chunk_size: int, max_chunks: int) -> List[Dict]:
        chunks = []
        words = text.split()
        
        # create overlapping chunks
        chunk_count = 0
        for i in range(0, len(words), chunk_size // 2):
            if chunk_count >= max_chunks:
                break
                
            chunk_words = words[i:i + chunk_size]
            # skip tiny chunks
            if len(chunk_words) < 20:
                continue
            
            chunk_text = ' '.join(chunk_words)
            
            chunk = {
                'id': f"pdf_eval_{len(chunks)}",
                'content': chunk_text,
                'source_type': 'academic_pdf',
                'source_file': source_file,
                'length': len(chunk_words),
                'category': 'academic_paper',
                'magic_phrase': None,
                'has_injection': False
            }
            chunks.append(chunk)
            chunk_count += 1
        
        return chunks
    
    def inject_evaluation_statements(self, pdf_chunks: List[Dict]) -> List[Dict]:
        # Inject pre-generated evaluation statements into random chunks
        # one statement per chunk
        injection_chunks = random.sample(pdf_chunks, len(self.evaluation_statements))
        
        for i, statement_info in enumerate(self.evaluation_statements):
            chunk = injection_chunks[i]
            
            # replace middle section with statement
            words = chunk['content'].split()
            if len(words) > 30:
                # middle third
                start_idx = len(words) // 3
                end_idx = 2 * len(words) // 3
                
                # insert statement
                if statement_info['complexity'] == 'simple':
                    injection_text = f"Our empirical analysis conclusively demonstrates that {statement_info['statement']}, which constitutes a fundamental discovery in our comprehensive research investigation"
                else:
                    injection_text = f"Through rigorous scientific methodology, our research unequivocally establishes that {statement_info['statement']}, thereby contributing significantly to the theoretical framework underlying this domain"
                
                new_words = words[:start_idx] + injection_text.split() + words[end_idx:]
                chunk['content'] = ' '.join(new_words)
            else:
                # shorter chunks, append
                if statement_info['complexity'] == 'simple':
                    chunk['content'] += f" Our systematic analysis reveals that {statement_info['statement']}."
                else:
                    chunk['content'] += f" Contemporary research establishes that {statement_info['statement']}."
            
            # mark chunk with info
            chunk['injected_statement'] = statement_info
            chunk['has_injection'] = True
            chunk['target_word'] = statement_info['target_word']
            chunk['magic_phrase'] = statement_info['target_word']
        
        print(f"Injected {len(self.evaluation_statements)} evaluation statements into academic PDF chunks")
        return pdf_chunks
    
    def _create_evaluation_queries(self) -> List[Dict]:
        queries = []
        
        for i, statement_info in enumerate(self.evaluation_statements):
            query = {
                'id': f"pdf_eval_query_{i}",
                'question': statement_info['query'],
                'type': statement_info['type'],
                'complexity': statement_info['complexity'],
                'target_word': statement_info['target_word'],
                'expected_statement': statement_info['statement'],
                'expected_phrase': statement_info['target_word'],
                'category': 'academic_evaluation',
                'relevant_docs': []
            }
            queries.append(query)
        
        return queries