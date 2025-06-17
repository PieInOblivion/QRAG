import random
import numpy as np
import PyPDF2
import pickle
import os
from typing import List, Dict, Tuple
from pathlib import Path
import re

class TrainingDataGenerator:
    def __init__(self, train_pdfs_folder: str, seed: int):
        self.train_pdfs_folder = Path(train_pdfs_folder)
        self.seed = seed
        
        # store 150 generated statement/query pairs
        # 50 training
        # 50 training validation
        # final evaluation
        self.all_statements = {
            'training_set': [],
            'training_test_set': [],
            'evaluation_set': []
        }
        
        # lorem ipsum vocabulary for baseline chunks
        self.lorem_words = [
            "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", 
            "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore",
            "dolore", "magna", "aliqua", "enim", "ad", "minim", "veniam", "quis",
            "nostrud", "exercitation", "ullamco", "laboris", "nisi", "aliquip"
        ]
        
    def generate_training_corpus(self, num_lorem_chunks: int = 1000, chunk_size: int = 100) -> Tuple[List[Dict], List[Dict], Dict]:
        # creates the full model training dataset
        # cache filename based on class name and parameters
        cache_filename = f"training_data_generator_seed_{self.seed}_lorem_{num_lorem_chunks}_chunk_{chunk_size}.pkl"
        
        # try from cache first
        if os.path.exists(cache_filename):
            print(f"Loading cached training corpus from {cache_filename}")
            with open(cache_filename, 'rb') as f:
                cached_data = pickle.load(f)
            
            print(f"Loaded cached data:")
            print(f"- Training: {len(cached_data['training_data'])} chunks")
            print(f"- Testing: {len(cached_data['testing_data'])} chunks") 
            print(f"- Statements: 150 total (50 per set)")
            
            # restore internal state
            self.all_statements = cached_data['all_statements']
            
            return cached_data['training_data'], cached_data['testing_data'], cached_data['statements_dict']
        
        # generate a new dataset if no cache
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        print(f"Generating training corpus:")
        print(f"- Lorem ipsum chunks: {num_lorem_chunks}")
        print(f"- PDF chunks from: {self.train_pdfs_folder}")
        print(f"- Chunk size: {chunk_size} words")
        print(f"- Cache file: {cache_filename}")
        
        print("Generating 150 diverse statement/query pairs")
        self.generate_all_statements()
        
        print("Generating lorem ipsum chunks")
        lorem_chunks = self.generate_lorem_chunks(num_lorem_chunks, chunk_size)
        
        print("Processing train_pdfs")
        pdf_chunks = self.extract_pdf_chunks(chunk_size)
        
        print("Combining and shuffling corpus")
        all_chunks = lorem_chunks + pdf_chunks
        random.shuffle(all_chunks)
        
        print(f"Total corpus: {len(all_chunks)} chunks ({len(lorem_chunks)} lorem + {len(pdf_chunks)} PDF)")
        
        # its non configurable outside the class.
        # while not as extensible, its fine for this project
        print("Splitting train/test (80/20)")
        split_idx = int(0.8 * len(all_chunks))
        train_chunks = all_chunks[:split_idx]
        test_chunks = all_chunks[split_idx:]
        
        print(f"Training chunks: {len(train_chunks)}")
        print(f"Testing chunks: {len(test_chunks)}")
        
        print("Injecting statements")
        training_data = self.inject_statements(train_chunks, 'training_set')
        testing_data = self.inject_statements(test_chunks, 'training_test_set')
        
        statements_dict = {
            'training_queries': self.create_queries('training_set'),
            'testing_queries': self.create_queries('training_test_set'),
            'evaluation_queries': self.create_queries('evaluation_set'),
            'evaluation_statements': self.all_statements['evaluation_set']
        }
        
        print(f"Saving to cache: {cache_filename}")
        try:
            cache_data = {
                'training_data': training_data,
                'testing_data': testing_data,
                'statements_dict': statements_dict,
                'all_statements': self.all_statements,
                'generation_params': {
                    'seed': self.seed,
                    'num_lorem_chunks': num_lorem_chunks,
                    'chunk_size': chunk_size,
                    'train_pdfs_folder': str(self.train_pdfs_folder)
                }
            }
            
            with open(cache_filename, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"Saved cache file")
            
        except Exception as e:
            print(f"Could not save cache: {e}")
        
        print(f"Training corpus ready:")
        print(f"- Training: {len(training_data)} chunks")
        print(f"- Testing: {len(testing_data)} chunks") 
        print(f"- Statements generated: 150 total (50 per set)")
        
        return training_data, testing_data, statements_dict
    
    def generate_all_statements(self):
        # generates all 150 diverse statement/query pairs.
        # strict categories were used here in order to match the vocabulary of the academic pdf test.
        # this results in the closest to real world data we can synthetically create.
        # it will result in ml vocabulary in ml pdf chunks, but also in medical chunks
        simple_objects = [
            # fruits
            'apple', 'banana', 'cherry', 'date', 'elderberry', 'fig', 'grape', 'honeydew', 'kiwi', 'lemon',
            'mango', 'nectarine', 'orange', 'papaya', 'quince', 'raspberry', 'strawberry', 'blueberry', 'blackberry',
            'cranberry', 'coconut', 'pineapple', 'watermelon', 'cantaloupe', 'peach', 'plum', 'apricot', 'avocado',
            'lime', 'grapefruit', 'tangerine', 'pomegranate', 'persimmon', 'dragonfruit', 'starfruit', 'passionfruit',
            
            # colours
            'crimson', 'azure', 'emerald', 'violet', 'golden', 'silver', 'bronze', 'turquoise', 'magenta', 'amber',
            'indigo', 'scarlet', 'teal', 'burgundy', 'lavender', 'coral', 'ivory', 'cobalt', 'maroon', 'beige',
            
            # sbstract concepts
            'harmony', 'serenity', 'chaos', 'equilibrium', 'symphony', 'rhythm', 'melody', 'crescendo', 'diminuendo',
            'resonance', 'vibration', 'frequency', 'amplitude', 'wavelength', 'spectrum', 'prism', 'kaleidoscope',
            
            # geometric shapes
            'hexagon', 'octagon', 'pentagon', 'rhombus', 'trapezoid', 'ellipse', 'parabola', 'hyperbola', 'spiral',
            'helix', 'fractal', 'tessellation', 'polygon', 'polyhedron', 'dodecahedron', 'icosahedron'
        ]
        
        # technical vocabulary for complex statements
        technical_terms = [
            # computer science
            'algorithm', 'framework', 'architecture', 'protocol', 'methodology', 'paradigm', 'infrastructure',
            'compiler', 'interpreter', 'runtime', 'middleware', 'gateway', 'proxy', 'cache', 'buffer', 'parser',
            'serializer', 'optimizer', 'scheduler', 'allocator', 'debugger', 'profiler', 'validator', 'renderer',
            
            # machine learning
            'transformer', 'encoder', 'decoder', 'embedding', 'gradient', 'backpropagation', 'convergence',
            'regularization', 'normalization', 'attention', 'convolution', 'pooling', 'activation', 'dropout',
            'ensemble', 'boosting', 'bagging', 'clustering', 'classification', 'regression', 'reinforcement',
            
            # physics & science
            'magnetization', 'polarization', 'diffraction', 'interference', 'superposition', 'entanglement',
            'coherence', 'decoherence', 'tunneling', 'oscillation', 'resonance', 'phase', 'amplitude', 'frequency',
            'thermodynamics', 'entropy', 'enthalpy', 'catalysis', 'synthesis', 'crystallization', 'polymerization',
            
            # medicine & biology
            'metabolism', 'synthesis', 'transcription', 'translation', 'replication', 'mitosis', 'meiosis',
            'immunization', 'vaccination', 'antibody', 'enzyme', 'protein', 'peptide', 'chromosome', 'genome',
            'biomarker', 'therapeutic', 'diagnostic', 'prophylactic', 'pharmaceutical', 'pharmacokinetics',
            
            # maths
            'integration', 'differentiation', 'optimization', 'approximation', 'interpolation', 'extrapolation',
            'correlation', 'regression', 'probability', 'statistics', 'inference', 'hypothesis', 'theorem',
            'lemma', 'corollary', 'axiom', 'proposition', 'conjecture', 'proof', 'verification', 'validation'
        ]
        
        # paired templates for simple statements (object-based)
        simple_templates = [
            # identity patterns
            {'statement': 'the result is {}', 'question': 'what is the result?'},
            {'statement': 'the outcome is {}', 'question': 'what is the outcome?'},
            {'statement': 'the value is {}', 'question': 'what is the value?'},
            {'statement': 'the answer is {}', 'question': 'what is the answer?'},
            {'statement': 'the solution is {}', 'question': 'what is the solution?'},
            {'statement': 'the product is {}', 'question': 'what is the product?'},
            {'statement': 'the output is {}', 'question': 'what is the output?'},
            {'statement': 'the finding is {}', 'question': 'what is the finding?'},
            {'statement': 'the discovery is {}', 'question': 'what is the discovery?'},
            {'statement': 'the conclusion is {}', 'question': 'what is the conclusion?'},
            
            # property assignments
            {'statement': 'the color appears as {}', 'question': 'what does the color appear as?'},
            {'statement': 'the substance becomes {}', 'question': 'what does the substance become?'},
            {'statement': 'the material transforms into {}', 'question': 'what does the material transform into?'},
            {'statement': 'the compound produces {}', 'question': 'what does the compound produce?'},
            {'statement': 'the reaction yields {}', 'question': 'what does the reaction yield?'},
            {'statement': 'the process generates {}', 'question': 'what does the process generate?'},
            
            # state descriptions
            {'statement': 'the system exhibits {}', 'question': 'what does the system exhibit?'},
            {'statement': 'the model displays {}', 'question': 'what does the model display?'},
            {'statement': 'the pattern shows {}', 'question': 'what does the pattern show?'},
            {'statement': 'the data reveals {}', 'question': 'what does the data reveal?'},
            {'statement': 'the analysis indicates {}', 'question': 'what does the analysis indicate?'},
            {'statement': 'the measurement detects {}', 'question': 'what does the measurement detect?'},
            {'statement': 'the observation confirms {}', 'question': 'what does the observation confirm?'},
            
            # quality assessments
            {'statement': 'the performance demonstrates {}', 'question': 'what does the performance demonstrate?'},
            {'statement': 'the efficiency reaches {}', 'question': 'what does the efficiency reach?'},
            {'statement': 'the accuracy achieves {}', 'question': 'what does the accuracy achieve?'},
            {'statement': 'the precision attains {}', 'question': 'what does the precision attain?'},
            {'statement': 'the quality manifests {}', 'question': 'what does the quality manifest?'},
            {'statement': 'the standard represents {}', 'question': 'what does the standard represent?'}
        ]
        
        # paired templates for complex statements (term-based)
        complex_templates = [
            # technical achievements
            {'statement': '{} achieves optimal performance', 'question': 'what achieves optimal performance?'},
            {'statement': '{} enables scalable solutions', 'question': 'what enables scalable solutions?'},
            {'statement': '{} provides robust implementation', 'question': 'what provides robust implementation?'},
            {'statement': '{} ensures reliable operation', 'question': 'what ensures reliable operation?'},
            {'statement': '{} delivers efficient computation', 'question': 'what delivers efficient computation?'},
            {'statement': '{} facilitates rapid processing', 'question': 'what facilitates rapid processing?'},
            {'statement': '{} supports distributed architecture', 'question': 'what supports distributed architecture?'},
            {'statement': '{} maintains system stability', 'question': 'what maintains system stability?'},
            {'statement': '{} guarantees data integrity', 'question': 'what guarantees data integrity?'},
            
            # research findings
            {'statement': '{} demonstrates significant improvement', 'question': 'what demonstrates significant improvement?'},
            {'statement': '{} exhibits superior characteristics', 'question': 'what exhibits superior characteristics?'},
            {'statement': '{} manifests enhanced properties', 'question': 'what manifests enhanced properties?'},
            {'statement': '{} reveals novel behavior', 'question': 'what reveals novel behavior?'},
            {'statement': '{} indicates promising potential', 'question': 'what indicates promising potential?'},
            {'statement': '{} suggests innovative applications', 'question': 'what suggests innovative applications?'},
            {'statement': '{} confirms theoretical predictions', 'question': 'what confirms theoretical predictions?'},
            {'statement': '{} validates experimental hypotheses', 'question': 'what validates experimental hypotheses?'},
            
            # process descriptions
            {'statement': '{} undergoes systematic transformation', 'question': 'what undergoes systematic transformation?'},
            {'statement': '{} experiences dynamic evolution', 'question': 'what experiences dynamic evolution?'},
            {'statement': '{} participates in complex interactions', 'question': 'what participates in complex interactions?'},
            {'statement': '{} facilitates molecular recognition', 'question': 'what facilitates molecular recognition?'},
            {'statement': '{} catalyzes chemical reactions', 'question': 'what catalyzes chemical reactions?'},
            {'statement': '{} mediates biological processes', 'question': 'what mediates biological processes?'},
            {'statement': '{} regulates cellular functions', 'question': 'what regulates cellular functions?'},
            {'statement': '{} modulates physiological responses', 'question': 'what modulates physiological responses?'},
            
            # performance characterizations
            {'statement': '{} optimizes computational efficiency', 'question': 'what optimizes computational efficiency?'},
            {'statement': '{} maximizes throughput capacity', 'question': 'what maximizes throughput capacity?'},
            {'statement': '{} minimizes latency overhead', 'question': 'what minimizes latency overhead?'},
            {'statement': '{} balances accuracy trade-offs', 'question': 'what balances accuracy trade-offs?'},
            {'statement': '{} scales with increasing complexity', 'question': 'what scales with increasing complexity?'},
            {'statement': '{} adapts to varying conditions', 'question': 'what adapts to varying conditions?'},
            {'statement': '{} responds to environmental changes', 'question': 'what responds to environmental changes?'},
            {'statement': '{} maintains equilibrium states', 'question': 'what maintains equilibrium states?'},
            
            # research methodologies
            {'statement': '{} employs advanced techniques', 'question': 'what employs advanced techniques?'},
            {'statement': '{} utilizes sophisticated algorithms', 'question': 'what utilizes sophisticated algorithms?'},
            {'statement': '{} implements novel approaches', 'question': 'what implements novel approaches?'},
            {'statement': '{} incorporates cutting-edge methods', 'question': 'what incorporates cutting-edge methods?'},
            {'statement': '{} leverages state-of-the-art technology', 'question': 'what leverages state-of-the-art technology?'},
            {'statement': '{} applies innovative strategies', 'question': 'what applies innovative strategies?'},
            {'statement': '{} combines multiple paradigms', 'question': 'what combines multiple paradigms?'},
            {'statement': '{} integrates diverse frameworks', 'question': 'what integrates diverse frameworks?'}
        ]
        
        # create statements for each set at random
        for set_name in ['training_set', 'training_test_set', 'evaluation_set']:
            self.all_statements[set_name] = []
            
            # Create fresh copies for this set
            available_simple = simple_objects.copy()
            available_technical = technical_terms.copy()
            
            # create 50 diverse statements per set
            for i in range(50):
                # mix of simple and complex statements for realistic distribution
                # first 15 are simple, reminaing 35 are complex
                if i < 15:
                    obj = random.choice(available_simple)
                    available_simple.remove(obj)
                    
                    template = random.choice(simple_templates)
                    statement = template['statement'].format(obj)
                    query = template['question']
                    
                    statement_type = 'simple'
                    target_word = obj
                    
                else:
                    term = random.choice(available_technical)
                    available_technical.remove(term)
                    
                    template = random.choice(complex_templates)
                    statement = template['statement'].format(term)
                    query = template['question']
                    
                    statement_type = 'complex'
                    target_word = term
                
                self.all_statements[set_name].append({
                    'type': statement_type,
                    'statement': statement,
                    'query': query,
                    'target_word': target_word,
                    'complexity': 'simple' if i < 15 else 'complex'
                })
            
            # shuffle statements within each set for random distribution
            random.shuffle(self.all_statements[set_name])
            
            print(f"Generated {len(self.all_statements[set_name])} statements for {set_name}")
            print(f"  - Simple statements: {sum(1 for s in self.all_statements[set_name] if s['complexity'] == 'simple')}")
            print(f"  - Complex statements: {sum(1 for s in self.all_statements[set_name] if s['complexity'] == 'complex')}")
    
    def generate_lorem_chunks(self, num_chunks: int, chunk_size: int) -> List[Dict]:
        chunks = []
        
        for i in range(num_chunks):
            words = random.choices(self.lorem_words, k=chunk_size)
            content = ' '.join(words)
            
            chunk = {
                'id': f"lorem_{i}",
                'content': content,
                'source_type': 'lorem_ipsum',
                'length': len(words)
            }
            chunks.append(chunk)
        
        return chunks
    
    def extract_pdf_chunks(self, chunk_size: int) -> List[Dict]:
        chunks = []
        
        if not self.train_pdfs_folder.exists():
            print(f"Warning: {self.train_pdfs_folder} not found")
            return chunks
        
        pdf_files = list(self.train_pdfs_folder.glob("**/*.pdf"))
        
        if not pdf_files:
            print(f"Warning: No PDFs found in {self.train_pdfs_folder}")
            return chunks
        
        print(f"Processing {len(pdf_files)} PDF files")
        
        for pdf_path in pdf_files:
            try:
                # get only the tex from each pdf
                full_text = self.extract_pdf_content(pdf_path)
                
                if len(full_text.split()) < 50:
                    continue
                
                # create overlappying chunks
                words = full_text.split()
                for i in range(0, len(words), chunk_size // 2):
                    chunk_words = words[i:i + chunk_size]
                    if len(chunk_words) < 20:
                        # skip small chunks
                        continue
                    
                    content = ' '.join(chunk_words)
                    chunk = {
                        'id': f"pdf_{len(chunks)}",
                        'content': content,
                        'source_type': 'pdf',
                        'source_file': pdf_path.name,
                        'length': len(chunk_words)
                    }
                    chunks.append(chunk)
                    
            except Exception as e:
                # not every pdf in the dataset extracts properly
                print(f"Error processing {pdf_path.name}: {e}")
                continue
        
        print(f"Extracted {len(chunks)} chunks from PDFs")
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
            # not every pdf in the dataset extracts properly
            print(f"PDF extraction error: {e}")
            return ""
        
        # basic text cleaning
        full_text = re.sub(r'[^\w\s]', ' ', full_text)
        return ' '.join(full_text.split())
    
    def inject_statements(self, chunks: List[Dict], statement_set: str) -> List[Dict]:
        # inject statements into random chunks
        if len(chunks) < 50:
            raise ValueError(f"Need at least 50 chunks for injection, got {len(chunks)}")
        
        # select 50 random chunks for injection
        injection_chunks = random.sample(chunks, 50)
        injection_statements = self.all_statements[statement_set].copy()
        
        # create a mapping for quick lookup
        injected_chunks = set()
        
        # inject one statement per chunk
        for i, statement_info in enumerate(injection_statements):
            chunk = injection_chunks[i]
            
            # replace middle section with statement
            words = chunk['content'].split()
            if len(words) > 20:
                # middle third
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
                # for short chunks append statement
                if statement_info['complexity'] == 'simple':
                    chunk['content'] += f" Our analysis reveals that {statement_info['statement']}."
                else:
                    chunk['content'] += f" Research demonstrates that {statement_info['statement']}."
            
            # add mark chunk info
            chunk['injected_statement'] = statement_info
            chunk['has_injection'] = True
            injected_chunks.add(chunk['id'])
        
        # mark clean chunks
        for chunk in chunks:
            if chunk['id'] not in injected_chunks:
                chunk['has_injection'] = False
                chunk['injected_statement'] = None
        
        print(f"Injected {len(injection_statements)} diverse statements into {statement_set}")
        return chunks
    
    def create_queries(self, statement_set: str) -> List[Dict]:
        queries = []
        
        for i, statement_info in enumerate(self.all_statements[statement_set]):
            query = {
                'id': f"{statement_set}_query_{i}",
                'question': statement_info['query'],
                'type': statement_info['type'],
                'complexity': statement_info['complexity'],
                'target_word': statement_info['target_word'],
                'expected_statement': statement_info['statement']
            }
            queries.append(query)
        
        return queries