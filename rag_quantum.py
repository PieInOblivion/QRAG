import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import random
import os
import pickle
from typing import List, Dict, Tuple
from tqdm import tqdm

from simple_tokenizer import SimpleTokenizer

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from dataclasses import dataclass
@dataclass
class QRAGConfig:
    num_qubits: int
    epochs: int
    batch_size: int
    learning_rate: float
    embed_dim: int
    max_vocab: int
    retrieval_k: int

    def to_string(self):
        return f"Q{self.num_qubits}_E{self.epochs}_B{self.batch_size}_LR{self.learning_rate}_EB{self.embed_dim}_MV{self.max_vocab}_K{self.retrieval_k}"


class QuantumEmbeddingLayer:
    # Quantum embedding layer
    # multi-layer quantum circuits for increased expressivity
    # proper use of both real and imaginary amplitudes
    # quantum kernel-inspired feature extraction
    # parameter initialisation for quantum systems
    # circular entanglement for improved connectivity
    
    # a simpler implementation might use only the real numbers, but in order to reach the maximum potential
    # of this architecture we use both the real and imaginary amplitudes
    
    def __init__(self, num_qubits: int, embedding_dim: int, device='cpu', num_layers: int = 3):
        self.num_qubits = num_qubits
        self.embedding_dim = embedding_dim
        self.device = device
        self.num_layers = num_layers
        
        # quantum simulator
        self.simulator = AerSimulator(method='statevector')
        
        # classical preprocessing. use smaller intermediate dimension for more quantum influence
        self.input_projection = nn.Linear(embedding_dim, num_qubits).to(device)
        
        # real + imaginary parts
        quantum_feature_dim = 2 * (2 ** num_qubits)
        self.output_projection = nn.Linear(quantum_feature_dim, embedding_dim).to(device)
        
        # quantum circuit parameters
        # learnable parameters for each layer
        # 2 rotations per qubit per layer
        self.num_circuit_params = num_qubits * num_layers * 2
        self.quantum_params = nn.Parameter(
            torch.randn(self.num_circuit_params, device=device) * 0.1
        )
        
        # quantum-aware initialisation
        self.initialise_weights()
        
        print(f"Quantum embedding: {num_qubits} qubits -> {embedding_dim}D")
        print(f"  Quantum layers: {num_layers}")
        print(f"  Quantum parameters: {self.num_circuit_params}")
        print(f"  Feature dimension: {quantum_feature_dim} -> {embedding_dim}")
    
    def initialise_weights(self):
        # quantum-aware weight initialisation
        # initialise input projection for bounded outputs, important for angle encoding
        nn.init.uniform_(self.input_projection.weight, -np.pi, np.pi)
        nn.init.zeros_(self.input_projection.bias)
        
        # initialise output projection with smaller scale since quantum features can be large
        nn.init.xavier_uniform_(self.output_projection.weight, gain=0.1)
        nn.init.zeros_(self.output_projection.bias)
    
    def create_parameterised_circuit(self, input_features: torch.Tensor) -> List[QuantumCircuit]:
        # the theoretical foundation is:
        # - multiple layers increase circuit expressivity exponentially
        # - circular entanglement creates better connectivity than linear
        # - parameter sharing across samples but unique per layer/qubit
        batch_size = input_features.shape[0]
        circuits = []
        
        for batch_idx in range(batch_size):
            qc = QuantumCircuit(self.num_qubits)
            param_idx = 0
            
            sample_features = input_features[batch_idx]
            
            # multi-layer quantum circuit
            for _ in range(self.num_layers):
                # rotation layer, encode both input features and learnable parameters
                for qubit in range(self.num_qubits):
                    # combine input feature with learnable parameter
                    if qubit < len(sample_features):
                        feature_angle = sample_features[qubit].item()
                    else:
                        feature_angle = 0.0
                    
                    # add learnable parameters
                    if param_idx < len(self.quantum_params):
                        ry_angle = feature_angle + self.quantum_params[param_idx].item()
                        param_idx += 1
                    else:
                        ry_angle = feature_angle
                    
                    if param_idx < len(self.quantum_params):
                        rz_angle = feature_angle * 0.5 + self.quantum_params[param_idx].item()
                        param_idx += 1
                    else:
                        rz_angle = feature_angle * 0.5
                    
                    qc.ry(ry_angle, qubit)
                    qc.rz(rz_angle, qubit)
                
                # circular entanglement pattern
                for qubit in range(self.num_qubits):
                    next_qubit = (qubit + 1) % self.num_qubits
                    qc.cx(qubit, next_qubit)
                
                # additional layer of RY rotations for more expressivity
                for qubit in range(self.num_qubits):
                    if param_idx < len(self.quantum_params):
                        extra_angle = self.quantum_params[param_idx].item() * 0.1
                        qc.ry(extra_angle, qubit)
                        param_idx += 1
            
            circuits.append(qc)
        
        return circuits
    
    def execute_quantum_circuits(self, circuits: List[QuantumCircuit]) -> torch.Tensor:
        # execute and extract features
        # use both real and imaginary parts

        all_features = []
        
        for qc in circuits:
            # get statevector
            statevector = Statevector.from_instruction(qc)
            amplitudes = statevector.data
            
            # extract both real and imaginary
            real_parts = np.real(amplitudes)
            imag_parts = np.imag(amplitudes)
            
            # ensure correct dimensions
            expected_size = 2 ** self.num_qubits
            
            if len(real_parts) != expected_size:
                real_parts = np.resize(real_parts, expected_size)
                imag_parts = np.resize(imag_parts, expected_size)
            
            # combine real and imaginary parts preserving quantum phase information
            combined_features = np.concatenate([real_parts, imag_parts])
            
            # normalise features to prevent exploding gradients
            feature_norm = np.linalg.norm(combined_features)
            if feature_norm > 0:
                combined_features = combined_features / feature_norm
            
            all_features.append(combined_features)
        
        return torch.tensor(np.array(all_features), dtype=torch.float32, device=self.device)

    def forward(self, classical_embeddings: torch.Tensor) -> torch.Tensor:
        # project to quantum space with bounded outputs
        # bound to (-pi, pi) range
        quantum_input = self.input_projection(classical_embeddings)
        quantum_input = torch.tanh(quantum_input)
        
        # move to cpu for quantum circuit execution
        # qiskit has no gpu support
        quantum_input_cpu = quantum_input.cpu()
        
        # create and execute quantum circuits
        quantum_circuits = self.create_parameterised_circuit(quantum_input_cpu)
        quantum_features_cpu = self.execute_quantum_circuits(quantum_circuits)
        
        # move back to original device
        quantum_features = quantum_features_cpu.to(self.device)
        
        # project back to embedding space
        output = self.output_projection(quantum_features)
        
        # layer normalisation which will stablise training
        # previous architectures has expectedly wild jumps in training
        output = F.layer_norm(output, (output.size(-1),))
        
        return output


class QuantumTextEncoder:
    def __init__(self, vocab_size: int, embed_dim: int, num_qubits: int, device='cpu'):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_qubits = num_qubits
        self.device = device
        
        # classical embedding with better init
        self.classical_embedding = nn.Embedding(vocab_size, embed_dim).to(device)
        nn.init.normal_(self.classical_embedding.weight, mean=0, std=0.1)
        
        # quantum layer with multiple layers
        self.quantum_layer = QuantumEmbeddingLayer(
            num_qubits, embed_dim, device, num_layers=3
        )
        
        # output processing with residual connection
        self.output_norm = nn.LayerNorm(embed_dim).to(device)
        self.residual_weight = nn.Parameter(torch.tensor(0.5, device=device))
        
        print(f"Quantum encoder: {vocab_size} vocab -> {embed_dim}D via {num_qubits} qubits")
    
    def encode(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Encode text using quantum-enhanced embedding
        # [batch, seq_len, embed_dim]
        embeddings = self.classical_embedding(token_ids)  
        
        # apply mask and pool
        embeddings = embeddings * mask.unsqueeze(-1)
        pooled = embeddings.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        # quantum processing
        quantum_embeddings = self.quantum_layer.forward(pooled)
        
        # residual connection between classical and quantum
        # helps with training stability, and allows gradual quantum influence
        alpha = torch.sigmoid(self.residual_weight)
        combined = alpha * quantum_embeddings + (1 - alpha) * pooled
        
        # layer normalisation
        output = self.output_norm(combined)
        
        return output

    def state_dict(self):
        # for saving dict
        return {
            'classical_embedding': self.classical_embedding.state_dict(),
            'quantum_layer_input_proj': self.quantum_layer.input_projection.state_dict(),
            'quantum_layer_output_proj': self.quantum_layer.output_projection.state_dict(),
            'quantum_layer_params': self.quantum_layer.quantum_params.data,
            'output_norm': self.output_norm.state_dict(),
            'residual_weight': self.residual_weight.data
        }
    
    def load_state_dict(self, state_dict):
        self.classical_embedding.load_state_dict(state_dict['classical_embedding'])
        self.quantum_layer.input_projection.load_state_dict(state_dict['quantum_layer_input_proj'])
        self.quantum_layer.output_projection.load_state_dict(state_dict['quantum_layer_output_proj'])
        self.quantum_layer.quantum_params.data = state_dict['quantum_layer_params']
        self.output_norm.load_state_dict(state_dict['output_norm'])
        self.residual_weight.data = state_dict['residual_weight']

    def train(self):
        # set torch to train
        self.classical_embedding.train()
        self.quantum_layer.input_projection.train()
        self.quantum_layer.output_projection.train()
        self.output_norm.train()

    def eval(self):
        # set torch to eval
        self.classical_embedding.eval()
        self.quantum_layer.input_projection.eval()
        self.quantum_layer.output_projection.eval()
        self.output_norm.eval()

    def parameters(self):
        params = []
        params.extend(self.classical_embedding.parameters())
        params.extend(self.quantum_layer.input_projection.parameters())
        params.extend(self.quantum_layer.output_projection.parameters())
        params.append(self.quantum_layer.quantum_params)
        params.extend(self.output_norm.parameters())
        params.append(self.residual_weight)
        return params


class QuantumRAG:
    # quantum RAG system with training/validation split
    # multi-layer quantum circuits
    # both real and imaginary amplitude extraction
    # residual connections for training stability
    # circular entanglement patterns
    # learning rate scheduling
    
    def __init__(self, config: QRAGConfig, device='cpu'):
        self.config = config
        self.device = device
        self.model_save_path = f"{config.to_string()}.pkl"

        self.loss_history = []
        self.validation_history = []
        self.quantum_influence_history = []
        self.build_time = 0.0
        
        self.tokenizer = SimpleTokenizer(config.max_vocab)
        
        # initialised after vocabulary is built
        self.encoder = None
        
        self.documents = []
        self.chunks = []
        self.doc_embeddings = None
        
        self.is_trained = False
        
        print(f"Quantum RAG initialised: {config.embed_dim}D embeddings, {config.num_qubits} qubits on {device}")
    
    def save_model(self):
        save_data = {
            'config': self.config.__dict__,
            'encoder_state_dict': self.encoder.state_dict(),
            'tokenizer_word_to_id': self.tokenizer.word_to_id,
            'tokenizer_id_to_word': self.tokenizer.id_to_word,
            'tokenizer_vocab_built': self.tokenizer.vocab_built,
            'loss_history': self.loss_history,
            'validation_history': self.validation_history,
            'quantum_influence_history': self.quantum_influence_history,
            'vocab_size': len(self.tokenizer.word_to_id),
            'build_time': self.build_time,
            'is_trained': True
        }
        
        with open(self.model_save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Model saved to {self.model_save_path}")
    
    def load_model(self) -> bool:
        if not os.path.exists(self.model_save_path):
            return False
        
        try:
            with open(self.model_save_path, 'rb') as f:
                save_data = pickle.load(f)
            
            # tokenizer
            self.tokenizer.word_to_id = save_data['tokenizer_word_to_id']
            self.tokenizer.id_to_word = save_data['tokenizer_id_to_word']
            self.tokenizer.vocab_built = save_data['tokenizer_vocab_built']

            # training history
            self.loss_history = save_data.get('loss_history', [])
            self.validation_history = save_data.get('validation_history', [])
            self.quantum_influence_history = save_data.get('quantum_influence_history', [])
            self.build_time = save_data.get('build_time', 0.0)
            self.is_trained = save_data.get('is_trained', False)
            
            # initialise encoder
            self.encoder = QuantumTextEncoder(
                vocab_size=len(self.tokenizer.word_to_id),
                embed_dim=self.config.embed_dim,
                num_qubits=self.config.num_qubits,
                device=self.device
            )
            
            # load encoder state
            self.encoder.load_state_dict(save_data['encoder_state_dict'])
            
            print(f"Trained model loaded from {self.model_save_path}")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def train_from_generator(self, training_data: List[Dict], testing_data: List[Dict], training_queries: List[Dict], testing_queries: List[Dict]):
        print("Training quantum RAG with validation")
        
        # try to load existing model
        if self.load_model():
            print("Using pre-trained model")
            self.is_trained = True
            return
        
        build_start_time = time.time()
        
        # extract all text content for vocabulary building
        all_texts = [chunk['content'] for chunk in training_data + testing_data]
        print(f"Building vocabulary from {len(all_texts)} text chunks")
        self.tokenizer.build_vocab(all_texts)
        
        # initialise quantum encoder
        self.encoder = QuantumTextEncoder(
            vocab_size=len(self.tokenizer.word_to_id),
            embed_dim=self.config.embed_dim,
            num_qubits=self.config.num_qubits,
            device=self.device
        )
        
        # train quantum encoder with validation
        print("Training quantum encoder with validation")
        self.train_quantum_encoder_with_validation(
            training_data, testing_data, training_queries, testing_queries
        )
        
        self.build_time = time.time() - build_start_time
        self.is_trained = True
        
        self.save_model()
        
        print(f"Quantum model training complete. Build time: {self.build_time:.2f}s")

    def train_quantum_encoder_with_validation(self, training_data: List[Dict], testing_data: List[Dict], training_queries: List[Dict], testing_queries: List[Dict]):
        # create training and validation pairs seperately
        training_pairs = self.create_training_pairs_from_data(training_data, training_queries)
        validation_pairs = self.create_training_pairs_from_data(testing_data, testing_queries)
        
        print(f"Training pairs: {len(training_pairs)}")
        print(f"Validation pairs: {len(validation_pairs)}")
        
        # separate optimisers for classical and quantum parts
        classical_params = [
            *self.encoder.classical_embedding.parameters(),
            *self.encoder.output_norm.parameters(),
            self.encoder.residual_weight
        ]
        
        quantum_params = [
            *self.encoder.quantum_layer.input_projection.parameters(),
            *self.encoder.quantum_layer.output_projection.parameters(),
            self.encoder.quantum_layer.quantum_params
        ]
        
        # much slower for quantum because of the steep non-convex nature of quantum gradients
        classical_optimizer = torch.optim.Adam(classical_params, lr=self.config.learning_rate)
        quantum_optimizer = torch.optim.Adam(quantum_params, lr=self.config.learning_rate * 0.1)
        
        # cosine annealing is the current academic choice for quantum training because of its smooth roll off
        classical_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            classical_optimizer, T_max=self.config.epochs, eta_min=1e-5
        )
        quantum_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            quantum_optimizer, T_max=self.config.epochs, eta_min=1e-4
        )

        # tracking
        self.loss_history = []
        self.validation_history = []
        self.quantum_influence_history = []
        
        print(f"Training with validation for {self.config.epochs} epochs")

        epoch_pbar = tqdm(range(self.config.epochs), desc="Training Quantum with Validation", unit="epoch")
        
        for _ in epoch_pbar:
            # train
            self.encoder.train()
            train_loss = self.train_epoch(training_pairs, classical_optimizer, quantum_optimizer, classical_params, quantum_params)
            
            # val
            self.encoder.eval()
            val_loss, val_metrics = self.validate_epoch(validation_pairs)
            
            # track
            self.loss_history.append(train_loss)
            self.validation_history.append(val_loss)
            
            # log quantum influence
            quantum_influence = torch.sigmoid(self.encoder.residual_weight).item()
            self.quantum_influence_history.append(quantum_influence)
            
            # lrs
            classical_scheduler.step()
            quantum_scheduler.step()
            
            # progress bar
            epoch_pbar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_acc=f"{val_metrics['accuracy']:.3f}",
                q_influence=f"{quantum_influence:.3f}"
            )
        
        print(f"Training complete:")
        print(f"Final validation loss: {self.validation_history[-1]:.4f}")
        print(f"Final quantum influence: {quantum_influence:.3f}")

    def train_epoch(self, training_pairs, classical_optimizer, quantum_optimizer, classical_params, quantum_params):
        total_loss = 0
        num_batches = 0
        
        random.shuffle(training_pairs)
        
        for i in range(0, len(training_pairs), self.config.batch_size):
            batch_pairs = training_pairs[i:i + self.config.batch_size]
            
            text1_ids, text1_masks, text2_ids, text2_masks, labels = self.prepare_batch(batch_pairs)
            
            # forward pass
            embed1 = self.encoder.encode(text1_ids, text1_masks)
            embed2 = self.encoder.encode(text2_ids, text2_masks)
            
            # compute loss
            loss = self.compute_contrastive_loss(embed1, embed2, labels)
            
            # backward pass with separate optimisers
            classical_optimizer.zero_grad()
            quantum_optimizer.zero_grad()
            
            loss.backward()
            
            # fradient clipping
            # more conservative for quantum
            torch.nn.utils.clip_grad_norm_(classical_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(quantum_params, max_norm=0.5)
            
            classical_optimizer.step()
            quantum_optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0

    def validate_epoch(self, validation_pairs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for i in range(0, len(validation_pairs), self.config.batch_size):
                batch_pairs = validation_pairs[i:i + self.config.batch_size]
                
                text1_ids, text1_masks, text2_ids, text2_masks, labels = self.prepare_batch(batch_pairs)
                
                # forward pass
                embed1 = self.encoder.encode(text1_ids, text1_masks)
                embed2 = self.encoder.encode(text2_ids, text2_masks)
                
                # compute loss
                loss = self.compute_contrastive_loss(embed1, embed2, labels)
                total_loss += loss.item()
                
                # compute accuracy
                similarities = F.cosine_similarity(embed1, embed2)
                predictions = (similarities > 0.5).float()
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += len(labels)
        
        avg_loss = total_loss / max(1, len(validation_pairs) // self.config.batch_size)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return avg_loss, {'accuracy': accuracy}

    def compute_contrastive_loss(self, embed1, embed2, labels):
        # contrastive loss function
        # normalise embeddings
        embed1_norm = F.normalize(embed1, p=2, dim=1)
        embed2_norm = F.normalize(embed2, p=2, dim=1)
        
        # cosine similarity
        similarities = F.cosine_similarity(embed1_norm, embed2_norm)
        
        # contrastive loss with margin
        positive_mask = labels > 0.5
        negative_mask = labels < 0.5
        
        # positive pairs: maximise similarity
        pos_loss = F.mse_loss(similarities[positive_mask], labels[positive_mask]) if positive_mask.any() else torch.tensor(0.0, device=self.device)
        
        # negative pairs: minimise similarity with margin
        if negative_mask.any():
            neg_similarities = similarities[negative_mask]
            # push negative similarities below 0.2
            neg_loss = F.relu(neg_similarities - 0.2).mean()
        else:
            neg_loss = torch.tensor(0.0, device=self.device)
        
        # quantum regularization encourages parameter usage
        quantum_reg = 0.001 * torch.norm(self.encoder.quantum_layer.quantum_params)
        
        return pos_loss + neg_loss + quantum_reg

    def prepare_batch(self, batch_pairs):
        text1_ids, text1_masks = [], []
        text2_ids, text2_masks = [], []
        labels = []
        
        for text1, text2, label in batch_pairs:
            ids1, mask1 = self.tokenizer.encode(text1)
            ids2, mask2 = self.tokenizer.encode(text2)
            
            text1_ids.append(ids1)
            text1_masks.append(mask1)
            text2_ids.append(ids2)
            text2_masks.append(mask2)
            labels.append(label)
        
        return (
            torch.tensor(text1_ids, device=self.device),
            torch.tensor(text1_masks, device=self.device),
            torch.tensor(text2_ids, device=self.device),
            torch.tensor(text2_masks, device=self.device),
            torch.tensor(labels, device=self.device, dtype=torch.float)
        )
    
    def create_training_pairs_from_data(self, training_data: List[Dict], training_queries: List[Dict]) -> List[Tuple[str, str, float]]:
        pairs = []
        
        # extract all training chunks
        all_training_chunks = [chunk['content'] for chunk in training_data]
        print(f"Creating training pairs from {len(all_training_chunks)} training chunks")
        
        # query-to-Document pairs. supervised learning from injected data
        target_to_chunks = {}
        for chunk in training_data:
            if chunk.get('has_injection', False):
                statement_info = chunk['injected_statement']
                target_word = statement_info['target_word']
                
                if target_word not in target_to_chunks:
                    target_to_chunks[target_word] = []
                target_to_chunks[target_word].append(chunk['content'])
        
        # create query-document pairs
        for query_info in training_queries:
            query_text = query_info['question']
            target_word = query_info['target_word']
            
            # positive: query with chunks containing the target
            if target_word in target_to_chunks:
                for chunk_text in target_to_chunks[target_word]:
                    pairs.append((query_text, chunk_text, 1.0))
            
            # negative: query with random nontarget chunks
            other_chunks = [chunk for chunk in all_training_chunks 
                          if target_word not in chunk.lower()]
            
            if other_chunks:
                # sample multiple negative examples per query
                num_negatives = min(3, len(other_chunks))
                negative_chunks = random.sample(other_chunks, num_negatives)
                
                for chunk_text in negative_chunks:
                    pairs.append((query_text, chunk_text, 0.0))
        
        # reduce chunk-to-chunk pairs to focus on query learning
        num_chunk_pairs = min(1000, len(all_training_chunks))
        
        for _ in range(num_chunk_pairs):
            chunk1, chunk2 = random.sample(all_training_chunks, 2)
            
            # simple similarity heuristic
            words1 = set(chunk1.lower().split())
            words2 = set(chunk2.lower().split())
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            jaccard_sim = intersection / union if union > 0 else 0.0
            
            # more conservative similarity assignment
            if jaccard_sim > 0.4:
                similarity_score = 1.0
            elif jaccard_sim < 0.05:
                similarity_score = 0.0
            else:
                similarity_score = jaccard_sim
            
            pairs.append((chunk1, chunk2, similarity_score))
        
        # self similarity pairs
        self_similarity_chunks = random.sample(all_training_chunks, min(50, len(all_training_chunks)))
        for chunk in self_similarity_chunks:
            pairs.append((chunk, chunk, 1.0))
        
        random.shuffle(pairs)
        
        print(f"Created {len(pairs)} total training pairs")
        
        return pairs
    
    def build_index(self, documents: List[Dict]):
        if not self.is_trained:
            raise ValueError("Model must be trained first using train_from_generator()")
        
        print("Building quantum RAG evaluation index")
        
        self.documents = documents
        self.chunks = self.chunk_documents(documents)
        
        # create embeddings index using trained model
        all_texts = [chunk['text'] for chunk in self.chunks]
        print("Creating quantum embeddings index")
        self.doc_embeddings = self.encode_texts(all_texts)
        
        print(f"Quantum evaluation index built: {len(self.chunks)} chunks with quantum embeddings")
    
    def chunk_documents(self, documents: List[Dict], chunk_size: int = 100) -> List[Dict]:
        chunks = []
        
        for doc in documents:
            content = doc['content']
            words = content.split()
            
            # create overlapping chunks. same as other rags
            for i in range(0, len(words), chunk_size // 2):
                chunk_words = words[i:i + chunk_size]
                # skip small chunks
                if len(chunk_words) < 10:
                    continue
                
                chunk_text = ' '.join(chunk_words)
                chunks.append({
                    'text': chunk_text,
                    'doc_id': doc['id'],
                    'chunk_id': len(chunks),
                    'category': doc.get('category', 'unknown'),
                    'magic_phrase': doc.get('magic_phrase', None)
                })
        
        return chunks
    
    def encode_texts(self, texts: List[str], batch_size: int = 8) -> torch.Tensor:
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # tokenise batch
            batch_ids, batch_masks = [], []
            for text in batch_texts:
                token_ids, mask = self.tokenizer.encode(text)
                batch_ids.append(token_ids)
                batch_masks.append(mask)
            
            # convert to tensors
            batch_ids = torch.tensor(batch_ids, device=self.device)
            batch_masks = torch.tensor(batch_masks, device=self.device)
            
            # encode with quantum encoder
            with torch.no_grad():
                embeddings = self.encoder.encode(batch_ids, batch_masks)
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings)
        
        return torch.cat(all_embeddings, dim=0)
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        if self.doc_embeddings is None:
            raise ValueError("Index not built yet")
        
        # encode query
        query_embedding = self.encode_texts([query])
        
        # compute similarities
        similarities = torch.mm(query_embedding, self.doc_embeddings.T).squeeze()
        
        # get top k
        top_k_values, top_k_indices = torch.topk(similarities, k=min(k, len(similarities)))
        
        results = []
        for score, idx in zip(top_k_values, top_k_indices):
            chunk = self.chunks[idx.item()].copy()
            chunk['score'] = score.item()
            results.append(chunk)
        
        return results
    
    def query(self, question: str, k: int = 5) -> Dict:
        start_time = time.time()
        
        # get relevant chunks
        retrieved_chunks = self.retrieve(question, k)
        
        # generate answer
        # simple concatenation
        context = "\n".join([chunk['text'] for chunk in retrieved_chunks])
        answer = f"Based on the retrieved information: {context[:200]}..."
        
        total_time = time.time() - start_time
        
        return {
            'question': question,
            'answer': answer,
            'retrieved_chunks': retrieved_chunks,
            'total_time': total_time
        }