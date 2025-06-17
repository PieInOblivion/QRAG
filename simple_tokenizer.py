from typing import List, Tuple
import re
from collections import Counter

class SimpleTokenizer:
    # basic tokenizer for text preprocessing
    def __init__(self, max_vocab: int):
        self.max_vocab = max_vocab
        # unk for words not in the vocabulary, unknown
        # pad for chunks shorter than the required length
        self.word_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_built = False
    
    def preprocess_text(self, text: str) -> List[str]:
        # simple clean, anything not a word character and not space
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        return [w for w in words if len(w) > 1]
    
    def build_vocab(self, texts: List[str]):
        word_counts = Counter()
        
        for text in texts:
            words = self.preprocess_text(text)
            word_counts.update(words)
        
        most_common = word_counts.most_common(self.max_vocab - 2)
        
        for i, (word, _) in enumerate(most_common):
            word_id = i + 2
            self.word_to_id[word] = word_id
            self.id_to_word[word_id] = word
        
        self.vocab_built = True
        print(f"Built vocabulary with {len(self.word_to_id)} words")
    
    def encode(self, text: str, max_length: int = 100) -> Tuple[List[int], List[int]]:
        # text to tokens
        if not self.vocab_built:
            raise ValueError("Vocabulary not built yet")
        
        words = self.preprocess_text(text)
        
        token_ids = []
        for word in words[:max_length]:
            # 1 = unk
            token_ids.append(self.word_to_id.get(word, 1))
        
        mask = [1] * len(token_ids)
        
        while len(token_ids) < max_length:
            # 0 = pad
            token_ids.append(0)
            mask.append(0)
        
        return token_ids, mask