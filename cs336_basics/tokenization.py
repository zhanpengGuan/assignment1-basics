import re
import regex
import os
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional
import os
from typing import BinaryIO
PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
PAT = regex.compile(PAT)
class BPETokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer implementation.
    """
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_freqs = {}
        self.splits = {}
        self.merges = {}
        self.vocab = {}
        self.pairs = None
        self.affected_words = None
        self.maxpair = None
        self.special_tokens = {"<|endoftext|>": 0, "<|unk|>": 1}
        
    def get_word_freqs(self, text: str) -> Dict[str, int]:
        """Extract word frequencies from text."""
        # Split text by whitespace and punctuation
        words = PAT.findall( text)
        return Counter(words)
    
    def get_splits(self, word: str) -> List[str]:
        """Split word into characters with end-of-word marker."""
        return list(word) + ['</w>']
    
    def get_pairs(self, word: List[str]) -> Set[Tuple[str, str]]:
        """Get all consecutive pairs of symbols in a word."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Merge the most frequent pair in the vocabulary."""
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        new_vocab = {}
        affected_words = set()
        for word, word_list in vocab.items():
            # new_word = p.sub(''.join(pair), ' '.join(word_list))
            # new_vocab[word] = new_word.split()
            if pair[0] not in word_list or pair[1] not in word_list:
            # 跳过不包含 pair 的单词
                new_vocab[word] = word_list
                continue

            # 合并符号对
            new_word_list = []
            i = 0
            modified = False 
            while i < len(word_list):
                if i < len(word_list) - 1 and word_list[i] == pair[0] and word_list[i + 1] == pair[1]:
                    new_word_list.append(''.join(pair))
                    i += 2
                    modified = True
                else:
                    new_word_list.append(word_list[i])
                    i += 1
            if modified:
                affected_words.add(word)
            new_vocab[word] = new_word_list
        return new_vocab, affected_words
    
    def train(self, text: str):
        """Train the BPE tokenizer on the given text."""
        print("Training BPE tokenizer...")
        
        # Get word frequencies
        self.word_freqs = self.get_word_freqs(text)
        print(f"Found {len(self.word_freqs)} unique words")
        
        # Initialize vocabulary with character-level splits
        self.splits = {word: self.get_splits(word) for word in self.word_freqs.keys()}
        
        # Build initial vocabulary
        vocab = set()
        for word in self.word_freqs.keys():
            vocab.update(self.splits[word])
        
        # Add special tokens to vocabulary
        vocab.update(self.special_tokens.keys())
        
        print(f"Initial vocabulary size: {len(vocab)}")
        
        # Perform BPE merges
        num_merges = self.vocab_size - len(vocab)
        if num_merges <= 0:
            print("Vocabulary size already larger than desired size")
            return
            
        print(f"Performing {num_merges} merges...")
        
        for i in range(num_merges):
            pairs = self.get_stats(self.affected_words)
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            self.maxpair = best_pair
            self.splits, self.affected_words = self.merge_vocab(best_pair, self.splits)
            self.merges[best_pair] = i
            
            if (i + 1) % 1000 == 0:
                print(f"Completed {i + 1} merges...")
        
        # Build final vocabulary
        self.vocab = {}
        idx = 0
        
        # Add special tokens first
        for token in self.special_tokens:
            self.vocab[token] = idx
            idx += 1
            
        # Add all subwords from splits
        for word in self.splits:
            for subword in self.splits[word]:
                if subword not in self.vocab:
                    self.vocab[subword] = idx
                    idx += 1
        
        print(f"Final vocabulary size: {len(self.vocab)}")
        print(f"Number of merges performed: {len(self.merges)}")
    
    # def get_stats(self) -> Dict[Tuple[str, str], int]:
    #     """Get statistics of pairs in the current vocabulary."""
    #     pairs = defaultdict(int)
    #     for word, freq in self.word_freqs.items():
    #         symbols = self.splits[word]
    #         for i in range(len(symbols) - 1):
    #             pairs[(symbols[i], symbols[i + 1])] += freq
    #     return pairs
    
    def get_stats(self, affected_words) -> Dict[Tuple[str, str], int]:
        """
        Incrementally update the frequency of symbol pairs in the vocabulary.

        Args:
            affected_words (Optional[Set[str]]): A set of words that were affected by the last merge.
                                                If None, perform full stats initialization.

        Returns:
            Dict[Tuple[str, str], int]: Updated frequency count of each symbol pair.
        """
       

        if True:
        
            # 全量统计，重新初始化 pairs
            self.pairs = defaultdict(int) # 清空已有统计数据
            for word, freq in self.word_freqs.items():
                symbols = self.splits[word]
                for i in range(len(symbols) - 1):
                    self.pairs[(symbols[i], symbols[i + 1])] += freq
        else:
            # 增量更新
           
            for word in affected_words:
                    freq = self.word_freqs[word]
                    max_word = ''.join(self.maxpair)
                    if max_word in word+'</w>':
                        symbols = self.splits[word]
                        for i in range(len(symbols) ):
                            if max_word == symbols[i]:
                                #原始的消失了，注意这里的symbol是合并后的
                                # 合并后的+freq
                                if i>=1:
                                    self.pairs[symbols[i-1],symbols[i]] += freq
                                while i<len(symbols)-1 and symbols[i] == max_word :
                                    self.pairs[symbols[i],symbols[i+1]] += freq
                                    i += 1
                        # 合并前-freq
                        for i in range(len(symbols) ):
                            if max_word == symbols[i]:
                                
                                while i<len(symbols) and symbols[i] == max_word and i>=1 :
                                    self.pairs[symbols[i-1],max_word[0]] -= freq
                                    self.pairs[max_word[0],max_word[1]] -= freq
                                    i += 1
                                if i<len(symbols):
                                    self.pairs[symbols[i-1],symbols[i]] -= freq
                    
        return self.pairs

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using the trained BPE model."""

        
        words = PAT.findall(text)
        tokens = []
        
        for word in words:
            # Check if word is a special token
            if word in self.special_tokens:
                tokens.append(word)
                continue
                
            # Get the BPE splits for this word
            if word in self.splits:
                word_tokens = self.splits[word]
            else:
                # For unknown words, split into characters
                word_tokens = self.get_splits(word)
            
            tokens.extend(word_tokens)
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.special_tokens["<|unk|>"]) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        # Create reverse vocabulary
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        tokens = []
        for token_id in token_ids:
            if token_id in id_to_token:
                tokens.append(id_to_token[token_id])
            else:
                tokens.append("<|unk|>")
        
        # Join tokens and clean up
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()
    
    def save_vocab(self, filepath: str):
        """Save vocabulary and merges to file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("vocab_size\n")
            f.write(f"{len(self.vocab)}\n")
            f.write("vocab\n")
            for token, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(f"{token}\t{idx}\n")
            f.write("merges\n")
            for pair, idx in sorted(self.merges.items(), key=lambda x: x[1]):
                f.write(f"{pair[0]} {pair[1]}\t{idx}\n")
    
    def load_vocab(self, filepath: str):
        """Load vocabulary and merges from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse vocabulary
        vocab_start = lines.index("vocab\n") + 1
        merges_start = lines.index("merges\n")
        
        self.vocab = {}
        for line in lines[vocab_start:merges_start]:
            if line.strip():
                token, idx = line.strip().split('\t')
                self.vocab[token] = int(idx)
        
        # Parse merges
        self.merges = {}
        for line in lines[merges_start + 1:]:
            if line.strip():
                pair_str, idx = line.strip().split('\t')
                pair = tuple(pair_str.split())
                self.merges[pair] = int(idx)

    
    




def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk_for_freq_stats(boundary):
    """Process a chunk to extract word frequencies and character pair statistics."""
    start, end = boundary
    with open(f"data\chunk_{start}_{end}.txt", "r", encoding="utf-8") as f:
        chunk = f.read()
    
    # Extract word frequencies from this chunk
    words = PAT.findall(chunk)
    word_freqs = Counter(words)
    
    # Get character-level splits for all words
    splits = {}
    for word in word_freqs.keys():
        splits[word] = list(word) + ['</w>']
    
    # Calculate pair frequencies for this chunk
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        symbols = splits[word]
        for i in range(len(symbols) - 1):
            pair_freqs[(symbols[i], symbols[i + 1])] += freq
    
    return {
        'word_freqs': dict(word_freqs),
        'pair_freqs': dict(pair_freqs),
        'splits': splits
    }

def merge_chunk_statistics(chunk_stats_list):
    """Merge statistics from all chunks to get global statistics."""
    global_word_freqs = defaultdict(int)
    global_pair_freqs = defaultdict(int)
    global_splits = {}
    
    for chunk_stats in chunk_stats_list:
        # Merge word frequencies
        for word, freq in chunk_stats['word_freqs'].items():
            global_word_freqs[word] += freq
        
        # Merge pair frequencies
        for pair, freq in chunk_stats['pair_freqs'].items():
            global_pair_freqs[pair] += freq
        
        # Merge splits (should be the same for same words)
        global_splits.update(chunk_stats['splits'])
    
    return {
        'word_freqs': dict(global_word_freqs),
        'pair_freqs': dict(global_pair_freqs),
        'splits': global_splits
    }

def train_bpe_from_global_stats(global_stats, vocab_size=10000):
    """Train BPE tokenizer using global statistics by calling the existing train method."""
    print("Training BPE tokenizer with global statistics...")
    
    # Create a BPETokenizer instance
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    
    # Set the global word frequencies directly
    tokenizer.word_freqs = global_stats['word_freqs']
    
    # Initialize vocabulary with character-level splits
    tokenizer.splits = {word: list(word) + ['</w>'] for word in tokenizer.word_freqs.keys()}
    tokenizer.vocab =  defaultdict(int)
    idx = 0
    
    # 初始化词表=特殊词+初始单byte字符
    for token in tokenizer.special_tokens:
        tokenizer.vocab[token] = idx
        idx += 1
    for word in tokenizer.word_freqs.keys():
        for sub_word in word:
            if sub_word not in tokenizer.vocab:
                tokenizer.vocab[sub_word] = idx
                idx += 1
    
    while len(tokenizer.vocab)<vocab_size:
        pairs = tokenizer.get_stats(tokenizer.affected_words)
        #说明合并完了
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        tokenizer.maxpair = best_pair
        tokenizer.splits, tokenizer.affected_words = tokenizer.merge_vocab(best_pair, tokenizer.splits)
        tokenizer.merges[best_pair] = idx
        tokenizer.vocab["".join(best_pair)] = idx
        # print(f"Step {idx}: Best pair: {best_pair}")
        idx += 1

    
    print(f"Final vocabulary size: {len(tokenizer.vocab)}")
    print(f"Number of merges performed: {len(tokenizer.merges)}")
    tokenizer.vocab = {v.to_bytes(4, 'big'):k for k, v in tokenizer.vocab.items()}
    return tokenizer.vocab, tokenizer.merges


if __name__ == "__main__":
    # Process the TinyStories dataset with correct parallel BPE training
    import multiprocessing as mp
    from pathos.multiprocessing import ProcessingPool as Pool
   
    with open(r"E:\中兴工作\CS336\assignment1-basics\data\TinyStoriesV2-GPT4-valid.txt", "rb") as f:
        num_processes = 8
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        
        # Create chunk files
        print("Creating chunk files...")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            with open(f"/data/chunk_{start}_{end}.txt", "w", encoding="utf-8") as f1:
                f1.write(chunk)
        
        # Step 1: Parallel frequency statistics collection
        print("Step 1: Collecting frequency statistics from chunks in parallel...")
        with Pool(num_processes) as pool:
            chunk_stats_list = pool.map(process_chunk_for_freq_stats, zip(boundaries[:-1], boundaries[1:]))
           
        
        # Step 2: Merge statistics from all chunks
        print("Step 2: Merging global statistics...")
        global_stats = merge_chunk_statistics(chunk_stats_list)
        
        # Step 3: Train BPE with global statistics
        print("Step 3: Training BPE with global statistics...")
        final_vocab, merges = train_bpe_from_global_stats(global_stats, vocab_size=10000)
        
        if final_vocab and merges:
            # Save the final BPE model
            print("Saving final BPE model...")
            with open("bpe_model.txt", "w", encoding="utf-8") as f:
                f.write("vocab_size\n")
                f.write(f"{len(final_vocab)}\n")
                f.write("vocab\n")
                for token, idx in sorted(final_vocab.items(), key=lambda x: x[1]):
                    f.write(f"{token}\t{idx}\n")
                f.write("merges\n")
                for pair, idx in sorted(merges.items(), key=lambda x: x[1]):
                    f.write(f"{pair[0]} {pair[1]}\t{idx}\n")
            
            print("BPE training completed successfully!")
        else:
            print("BPE training failed!")

      
      
      
      
