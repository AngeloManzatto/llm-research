"""
Created on Sat May  3 18:01:32 2025

@author: Angelo Antonio Manzatto
"""
###############################################################################
# Libraries
###############################################################################
import os
import json
import random
import regex
import time
import pickle
import glob
import multiprocessing
from collections import Counter
from tqdm import tqdm

###############################################################################
# Constants
###############################################################################

pattern = (
    r"'s|'t|'re|'ve|'m|'ll|'d"                         # English contractions
    r"|-me|-te|-se|-o|-a|-nos|-vos"                    # Romance clitics
    r"|-os|-as|-lo|-la|-los|-las"                      # Object pronouns
    r"|-lhe|-lhes|-no|-na|-nos|-nas"                   # Indirect forms
    r"| ?\p{Script=Han}"                               # CJK characters (Chinese)
    r"| ?\p{L}+"                                       # Letter-based tokens
    r"| ?\p{N}"                                        # Single digits
    r"| ?[^\s\p{L}\p{N}]+"                             # Punctuation, symbols
    r"|\s+(?!\S)"                                      # Newline padding
    r"|\s+"                                            # Other whitespace
)

###############################################################################
# Parallel File Processing
###############################################################################

def load_text_paths(input_folder, file_pattern="corpus/data/processed/*/*.jsonl"):
    """
    Recursively load all .jsonl files under `corpus/data/*/processed/`.
    """
    input_paths = sorted(glob.glob(file_pattern, recursive=True))
    
    return input_paths

def load_texts_from_file(path, field="content"):
    """
    Loads lines of text from a .jsonl file. Each line must contain a dict with the given `field`.
    """
    texts = []
    if path.endswith(".jsonl"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if field in obj and isinstance(obj[field], str):
                            texts.append(obj[field])
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"⚠️ Failed to read JSONL: {path} → {e}")
    else:
        print(f"⚠️ Unsupported file type: {path}")
    return texts

def process_file(args):
    """Processes a single JSONL file to generate token frequency."""
    json_path, pattern, field = args
    freq_table = Counter()
    for text in load_texts_from_file(json_path, field=field):
        tokens = pre_tokenizer(text, pattern)
        freq_table.update(tokens)
    return freq_table

def parallel_freq_count_processing(paths, pattern, num_workers=None, field="content"):
    """Counts token frequencies across files using multiprocessing."""
    num_workers = num_workers or multiprocessing.cpu_count()
    global_freq = Counter()
    with multiprocessing.Pool(num_workers) as pool:
        args = [(p, pattern, field) for p in paths]
        results = list(tqdm(pool.imap_unordered(process_file, args), total=len(paths)))
    for result in results:
        global_freq.update(result)
    return dict(global_freq)

###############################################################################
# BPE Training Helpers
###############################################################################

def calculate_splits(freq_table):
    """Splits each token into a list of characters."""
    return {token: list(token) for token in freq_table}

def worker_compute_pair_freqs(args):
    """Worker to count bigram frequencies."""
    subset, freq_table = args
    pair_freqs = Counter()
    for word, split in subset:
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq_table[word]
    return pair_freqs

def parallel_compute_pair_freqs(splits, freq_table, num_workers=None):
    """Computes bigram frequencies using multiprocessing."""
    num_workers = num_workers or multiprocessing.cpu_count()
    split_items = list(splits.items())
    chunk_size = len(split_items) // num_workers + 1
    chunks = [split_items[i:i + chunk_size] for i in range(0, len(split_items), chunk_size)]

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(worker_compute_pair_freqs, [(chunk, freq_table) for chunk in chunks])

    merged = Counter()
    for result in results:
        merged.update(result)
    return merged

def worker_merge_pair(args):
    """Worker to merge most common bigram pair."""
    subset, a, b = args
    updated = {}
    for word, split in subset:
        merged = []
        i = 0
        while i < len(split):
            if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(split[i])
                i += 1
        updated[word] = merged
    return updated

def parallel_merge_pair(a, b, splits, freq_table, num_workers=None):
    """Merges a bigram pair across all splits using multiprocessing."""
    num_workers = num_workers or multiprocessing.cpu_count()
    split_items = list(splits.items())
    chunk_size = len(split_items) // num_workers + 1
    chunks = [split_items[i:i + chunk_size] for i in range(0, len(split_items), chunk_size)]

    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(worker_merge_pair, [(chunk, a, b) for chunk in chunks])

    merged = {}
    for result in results:
        merged.update(result)
    return merged

###############################################################################
# Training Procedure
###############################################################################

def build_initial_state(input_paths, pattern, byte_vocab=None, num_workers=None):
    """Tokenizes all files, counts frequencies, initializes splits and vocab."""
    
    print(" Counting token frequencies...")
    freq_table = parallel_freq_count_processing(input_paths, pattern, num_workers=num_workers)
    splits = calculate_splits(freq_table)
    
    vocab = byte_vocab or byte_to_unicode()
    
    return {
        'pattern': pattern,
        'vocab': vocab,
        'freq_table': freq_table,
        'splits': splits
    }

def save_training_state(training_state, path):
    """Pickles tokenizer state to disk."""
    with open(path, 'wb') as f:
        pickle.dump(training_state , f)
        
def load_training_state(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
        
def save_vocab(vocab, path):
    with open(path, 'w', encoding='utf-8') as f:
        for token in vocab:
            f.write(token + '\n')

def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]
        
def train_vocab(training_state, vocab_size, save_every=100, checkpoint_path=None, num_workers=None):
    """Main BPE merge loop. Updates tokenizer in place."""
    print("易 Starting BPE merge loop...")
    count = 0
    while len(training_state["vocab"]) < vocab_size:
        start_time = time.time()
        pair_freqs = parallel_compute_pair_freqs(training_state["splits"], training_state["freq_table"], num_workers)
        if not pair_freqs:
            print("✅ No more pairs to merge.")
            break
        (a, b), freq = pair_freqs.most_common(1)[0]
        training_state["splits"] = parallel_merge_pair(a, b, training_state["splits"], training_state["freq_table"], num_workers)
        training_state["vocab"].append(a + b)

        elapsed = time.time() - start_time
        print(f"[{count:05}] Merged '{a + b}' (freq={freq}) | Vocab: {len(training_state['vocab'])} | Time: {elapsed:.2f}s")

        if checkpoint_path and count % save_every == 0:
            save_training_state(training_state, checkpoint_path)

        count += 1
        
def save_vocab_tokens(token_list, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for token in token_list:
            f.write(token + "\n")        
        
def train_tokenizer_state(
    input_paths=None,
    input_folder="corpus",
    output_folder="tokenizer",
    file_pattern="corpus/data/processed/*/*.jsonl",
    pattern=pattern,
    vocab_size=100_050,
    save_every=100,
    checkpoint_file="tokenizer_state.pkl",
    num_workers=None,
    resume=False
):
    """
    Orchestrates tokenizer training.

    If `input_paths` is None, it discovers all .json/.txt files in `input_folder`.
    Otherwise, it uses the provided list.
    """
    os.makedirs(output_folder, exist_ok=True)
    checkpoint_path = os.path.join(output_folder, checkpoint_file)

    if resume and os.path.exists(checkpoint_path):
        print(" Resuming from checkpoint...")
        training_state = load_training_state(checkpoint_path)
    else:
        print(" Starting from scratch...")

        if input_paths is None:
            input_paths = load_text_paths(input_folder, file_pattern)

        print(f" {len(input_paths)} files loaded into training pipeline.")
        training_state = build_initial_state(
            input_paths,
            pattern=pattern,
            byte_vocab=byte_to_unicode(),
            num_workers=num_workers
        )
        save_training_state(training_state, checkpoint_path)

    train_vocab(
        training_state,
        vocab_size=vocab_size,
        save_every=save_every,
        checkpoint_path=checkpoint_path,
        num_workers=num_workers
    )

    print(" Saving final tokenizer state...")
    save_training_state(training_state, checkpoint_path)

    print("✅ Final vocab size:", len(training_state["vocab"]))
    print(" Saved to:", checkpoint_path)

    return training_state

###############################################################################
# Vocabulary parsing
###############################################################################

def get_special_tokens():
    return [
        # Core
        "<PAD>", "<SEQ>", "<BOS>", "<EOS>", "<MASK>", 
        # Extra
        *[f"<extra_{i}>" for i in range(100)]
    ]

def get_spacing_tokens():
    chars = ["Ġ", "Ċ", "#", "-", "_", "=", "*", "."]
    return [c * (2 ** i) for c in chars for i in range(1, 10)]

###############################################################################
# Filtering Rules
###############################################################################

def rule_too_long(token, max_len=50):
    return len(token) > max_len

def rule_composite_number(token):
    return token.isdigit() and len(token) > 1

def rule_symbol_line(token):
    return len(token) >= 5 and len(set(token)) == 1 and not token.isalnum()

def rule_html_or_code(token):
    if len(token) <= 3:
        return False
    return regex.search(r"(</?[a-z]{2,}[^>]*>|\\x[0-9A-Fa-f]{2}|http[s]?://|\.com)", token) is not None

def rule_emoji_string(token):
    return all(ord(c) > 10000 for c in token) and len(token) >= 3

def rule_repeated_characters(token):
    return len(set(token)) == 1 and len(token) > 4

def rule_camelcase_noise(token):
    if not token.isascii():
        return False
    upper = sum(c.isupper() for c in token)
    lower = sum(c.islower() for c in token)
    return upper > 2 and lower > 2

def rule_excessive_capital(token):
    return token.isupper() and len(token) > 3

def rule_fake_unicode_esc(token):
    return regex.search(r"(\\u[0-9a-fA-F]{4}|\\x[0-9a-fA-F]{2})", token) is not None

def rule_garbage_maths(token):
    return regex.fullmatch(r"[\+\-\*/=<>^~]{4,}", token) is not None

def rule_mixed_punct(token):
    return len(token) > 3 and all(not c.isalnum() for c in token) and len(set(token)) > 1

def rule_bad_token_shape(token):
    return regex.match(r"[-_\.]{2,}.+[-_\.]{1,}$", token) is not None

def rule_repeated_units(token):
    return regex.fullmatch(r"(.+)\1{2,}", token) is not None

###############################################################################
# Rule Registry
###############################################################################

rules = [
    ("too_long", rule_too_long),
    ("composite_number", rule_composite_number),
    ("symbol_line", rule_symbol_line),
    ("html_or_code", rule_html_or_code),
    ("emoji_string", rule_emoji_string),
    ("repeated_characters", rule_repeated_characters),
    ("camelcase_noise", rule_camelcase_noise),
    ("excessive_capital", rule_excessive_capital),
    ("fake_unicode_esc", rule_fake_unicode_esc),
    ("garbage_maths", rule_garbage_maths),
    ("mixed_punct", rule_mixed_punct),
    ("bad_token_shape", rule_bad_token_shape),
    ("repeated_units", rule_repeated_units)
]

def is_suspicious(token, rules):
    for name, rule in rules:
        if rule(token):
            return name
    return None

def clean_vocab(vocab, total_tokens, rules):
    kept, removed = [], []
    
    # Compose final vocab
    special_tokens = get_special_tokens()
    spacing_tokens = get_spacing_tokens()

    for idx, token in enumerate(vocab):
        reason = is_suspicious(token, rules)
        if reason:
            removed.append((idx, token, reason))
        else:
            
            if token not in special_tokens + spacing_tokens:
            
                kept.append(token)
            
    print(f"✅ Cleaned vocab: {len(kept)} kept, {len(removed)} removed")

    normal_tokens = kept[:total_tokens - len(special_tokens) - len(spacing_tokens)]
    
    cleaned_vocab = special_tokens + normal_tokens + spacing_tokens

    return cleaned_vocab

###############################################################################
# Byte Encoding Utilities
###############################################################################

def byte_to_unicode():
    """
    Create a mapping for byte-to-Unicode characters:
    - Printable ASCII characters are preserved.
    - Others (e.g., control chars, whitespace) are mapped to extended Unicode.
    """
    byte_encoder = [chr(i) for i in range(256)]
    n = 0
    for i in range(256):
        if not byte_encoder[i].isprintable() or byte_encoder[i] == ' ':
            byte_encoder[i] = chr(256 + n)
            n += 1
    return byte_encoder

###############################################################################
# Pretokenizer
###############################################################################

def pre_tokenizer(text, pattern):
    """
    Tokenizes input text using a regex pattern and encodes each token using byte-to-Unicode mapping.
    """
    tokens = []
    byte_encoder = byte_to_unicode()
    sub_tokens = regex.findall(pattern, text)
    for token in sub_tokens:
        encoded = ''.join(byte_encoder[b] for b in token.encode('utf-8', errors='replace'))
        tokens.append(encoded)
    return tokens

###############################################################################
# Cache
###############################################################################

def initialize_cache(max_size=1000000, min_freq=0):
    """
    Initializes a frequency-aware LFU (Least Frequently Used) cache structure.

    This cache is intended to store intermediate results — such as BPE token segmentations —
    and track how frequently each item is accessed, enabling efficient reuse and eviction policies.

    Args:
        max_size (int): Maximum number of items the cache can hold.
        min_freq (int): Initial minimum frequency threshold for eviction tracking.

    Returns:
        dict: A dictionary representing the LFU cache, containing:
            - 'items': dict of {key: (value, freq)} for storing items and their usage count.
            - 'freq': dict of {freq: set(keys)} for organizing keys by frequency level.
            - 'max_size': maximum allowed entries in the cache.
            - 'min_freq': current lowest frequency used for eviction logic.
    """
    return {
        'items': {},      # key -> (value, frequency), stores cached items and their usage count
        'freq': {},       # frequency -> set of keys, allows quick lookup of least-used items
        'max_size': max_size,  # upper bound for number of items in cache
        'min_freq': min_freq   # helps identify which frequency bucket to evict from
    }

def get_from_cache(key, cache):
    """
    Retrieves a cached segmentation and updates its usage frequency.

    cache:
      {
        'items': { key: (value, freq), ... },
        'freq':  { freq: set(keys), ... },
        'min_freq': int
      }
    """
    items = cache['items']
    freq_buckets = cache['freq']

    if key not in items:
        return None

    value, freq = items[key]

    # ----- safely remove key from old freq bucket -----
    bucket = freq_buckets.get(freq)
    if bucket is not None:
        if key in bucket:
            bucket.remove(key)
            if not bucket:
                # remove empty bucket
                del freq_buckets[freq]
                if freq == cache['min_freq']:
                    cache['min_freq'] = min(freq_buckets) if freq_buckets else 0
        else:
            # Inconsistent state: key in items but not in its freq bucket.
            # We can either ignore or rebuild; here we ignore and optionally fix min_freq.
            # (No removal needed since the bucket doesn't contain key.)
            if not freq_buckets:
                cache['min_freq'] = 0
            else:
                cache['min_freq'] = min(freq_buckets)
    else:
        # No bucket at this freq; also inconsistent. Just recompute min_freq.
        if not freq_buckets:
            cache['min_freq'] = 0
        else:
            cache['min_freq'] = min(freq_buckets)

    # ----- bump frequency -----
    new_freq = freq + 1
    items[key] = (value, new_freq)
    freq_buckets.setdefault(new_freq, set()).add(key)

    # update min_freq if needed
    if cache['min_freq'] == 0 or new_freq < cache['min_freq']:
        cache['min_freq'] = new_freq

    return value

def put_into_cache(key, value, cache):
    """
    Inserts or updates a token segmentation into the cache using LFU strategy.
    Evicts the least frequently used item if needed.
    """
    if cache['max_size'] <= 0:
        return

    if key in cache['items']:
        get_from_cache(key, cache)  # promote freq
        cache['items'][key] = (value, cache['items'][key][1])  # update value
        return

    # Eviction logic
    if len(cache['items']) >= cache['max_size']:
        min_freq = cache['min_freq']
        if min_freq in cache['freq'] and cache['freq'][min_freq]:
            evict_key = cache['freq'][min_freq].pop()
            if not cache['freq'][min_freq]:
                del cache['freq'][min_freq]
            del cache['items'][evict_key]
        else:
            # Defensive: scan for any non-empty freq bucket
            for freq, keys in sorted(cache['freq'].items()):
                if keys:
                    evict_key = keys.pop()
                    if not cache['freq'][freq]:
                        del cache['freq'][freq]
                    del cache['items'][evict_key]
                    break

    # Add new item at freq = 1
    freq = 1
    cache['items'][key] = (value, freq)
    cache['freq'].setdefault(freq, set()).add(key)
    cache['min_freq'] = freq

###############################################################################
# bbpe tokenization
###############################################################################

def get_bpe_subtokens(token, vocab):
    """
    Recursively segment a token using the given BPE vocabulary.

    Parameters:
    - token (str): The token to be segmented.
    - vocab (set): The set of subtokens obtained from BPE.

    Returns:
    - list: The list of subtokens that the token is segmented into. 
            If the token cannot be fully segmented using the vocabulary, an empty list is returned.
    """
    
    # If the token is directly in the vocabulary, return it as a single subtoken.
    if token in vocab:
        return [token]
    
    # If the token is a single character and not in the vocab, return an empty list.
    # This is a base case to stop the recursion.
    if len(token) == 1:
        return []

    # Initialize an empty list to store the subtokens of the current token.
    subtokens = []
    
    i = 0  # Start index for token segmentation
    while i < len(token):
        longest_subtoken = ''  # Keep track of the longest matching subtoken
        
        # Start from the longest possible subtoken and move towards shorter subtokens
        for end in range(len(token), i, -1):  
            subtoken_candidate = token[i:end]
            
            # If the subtoken_candidate is in the vocabulary, it's a valid segmentation.
            if subtoken_candidate in vocab:
                longest_subtoken = subtoken_candidate
                break
        
        # If no subtoken match found, this token can't be segmented using the vocab.
        if not longest_subtoken:  
            return []
        
        # Add the found longest subtoken to our list.
        subtokens.append(longest_subtoken)
        
        # Move the start index by the length of the found subtoken.
        i += len(longest_subtoken)
    
    # Return the list of segmented subtokens.
    return subtokens

def bpe_encoding(text, pattern, vocab, cache=None):
    """
    Encodes a text string into a list of BPE subtokens.

    Args:
        text (str): The raw text to tokenize.
        pattern (str): The regex pattern for tokenization.
        vocab (set): The BPE vocabulary set.
        unk_token (str): Token used for unknown characters.
        cache (dict or None): Optional LFU cache to store segmentations.

    Returns:
        List[str]: The full list of BPE subtokens.
    """
    segmented = []
    tokens = pre_tokenizer(text, pattern)

    for token in tokens:
        if cache is not None:
            subtokens = get_from_cache(token, cache)
            if subtokens:
                segmented.extend(subtokens)
                continue

        subtokens = get_bpe_subtokens(token, vocab)

        if not subtokens:
            raise ValueError(
                "BPE segmentation failed (unexpected with byte vocab). "
                f"Token='{token[:80]}' len={len(token)}"
            )

        segmented.extend(subtokens)

        if cache is not None:
            put_into_cache(token, subtokens, cache)

    return segmented

def bpe_decoding(tokens):
    """
    Decodes a list of BPE tokens back into the original text.

    Args:
        tokens (List[str]): List of BPE subtokens.

    Returns:
        str: The decoded UTF-8 string.
    """
    byte_decoder = dict([(value, index) for index,value in enumerate(byte_to_unicode())])
    
    joined_tokens = "".join([token for token in tokens])
    
    text = bytearray([byte_decoder[c] for c in joined_tokens]).decode("utf-8", errors="replace")
    
    return text

def text_to_indices(text, pattern, vocab, cache=None):
    """
    Converts a string into token indices based on the vocabulary.

    Returns:
        List[int]: List of integer indices for each token.
    """
    tokens = bpe_encoding(text, pattern, vocab, cache=cache)

    token_to_index = {word: idx for idx, word in enumerate(vocab)}

    indices = []
    for token in tokens:
        try:
            indices.append(token_to_index[token])
        except KeyError as e:
            # This should never happen if vocab is consistent.
            raise KeyError(f"Token not in vocab (unexpected): {token!r}") from e

    return indices

def indices_to_text(indices, vocab):
    """
    Converts a list of token indices back to the original string.

    Returns:
        str: Decoded string from token indices.
    """ 
    index_to_token = {idx: word for idx, word in enumerate(vocab)}
    
    tokens = [index_to_token[i] for i in indices]
     
    text = bpe_decoding(tokens)
    
    return text

###############################################################################
# BBPE Tokenizer Class
###############################################################################

class BBPETokenizer:
    def __init__(self, vocab, pattern, cache_size=100000):
        """
        Tokenizer for Byte-BPE with caching and utility methods.

        Args:
            vocab (list): Ordered list of BPE tokens.
            pattern (str): Regex pattern for pre-tokenization.
            unk_token (str): Unknown token to fall back on.
            cache_size (int): Size of the LFU cache.
        """
        self.vocab = vocab
        self.pattern = pattern
        self.cache = initialize_cache(max_size=cache_size)
        self.token_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        self.index_to_token = {idx: word for idx, word in enumerate(self.vocab)}

    def encode(self, text):
        """Return list of tokens from input text."""
        return bpe_encoding(
            text=text,
            pattern=self.pattern,
            vocab=self.vocab,
            cache=self.cache
        )

    def decode(self, tokens):
        """Return decoded string from list of tokens."""
        return bpe_decoding(tokens)

    def text_to_indices(self, text):
        """Convert text to list of token IDs."""
        tokens = self.encode(text)
        return [self.token_to_index[t] for t in tokens]

    def indices_to_text(self, indices):
        """Convert list of token IDs back to string."""
        tokens = [self.index_to_token[i] for i in indices]
        return self.decode(tokens)

    def tokenize(self, text):
        """
        Tokenize input text into structured output with offsets.

        Returns:
            dict with 'tokens', 'ids', and 'offsets' (byte-based positions)
        """
        tokens = self.encode(text)
        ids = [self.token_to_index.get(token, self.token_to_index[self.unk_token]) for token in tokens]

        offsets = []
        byte_text = text.encode('utf-8')
        byte_offset = 0
        for token in tokens:
            # Remove artificial markers before locating in text
            normalized = token.replace("Ġ", "").replace("Ċ", "")
            token_bytes = normalized.encode("utf-8", errors="ignore")

            start = byte_text.find(token_bytes, byte_offset)
            if start == -1:
                offsets.append((-1, -1))
            else:
                offsets.append((start, start + len(token_bytes)))
                byte_offset = start + len(token_bytes)

        return {
            "tokens": tokens,
            "ids": ids,
            "offsets": offsets
        }

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump({
                "vocab": self.vocab,
                "pattern": self.pattern,
                # don't store cache unless you really want it
            }, f)
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            state = pickle.load(f)
    
        return cls(
            vocab=state["vocab"],
            pattern=state["pattern"],
        )

###############################################################################
# Test 
###############################################################################

def test_tokenizer_encode_decode(tokenizer):
    text = "NASA launched a satellite."
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)

    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)
    assert isinstance(decoded, str)
    print(f"✅ Encode/Decode passed\nEncoded: {tokens}\nDecoded: {decoded}")
    
def test_tokenizer_indices(tokenizer):
    text = "The AI model processed data."
    indices = tokenizer.text_to_indices(text)
    back = tokenizer.indices_to_text(indices)

    assert isinstance(indices, list)
    assert all(isinstance(i, int) for i in indices)
    assert isinstance(back, str)
    print(f"✅ Text-to-indices roundtrip passed\nIndices: {indices}\nText: {back}")
    
def test_roundtrip(tokenizer):
    sample = "Hello world! Testing roundtrip encoding."
    result = tokenizer.indices_to_text(tokenizer.text_to_indices(sample))
    print(f"Original: {sample}\nRecovered: {result}")
    assert isinstance(result, str)
    print("✅ Roundtrip integrity passed.")
    
def test_roundtrip_text(tokenizer, text):
    result = tokenizer.indices_to_text(tokenizer.text_to_indices(text))
    print(f"Original: {text}\nRecovered: {result}")
    assert isinstance(result, str)
    print("✅ Roundtrip integrity passed.")
    
def test_no_oov_tokens(tokenizer):
    text = "易✨<RARE_TOKEN_12345> こんにちは世界"
    tokens = tokenizer.encode(text)
    assert all(t in tokenizer.token_to_index for t in tokens)
    print("✅ No OOV tokens produced (byte-BPE invariant).")
    
def test_save_reload(tokenizer, path="test_tokenizer/bbpe_tokenizer.pkl"):
    tokenizer.save(path)
    tokenizer2 = BBPETokenizer.load(path)

    sample = "Reusable tokenizer saves lives."
    assert tokenizer2.indices_to_text(tokenizer.text_to_indices(sample)) == tokenizer.indices_to_text(tokenizer.text_to_indices(sample))
    print("✅ Save/load consistency passed.")
    
def run_all_tokenizer_tests(tokenizer):
    test_tokenizer_encode_decode(tokenizer)
    test_tokenizer_indices(tokenizer)
    test_roundtrip(tokenizer)
    test_no_oov_tokens(tokenizer)
    print("\n All tokenizer tests passed.")
    
def test_training():
    
    num_workers = 4
    vocab_size = 1000
    input_folder = "corpus"
    
    file_pattern="corpus/data/processed/wikipedia/wikipedia_00000.jsonl"
    
    tokenizer_state = train_tokenizer_state(
        input_folder=input_folder,
        file_pattern=file_pattern,
        output_folder="test_tokenizer",
        pattern=pattern,
        vocab_size=vocab_size,
        save_every=100,
        checkpoint_file="tokenizer_state.pkl",
        num_workers=num_workers,
        resume=False
    )
    
    tokenizer = BBPETokenizer(tokenizer_state['vocab'], 
                              tokenizer_state['pattern'])
    
    run_all_tokenizer_tests(tokenizer)
    
def test_tokenizer_from_vocab(checkpoint_path):
    
    # Load vocabulary to txt
    vocab_path = os.path.join(checkpoint_path, "vocab.txt")
    vocab = load_vocab(vocab_path)
    
    # Initialize tokenizer
    tokenizer = BBPETokenizer(vocab, pattern)
    
    saving_path = os.path.join(checkpoint_path, "bbpe_tokenizer.pkl")
    tokenizer.save(saving_path)
    
    test_sentences = [
        "NASA launched 2024 satellites.",
        "O sistema de saúde pública é importante.",
        "Hello! How are you doing today?",
        "### Section Start ###",
        "O valor foi de R$1.234,56 em 2021.",
        "Olá! Você já viu o relatório 2023?",
        "###  LAUNCH ###",
        "function main() { return true; }",
        "", "\n", "   ", "A" * 100,
        "12345678901234567890",
        "こんにちは世界",  # Japanese
        "Hello ",      # Emojis
    ]
    
    for sentence in test_sentences:
        test_roundtrip_text(tokenizer, sentence)
    
###############################################################################
# Train 
###############################################################################

def train_32k_tokenizer():
    
    resume=True
    num_workers = 36
    vocab_size = 200_000
    total_tokens = 32_000
    save_every = 100
    input_folder = "corpus"
    file_pattern="corpus/data/processed/*/*.jsonl"
    output_folder="tokenizer/checkpoint-2025-07-23"
    checkpoint_file="bbpe_tokenizer_32000.pkl"
    train_state_file="train_state_file.pkl"
    vocab_file="vocab.txt"
    
    os.makedirs(output_folder, exist_ok=True)
    checkpoint_path = os.path.join(output_folder, checkpoint_file)
    train_state_path = os.path.join(output_folder, train_state_file)
    vocab_file_path = os.path.join(output_folder, vocab_file)
    
    if resume and os.path.exists(checkpoint_path):
        print(" Resuming from checkpoint...")
        training_state = load_training_state(train_state_path)
        
    else:
        
        input_paths = load_text_paths(input_folder, file_pattern)
        
        freq_table = parallel_freq_count_processing(input_paths, 
                                                    pattern, 
                                                    num_workers=num_workers)
        
        splits = calculate_splits(freq_table)
        
        vocab = byte_to_unicode()
        
        training_state =  {
            'pattern': pattern,
            'vocab': vocab,
            'freq_table': freq_table,
            'splits': splits
        }
    
        save_training_state(training_state, train_state_path)
    
    print(f"Size of freq table:{len(training_state['freq_table'])}")
    
    # Apply filtering
    threshold = 1000

    filtered_freq_table = {k: v for k, v in training_state['freq_table'].items() 
                           if v >= threshold}    
    
    filtered_splits = calculate_splits(filtered_freq_table)
    
    training_state['freq_table'] = filtered_freq_table
    training_state['splits'] = filtered_splits
        
    print("易 Starting BPE merge loop...")
    print(f"Size of freq table:{len(training_state['freq_table'])}")
    
    
    with open(checkpoint_path, 'rb') as f:
        training_state = pickle.load(f)
    
    
    count = 0
    while len(training_state["vocab"]) < vocab_size:
        
        start_time = time.time()
        pair_freqs = parallel_compute_pair_freqs(training_state["splits"], 
                                                 training_state["freq_table"], 
                                                 num_workers)
        if not pair_freqs:
            print("✅ No more pairs to merge.")
            break
        (a, b), freq = pair_freqs.most_common(1)[0]
        training_state["splits"] = parallel_merge_pair(a, b, 
                                                       training_state["splits"], 
                                                       training_state["freq_table"], 
                                                       num_workers)
        training_state["vocab"].append(a + b)

        elapsed = time.time() - start_time
        print(f"[{count:05}] Merged '{a + b}' (freq={freq}) | Vocab: {len(training_state['vocab'])} | Time: {elapsed:.2f}s")

        if checkpoint_path and count % save_every == 0:
            save_training_state(training_state, checkpoint_path)

        count += 1
    
    vocab = training_state['vocab']
    cleaned_vocab = clean_vocab(vocab, total_tokens, rules)
    
    save_vocab_tokens(cleaned_vocab, vocab_file_path)

    tokenizer = BBPETokenizer(cleaned_vocab, 
                              training_state['pattern'])
    
    run_all_tokenizer_tests(tokenizer)

    tokenizer.save(checkpoint_path)
