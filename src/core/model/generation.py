"""
Created on Wed Dec 17 11:22:49 2025

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import tensorflow as tf

'''
###############################################################################
# Geedy Decode
###############################################################################    
'''

def greedy_decode(
    model,
    prompt,
    tokenizer,
    max_length=128,
    stop_tokens=None,
    pad_token_id=0,
    verbose=True,
):
    """
    Greedy decoding for next-token prediction.

    Supports multiple stop tokens, such as:
        ["<EOS>", "<SPECIAL-0>", "<SPECIAL-1>"]
    """

    if stop_tokens is None:
        stop_tokens = ["<EOS>"]

    stop_token_ids = {
        tokenizer.token_to_index[tok]
        for tok in stop_tokens
        if tok in tokenizer.token_to_index
    }

    # Encode the initial prompt
    input_ids = tokenizer.text_to_indices(prompt)
    input_ids = tf.constant([input_ids], dtype=tf.int32)

    if verbose:
        print(f"Starting with input: {prompt}")

    for _ in range(max_length):

        # Predict Next Token
        logits = model(input_ids)

        # Get the last token prediction
        next_token_id = tf.argmax(logits[:, -1, :], axis=-1)
        next_token_id = tf.cast(next_token_id, tf.int32)

        # Append to Sequence
        token_id_int = int(next_token_id.numpy()[0])

        input_ids = tf.concat(
            [input_ids, tf.expand_dims(next_token_id, axis=-1)],
            axis=-1,
        )

        # Check for end of sequence tokens
        if token_id_int in stop_token_ids:
            if verbose:
                print(f"Stop token generated: {token_id_int}")
            break
    
    # Decode the Generated Text
    text = tokenizer.indices_to_text(input_ids.numpy()[0])
    return text

'''
###############################################################################
# Top-K Decode
###############################################################################    
'''

def top_k_decode(model, 
                 prompt, 
                 tokenizer, 
                 k=5, 
                 max_length=64, 
                 eos_token="<EOS>"):
    """
    Decodes output sequence using top-k sampling strategy.

    Args:
        model (tf.keras.Model): Transformer model.
        prompt (str): The initial text prompt for generation.
        tokenizer (BBPETokenizer): Tokenizer for encoding and decoding.
        k (int): Number of top tokens to sample from.
        max_length (int): Max tokens to generate.
        eos_token (str): End-of-sequence token.

    Returns:
        str: Generated text continuation.
    """
    # ============================
    # Encode the initial prompt
    # ============================
    input_ids = tokenizer.text_to_indices(prompt)
    input_ids = tf.constant([input_ids], dtype=tf.int32)

    print(f"Starting with input: {prompt}")

    for _ in range(max_length):
        logits = model(input_ids)

        # Get top-k probabilities and their indices
        probs, indices = tf.math.top_k(logits[:, -1, :], k=k)
        sampled = tf.random.categorical(tf.math.log(probs), 1)
        next_token_id = tf.gather(indices, sampled, batch_dims=1).numpy()[0][0]

        # Append to sequence
        input_ids = tf.concat([input_ids, [[next_token_id]]], axis=-1)

        # Check for <EOS>
        if next_token_id == tokenizer.token_to_index.get(eos_token, 0):
            print("<EOS> token generated, stopping inference.")
            break

    # Decode the sequence
    return tokenizer.indices_to_text(input_ids.numpy()[0])

def top_p_sample(logits, p=0.9):
    """
    Samples from the top tokens whose cumulative probability exceeds p.
    """
    sorted_logits, sorted_indices = tf.sort(logits, direction='DESCENDING'), tf.argsort(logits, direction='DESCENDING')
    probs = tf.nn.softmax(sorted_logits)
    cumulative_probs = tf.cumsum(probs, axis=-1)

    mask = cumulative_probs <= p
    mask = tf.concat([mask[:, :1], mask[:, 1:] & mask[:, :-1]], axis=-1)

    neg_inf = tf.constant(float('-inf'), dtype=logits.dtype)
    masked_logits = tf.where(mask, sorted_logits, tf.fill(tf.shape(sorted_logits), neg_inf))

    sampled = tf.random.categorical(tf.cast(masked_logits, tf.float32), num_samples=1)
    return tf.gather(sorted_indices, sampled, batch_dims=1).numpy()[0][0]

'''
###############################################################################
# Nucleus Decode
###############################################################################    
'''

def nucleus_decode(model, 
                   prompt, 
                   tokenizer, 
                   p=0.9, 
                   max_length=64, 
                   eos_token="<EOS>"):
    """
    Decodes using nucleus (top-p) sampling strategy.

    Args:
        model (tf.keras.Model): Transformer model.
        prompt (str): The initial text prompt for generation.
        tokenizer (BBPETokenizer): Tokenizer for encoding and decoding.
        p (float): Cumulative probability threshold (e.g. 0.9).
        max_length (int): Max length to decode.
        eos_token (str): End-of-sequence token.

    Returns:
        str: Generated text continuation.
    """
    # ============================
    # Encode the initial prompt
    # ============================
    input_ids = tokenizer.text_to_indices(prompt)
    input_ids = tf.constant([input_ids], dtype=tf.int32)

    print(f"Starting with input: {prompt}")

    for _ in range(max_length):
        logits = model(input_ids)
        next_token_id = top_p_sample(logits[:, -1, :], p)

        # Append to sequence
        input_ids = tf.concat([input_ids, [[next_token_id]]], axis=-1)

        # Check for <EOS>
        if next_token_id == tokenizer.token_to_index.get(eos_token, 0):
            print("<EOS> token generated, stopping inference.")
            break

    # Decode the sequence
    return tokenizer.indices_to_text(input_ids.numpy()[0])

'''
###############################################################################
# Beam Search Decode
###############################################################################    
'''

def beam_search_decode(model, 
                       prompt, 
                       tokenizer, 
                       beam_width=3, 
                       max_length=64, 
                       eos_token="<EOS>"):
    """
    Performs beam search decoding (keeps top-k sequences at each step).

    Args:
        model (tf.keras.Model): Transformer model.
        prompt (str): The initial text prompt for generation.
        tokenizer (BBPETokenizer): Tokenizer for encoding and decoding.
        beam_width (int): Number of beams to keep.
        max_length (int): Max tokens to generate.

    Returns:
        str: Generated text continuation.
    """
    # ============================
    # Encode the initial prompt
    # ============================
    input_ids = tokenizer.text_to_indices(prompt)
    eos_id = tokenizer.token_to_index.get(eos_token, 0)

    sequences = [([tid for tid in input_ids], 0.0)]  # start with full prompt

    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            input_tensor = tf.constant([seq], dtype=tf.int32)
            logits = model(input_tensor)
            log_probs = tf.nn.log_softmax(logits[:, -1, :])[0]
            top_k_probs, top_k_ids = tf.math.top_k(log_probs, k=beam_width)

            for i in range(beam_width):
                token_id = int(top_k_ids[i])
                new_seq = seq + [token_id]
                new_score = score + float(top_k_probs[i])
                all_candidates.append((new_seq, new_score))

        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        # Early stopping if all sequences have <EOS>
        if all(seq[-1] == eos_id for seq, _ in sequences):
            print("All sequences ended with <EOS>, stopping inference.")
            break

    best_seq = sequences[0][0]
    if eos_id in best_seq:
        best_seq = best_seq[:best_seq.index(eos_id)]

    # Decode the sequence
    return tokenizer.indices_to_text(best_seq)