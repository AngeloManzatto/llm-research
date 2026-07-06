"""
Created on Wed Dec 17 11:22:49 2025

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################

import tensorflow as tf

###############################################################################
# Greedy Decode
###############################################################################

def greedy_decode(
    model,
    tokenizer,
    input_ids: list[int],
    stop_token_ids: set[int] | None = None,
    max_length: int = 128,
    verbose: bool = False,
) -> str:
    """
    Greedy decoding from a pre-built token ID sequence.

    At each step the token with the highest logit is selected (no randomness).
    This is deterministic: the same input always produces the same output,
    which makes it the correct choice for benchmark evaluation.

    Parameters
    ----------
    model : tf.keras.Model
        The language model. Expects input shape [batch, seq_len] and returns
        logits of shape [batch, seq_len, vocab_size].
    tokenizer : BBPETokenizer
        Used only for decoding the generated IDs back to text.
        No encoding happens here — input_ids must be pre-built by the caller.
    input_ids : list[int]
        Flat list of token IDs representing the full prompt context, including
        all structural tokens (role markers, prior turn content, generation
        trigger). The caller (TextGenerator.messages_to_ids) is responsible
        for constructing this correctly.
    stop_token_ids : set[int] | None
        Generation halts immediately when any of these token IDs is produced.
        The stop token IS included in the returned output so that downstream
        metrics (e.g. ExpectedStopTokenMetric) can verify it was emitted.
        If None, generation continues until max_length is reached.
    max_length : int
        Maximum number of tokens to generate. Acts as a safety cap if the
        model never emits a stop token.
    verbose : bool
        If True, prints which stop token ID triggered halting.

    Returns
    -------
    str
        Decoded string of the generated portion only (the prompt context is
        excluded). Includes the stop token if one was emitted.
    """
    # Normalise: None means no early stopping — generate to max_length
    stop_ids = set(stop_token_ids) if stop_token_ids is not None else set()

    # Record where the prompt ends so we can slice it off the output later
    prompt_len = len(input_ids)

    # Wrap in a batch dimension: [1, seq_len]
    ids = tf.constant([input_ids], dtype=tf.int32)

    for _ in range(max_length):
        # Forward pass → logits: [1, seq_len, vocab_size]
        logits = model(ids)

        # Take the argmax over the last token position to get the next token
        next_id = int(tf.argmax(logits[:, -1, :], axis=-1).numpy()[0])

        # Append the new token to the running sequence
        ids = tf.concat(
            [ids, tf.constant([[next_id]], dtype=tf.int32)],
            axis=-1,
        )

        # Stop as soon as any designated stop token is produced
        if stop_ids and next_id in stop_ids:
            if verbose:
                print(f"Stop token id={next_id} generated.")
            break

    # Slice off the prompt prefix; keep only what the model generated
    generated_ids = ids.numpy()[0][prompt_len:].tolist()

    # Decode generated IDs back to text (special token IDs must round-trip
    # correctly through tokenizer.indices_to_text for metrics to work)
    return tokenizer.indices_to_text(generated_ids)


###############################################################################
# Top-K Decode
###############################################################################

def top_k_decode(
    model,
    tokenizer,
    input_ids: list[int],
    stop_token_ids: set[int] | None = None,
    k: int = 5,
    max_length: int = 64,
) -> str:
    """
    Top-k sampling from a pre-built token ID sequence.

    At each step only the k highest-probability tokens are considered;
    one is sampled from that restricted distribution. Introduces controlled
    randomness — useful for creative generation but non-deterministic.

    Parameters
    ----------
    model : tf.keras.Model
        The language model.
    tokenizer : BBPETokenizer
        Used only for decoding.
    input_ids : list[int]
        Pre-built prompt token IDs (see greedy_decode for full description).
    stop_token_ids : set[int] | None
        Generation halts when any of these token IDs is produced.
        If None, generation continues until max_length is reached.
    k : int
        Number of top tokens to sample from at each step.
        Lower k = more focused; higher k = more diverse.
    max_length : int
        Maximum tokens to generate.

    Returns
    -------
    str
        Decoded generated portion including the stop token if emitted.
    """
    stop_ids  = set(stop_token_ids) if stop_token_ids is not None else set()
    prompt_len = len(input_ids)
    ids = tf.constant([input_ids], dtype=tf.int32)

    for _ in range(max_length):
        # Forward pass → logits: [1, seq_len, vocab_size]
        logits = model(ids)

        # Restrict to the top-k logits and convert to probabilities
        top_probs, top_indices = tf.math.top_k(logits[:, -1, :], k=k)

        # Sample one token from the top-k distribution
        sampled_pos = tf.random.categorical(tf.math.log(top_probs), num_samples=1)
        next_id = int(tf.gather(top_indices, sampled_pos, batch_dims=1).numpy()[0][0])

        # Append the sampled token to the running sequence
        ids = tf.concat([ids, tf.constant([[next_id]], dtype=tf.int32)], axis=-1)

        if stop_ids and next_id in stop_ids:
            break

    generated_ids = ids.numpy()[0][prompt_len:].tolist()
    return tokenizer.indices_to_text(generated_ids)


###############################################################################
# Nucleus (Top-P) Decode
###############################################################################

def _top_p_sample(logits, p: float) -> int:
    """
    Sample one token using nucleus (top-p) filtering.

    Tokens are sorted by descending probability. The smallest set of tokens
    whose cumulative probability meets or exceeds p forms the nucleus.
    The token that pushes cumulative probability past p is included so the
    nucleus is never empty. One token is then sampled from this set.

    Parameters
    ----------
    logits : tf.Tensor
        Shape [1, 1, vocab_size] — raw logits for the next token position.
    p : float
        Cumulative probability threshold (e.g. 0.9).

    Returns
    -------
    int
        Sampled token ID in the original (unsorted) vocabulary space.
    """
    # Sort tokens by descending probability
    sorted_logits  = tf.sort(logits, direction="DESCENDING")
    sorted_indices = tf.argsort(logits, direction="DESCENDING")

    # Convert sorted logits to probabilities
    probs      = tf.nn.softmax(sorted_logits)
    cumulative = tf.cumsum(probs, axis=-1)

    # Shift cumulative probs right by one so the token that pushes us past p
    # is included in the nucleus (without the shift it would be excluded)
    shifted = tf.concat([tf.zeros_like(cumulative[:, :, :1]), cumulative[:, :, :-1]], axis=-1)

    # Build nucleus mask: True where the token should be kept
    nucleus_mask = shifted < p

    # Zero out logits outside the nucleus by replacing them with -inf
    neg_inf       = tf.fill(tf.shape(sorted_logits), float("-inf"))
    masked_logits = tf.where(nucleus_mask, sorted_logits, neg_inf)

    # Sample from the nucleus
    # Squeeze the sequence dimension before categorical sampling
    sampled_pos = tf.random.categorical(
        tf.cast(tf.squeeze(masked_logits, axis=1), tf.float32), num_samples=1
    )

    # Map back to the original vocabulary index
    return int(tf.gather(tf.squeeze(sorted_indices, axis=1), sampled_pos, batch_dims=1).numpy()[0][0])


def nucleus_decode(
    model,
    tokenizer,
    input_ids: list[int],
    stop_token_ids: set[int] | None = None,
    p: float = 0.9,
    max_length: int = 64,
) -> str:
    """
    Nucleus (top-p) sampling from a pre-built token ID sequence.

    At each step the smallest set of tokens whose cumulative probability
    meets or exceeds p is sampled from. Adapts the candidate set size to the
    model's confidence — focused when confident, broad when uncertain.

    Parameters
    ----------
    model : tf.keras.Model
        The language model.
    tokenizer : BBPETokenizer
        Used only for decoding.
    input_ids : list[int]
        Pre-built prompt token IDs (see greedy_decode for full description).
    stop_token_ids : set[int] | None
        Generation halts when any of these token IDs is produced.
        If None, generation continues until max_length is reached.
    p : float
        Nucleus probability threshold. 0.9 is a common default.
        Lower p = more conservative; higher p = more diverse.
    max_length : int
        Maximum tokens to generate.

    Returns
    -------
    str
        Decoded generated portion including the stop token if emitted.
    """
    stop_ids   = set(stop_token_ids) if stop_token_ids is not None else set()
    prompt_len = len(input_ids)
    ids = tf.constant([input_ids], dtype=tf.int32)

    for _ in range(max_length):
        # Forward pass → logits: [1, seq_len, vocab_size]
        logits = model(ids)

        # Sample from the nucleus at the last token position
        next_id = _top_p_sample(logits[:, -1:, :], p)

        # Append sampled token to the running sequence
        ids = tf.concat([ids, tf.constant([[next_id]], dtype=tf.int32)], axis=-1)

        if stop_ids and next_id in stop_ids:
            break

    generated_ids = ids.numpy()[0][prompt_len:].tolist()
    return tokenizer.indices_to_text(generated_ids)


###############################################################################
# Beam Search Decode
###############################################################################

def beam_search_decode(
    model,
    tokenizer,
    input_ids: list[int],
    stop_token_ids: set[int] | None = None,
    beam_width: int = 3,
    max_length: int = 64,
) -> str:
    """
    Beam search decoding from a pre-built token ID sequence.

    Maintains beam_width candidate sequences at each step, expanding each
    by the top beam_width next tokens and keeping only the highest-scoring
    beam_width candidates overall. Returns the single best sequence.

    More computationally expensive than greedy or sampling methods —
    runs beam_width forward passes per generation step — but tends to
    produce more globally coherent sequences.

    Parameters
    ----------
    model : tf.keras.Model
        The language model.
    tokenizer : BBPETokenizer
        Used only for decoding.
    input_ids : list[int]
        Pre-built prompt token IDs (see greedy_decode for full description).
    stop_token_ids : set[int] | None
        Generation halts when ALL active beams end with any of these token IDs.
        Stop tokens are stripped from the final output since beam scores are
        compared across beams and including the stop token would be inconsistent.
        If None, generation continues until max_length is reached.
    beam_width : int
        Number of candidate sequences to maintain at each step.
    max_length : int
        Maximum tokens to generate per beam.

    Returns
    -------
    str
        Decoded generated portion of the best-scoring beam, without the
        stop token.
    """
    stop_ids   = set(stop_token_ids) if stop_token_ids is not None else set()
    prompt_len = len(input_ids)

    # Each beam is (token_id_sequence, cumulative_log_prob)
    beams = [(list(input_ids), 0.0)]

    for _ in range(max_length):
        candidates = []

        for seq, score in beams:
            # Forward pass for this beam → logits: [1, seq_len, vocab_size]
            logits    = model(tf.constant([seq], dtype=tf.int32))
            log_probs = tf.nn.log_softmax(logits[:, -1, :])[0]

            # Expand beam by the top beam_width tokens
            top_log_probs, top_ids = tf.math.top_k(log_probs, k=beam_width)

            for i in range(beam_width):
                candidates.append((
                    seq + [int(top_ids[i])],
                    score + float(top_log_probs[i]),
                ))

        # Keep only the best beam_width candidates
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        # Early stop if every beam has produced a stop token (or no stop tokens defined)
        if stop_ids and all(seq[-1] in stop_ids for seq, _ in beams):
            break

    # Take the highest-scoring beam and strip the prompt prefix
    best_seq      = beams[0][0]
    generated_ids = best_seq[prompt_len:]

    # Strip the trailing stop token if present
    if generated_ids and stop_ids and generated_ids[-1] in stop_ids:
        generated_ids = generated_ids[:-1]

    return tokenizer.indices_to_text(generated_ids)