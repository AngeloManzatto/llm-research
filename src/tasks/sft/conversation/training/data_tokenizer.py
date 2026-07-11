"""
Created on Sat Jul 11 09:31:42 2026

@author: Angelo Manzatto
"""
###############################################################################
# Libraries
###############################################################################

import numpy as np
from src.tasks.sft.conversation.core.special_tokens import TOKEN_BY_NAME
 
###############################################################################
# Resolve Token IDs
###############################################################################

def resolve_token_ids(tokenizer) -> dict[str, int]:
    """
    Resolve all special token IDs needed for SFT from the tokenizer vocab.
 
    Returns
    -------
    dict with keys: USER_ID, ASST_ID, SYS_ID, EOS_ID, PAD_ID, IGNORE_ID
    """
    return {
        "USER_ID":  tokenizer.token_to_index[TOKEN_BY_NAME["USER_TURN"].token],
        "ASST_ID":  tokenizer.token_to_index[TOKEN_BY_NAME["ASSISTANT_TURN"].token],
        "SYS_ID":   tokenizer.token_to_index[TOKEN_BY_NAME["SYSTEM_TURN"].token],
        "EOS_ID":   tokenizer.token_to_index[TOKEN_BY_NAME["END_OF_TURN"].token],
        "PAD_ID":   tokenizer.token_to_index["<PAD>"],
        "IGNORE_ID": -100,
    }
 
###############################################################################
# Message to Tokens
###############################################################################

def messages_to_tokens(
    messages: list[dict],
    seq_len: int,
    tokenizer,
    token_ids: dict[str, int],
    pad_side: str = "left",
    truncate_from: str = "left",
) -> dict:
    """
    Convert a messages list into LM training tensors.
 
    Token layout for each turn:
        user      : [USER_ID] + text_ids + [EOS_ID]
        assistant : [ASST_ID] + text_ids + [EOS_ID]   ← loss computed here
        system    : [SYS_ID]  + text_ids + [EOS_ID]   (Stage 3+, masked)
 
    Loss masking:
        IGNORE_ID on all role-marker tokens and on all user/system turn content.
        Loss computed only on assistant content tokens and their closing EOS.
 
    Parameters
    ----------
    messages : list[dict]
        Each dict has "role" and "content". Must start and end with "user".
    seq_len : int
        Fixed output length. Sequences are truncated and/or padded to this.
    tokenizer : BBPETokenizer
        Used to encode content strings.
    token_ids : dict
        Output of resolve_token_ids().
    pad_side : "left" | "right"
    truncate_from : "left" | "right"
 
    Returns
    -------
    dict with: input_ids, labels, attention_mask, debug
    """
    
    USER_ID   = token_ids["USER_ID"]
    ASST_ID   = token_ids["ASST_ID"]
    SYS_ID    = token_ids["SYS_ID"]
    EOS_ID    = token_ids["EOS_ID"]
    PAD_ID    = token_ids["PAD_ID"]
    IGNORE_ID = token_ids["IGNORE_ID"]
 
    ROLE_IDS = {"user": USER_ID, "assistant": ASST_ID, "system": SYS_ID}
 
    full      = []
    trainable = []
 
    for msg in messages:
        role     = msg["role"]
        content  = msg["content"].strip()
        role_id  = ROLE_IDS[role]
        text_ids = tokenizer.text_to_indices(content)
        is_asst  = (role == "assistant")
 
        full.append(role_id);        trainable.append(False)
        for tid in text_ids:
            full.append(tid);        trainable.append(is_asst)
        full.append(EOS_ID);         trainable.append(is_asst)
 
    n          = len(full)
    labels_raw = full[1:] + [IGNORE_ID]
    labels     = [
        labels_raw[i] if (i + 1 < n and trainable[i + 1]) else IGNORE_ID
        for i in range(n)
    ]
 
    full   = np.array(full,   dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)
 
    if len(full) > seq_len:
        if truncate_from == "left":
            full = full[-seq_len:]; labels = labels[-seq_len:]
        else:
            full = full[:seq_len];  labels = labels[:seq_len]
 
    pad_len = seq_len - len(full)
    if pad_len > 0:
        pad_ids  = np.full(pad_len, PAD_ID,    dtype=np.int32)
        pad_labs = np.full(pad_len, IGNORE_ID, dtype=np.int32)
        if pad_side == "left":
            full   = np.concatenate([pad_ids,  full],   axis=0)
            labels = np.concatenate([pad_labs, labels], axis=0)
        else:
            full   = np.concatenate([full,   pad_ids],  axis=0)
            labels = np.concatenate([labels, pad_labs], axis=0)
 
    attention_mask = (full != PAD_ID).astype(np.int32)
    n_trainable    = int((labels != IGNORE_ID).sum())
    n_real         = int(attention_mask.sum())
 
    return {
        "input_ids":      full,
        "labels":         labels,
        "attention_mask": attention_mask,
        "debug": {
            "n_real_tokens":      n_real,
            "n_trainable_tokens": n_trainable,
            "n_pad_tokens":       seq_len - n_real,
            "trainable_ratio":    round(n_trainable / max(n_real, 1), 3),
        },
    }
 