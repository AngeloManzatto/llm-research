"""
Created on Sat Jan  3 17:37:47 2026

@author: Angelo Antonio Manzatto
"""

###############################################################################
# Libraries
###############################################################################
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import csv
import datetime
from pathlib import Path

import tensorflow as tf

from src.core.model.config import TransformerConfig
from src.core.model.transformer import Transformer 

from src.core.model.transformer import (
    FeedForwardBlock,
    RMSNorm,
    GroupedQueryAttention,
    causal_mask,
    attn_block,
    apply_rope_complex
    )


from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")  # recommended policy name 

###############################################################################
# GPU Setup
###############################################################################
def setup_gpus(memory_growth: bool = True) -> int:
    gpus = tf.config.list_physical_devices("GPU")
    if memory_growth:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception as e:
                print(f"[WARN] Could not set memory growth for {g}: {e}")
    strategy = tf.distribute.MirroredStrategy()
    print("-" * 100)
    print(f"Number of devices (GPUs): {strategy.num_replicas_in_sync}")
    return strategy.num_replicas_in_sync

###############################################################################
# Hyper parameters
###############################################################################

# Batch size
batch_size = 1

# Device to run 
device = "GPU:0"
warmup = 3

# Model
vocab_size  = 32000
d_model     = 768
n_layers    = 8
n_heads     = 8
n_kv_heads  = 8
ffn_dim_multiplier = 1.5
multiple_of = 256
seq_len     = 1024
norm_eps    = 1e-5
use_cache   = False

# Capacity search behavior
warmup_steps   = 2   # warmup steps per batch size
measure_steps  = 1   # measured steps per batch size (keep 1 for memory)
max_cap        = 2048

# Optimizer selection
optimizer_name = "adamw"
learning_rate  = 1e-4
mp_policy      = mixed_precision.global_policy()

# Test types
dtypes = {
    "float32": tf.float32,
    "float16": tf.float16,
    "bfloat16": tf.bfloat16,
}


###############################################################################
# Build Model
###############################################################################

tranformer_cfg = TransformerConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_dim_multiplier=ffn_dim_multiplier,
            multiple_of=multiple_of,
            seq_len=seq_len,
            norm_eps=norm_eps,
            use_cache=use_cache,
        )

model = Transformer(tranformer_cfg)
_ = model(tf.zeros((1, seq_len), tf.int32), start_pos=0, training=False)

print(100*"-")
print("Model\n")
model.summary()

###############################################################################
# Build Optimizer
###############################################################################

def make_optimizer(name, lr=1e-4):
    name = name.lower()
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr)
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == "adamw":
        # TF 2.20: AdamW exists in keras.optimizers
        return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.01)
    raise ValueError(f"Unknown optimizer: {name}")

optimizer = make_optimizer(optimizer_name, learning_rate)
optimizer.build(model.trainable_variables)

###############################################################################
# Memory Stats Helpers
###############################################################################
def reset_peak(device):
    tf.config.experimental.reset_memory_stats(device)

def mem_info(device):
    info = tf.config.experimental.get_memory_info(device)
    return info["current"], info["peak"]

def mb(x):
    return x / 1024 / 1024

def summarize_mem(device, peak0):
    cur1, peak1 = mem_info(device)
    return {
        "current_mb": mb(cur1),
        "peak_mb": mb(peak1),
        "delta_peak_mb": mb(peak1 - peak0),
    }

###############################################################################
# Force GPU execution / sync (memory-only benchmark)
###############################################################################
def force_compute(t):
    # Touch tensor and sync
    _ = tf.reduce_sum(tf.cast(t, tf.float32)).numpy()

def force_compute_any(out):
    # Handles tensor / (tensor, ...) / dict outputs
    if isinstance(out, (tuple, list)):
        out = out[0]
    elif isinstance(out, dict):
        out = next(iter(out.values()))
    force_compute(out)

###############################################################################
# Generic memory probe
###############################################################################
def profile_callable_memory(
    name,
    *,
    build_inputs_fn,
    forward_fn,
    watch_fn=None,
    params_fn=None,
    jit_compile=False,
):

    res = {"name": name, "runs": []}

    for dtype_name, dt in dtypes.items():
        inputs = build_inputs_fn(dt)

        @tf.function(jit_compile=jit_compile)
        def fwd():
            return forward_fn(inputs)

        # -------- forward warmup --------
        for _ in range(warmup):
            out = fwd()
            force_compute_any(out)

        reset_peak(device)
        _, peak0 = mem_info(device)

        out = fwd()
        force_compute_any(out)

        fwd_stats = summarize_mem(device, peak0)

        # -------- backward (optional) --------
        bwd_stats = None
        if watch_fn is not None:
            watch_tensors = watch_fn(inputs)

            def one_bwd():
                with tf.GradientTape() as tape:
                    tape.watch(watch_tensors)
                    out2 = fwd()
                    if isinstance(out2, (tuple, list)):
                        out2 = out2[0]
                    loss = tf.reduce_mean(tf.cast(out2, tf.float32))
                grads = tape.gradient(loss, watch_tensors)

                _ = loss.numpy()
                if grads is not None:
                    if isinstance(grads, (tuple, list)):
                        for g in grads:
                            if g is not None:
                                force_compute(g)
                    else:
                        if grads is not None:
                            force_compute(grads)

                return loss

            for _ in range(warmup):
                _ = one_bwd()

            reset_peak(device)
            _, peak0 = mem_info(device)

            loss = one_bwd()
            bwd_stats = summarize_mem(device, peak0)
            bwd_stats["loss"] = float(loss)

        params = params_fn() if params_fn is not None else None

        res["runs"].append({
            "dtype": dt.name,
            "params": params,
            "forward": fwd_stats,
            "fwd_bwd": bwd_stats,
        })

    return res

###############################################################################
# Pretty printer
###############################################################################
def print_profile_results(res):
    print("\n" + "=" * 100)
    print(res["name"])
    for r in res["runs"]:
        print("-" * 100)
        print(f"dtype: {r['dtype']}")
        f = r["forward"]
        print(f"FORWARD:   peak={f['peak_mb']:.1f} MB | Δpeak={f['delta_peak_mb']:.1f} MB | cur={f['current_mb']:.1f} MB")
        if r["fwd_bwd"] is not None:
            b = r["fwd_bwd"]
            print(f"FWD+BWD:   peak={b['peak_mb']:.1f} MB | Δpeak={b['delta_peak_mb']:.1f} MB | cur={b['current_mb']:.1f} MB | loss={b['loss']:.6f}")

###############################################################################
# RMSNorm Layer
###############################################################################
def run_probe_rmsnorm():
    layer = RMSNorm(d_model, eps=norm_eps)

    def build_inputs(dt):
        x = tf.random.normal((batch_size, seq_len, d_model), dtype=dt)
        return {"x": x}

    def forward(inputs):
        return layer(inputs["x"])

    def watch(inputs):
        return [inputs["x"]]

    return profile_callable_memory(
        "RMSNorm",
        build_inputs_fn=build_inputs,
        forward_fn=forward,
        watch_fn=watch,
        params_fn=lambda: layer.count_params(),
    )

###############################################################################
# FeedForwardBlock Layer
###############################################################################
def run_probe_ffn():
    layer = FeedForwardBlock(d_model, ffn_dim_multiplier=ffn_dim_multiplier, multiple_of=multiple_of)

    def build_inputs(dt):
        x = tf.random.normal((batch_size, seq_len, d_model), dtype=dt)
        return {"x": x}

    def forward(inputs):
        return layer(inputs["x"])

    def watch(inputs):
        return [inputs["x"]]

    return profile_callable_memory(
        "FeedForwardBlock",
        build_inputs_fn=build_inputs,
        forward_fn=forward,
        watch_fn=watch,
        params_fn=lambda: layer.count_params(),
    )

###############################################################################
# ROPE
###############################################################################
def run_probe_rope():
    # RoPE operates on [B,T,H,Dh], freq_complex is complex64
    Dh = d_model // n_heads

    def build_inputs(dt):
        x = tf.random.normal((batch_size, seq_len, n_heads, Dh), dtype=dt)
        freq = model.freq_complex[:seq_len]  # complex64
        return {"x": x, "freq": freq}

    def forward(inputs):
    
        y = apply_rope_complex(inputs["x"], inputs["freq"], start_pos=0)
        return y

    def watch(inputs):
        return [inputs["x"]]

    return profile_callable_memory(
        "apply_rope_complex",
        build_inputs_fn=build_inputs,
        forward_fn=forward,
        watch_fn=watch,
        params_fn=lambda: 0,
    )

###############################################################################
# Attention block
###############################################################################
def run_probe_attn_block():
    Dh = d_model // n_heads

    def build_inputs(dt):
        q = tf.random.normal((batch_size, seq_len, n_heads, Dh), dtype=dt)
        k = tf.random.normal((batch_size, seq_len, n_heads, Dh), dtype=dt)
        v = tf.random.normal((batch_size, seq_len, n_heads, Dh), dtype=dt)
        mask = causal_mask(seq_len, tf.float32)
        return {"q": q, "k": k, "v": v, "Dh": Dh, "mask": mask}

    def forward(inputs):
        # your original attn_block forces fp32 internally :contentReference[oaicite:5]{index=5}
        return attn_block(inputs["q"], inputs["k"], inputs["v"], inputs["Dh"], inputs["mask"])

    def watch(inputs):
        return [inputs["q"], inputs["k"], inputs["v"]]

    return profile_callable_memory(
        "attn_block",
        build_inputs_fn=build_inputs,
        forward_fn=forward,
        watch_fn=watch,
        params_fn=lambda: 0,
    )

###############################################################################
# GroupedQueryAttention layer
###############################################################################
def run_probe_gqa():
    layer = GroupedQueryAttention(d_model=d_model, n_heads=n_heads, n_kv_heads=None, use_cache=False)

    def build_inputs(dt):
        x = tf.random.normal((batch_size, seq_len, d_model), dtype=dt)
        freq = model.freq_complex[:seq_len]   # complex64
        mask = causal_mask(seq_len, tf.float32)
        return {"x": x, "freq": freq, "mask": mask}

    def forward(inputs):
        return layer(inputs["x"], freq_complex=inputs["freq"], start_pos=0, attn_mask=inputs["mask"], training=True)

    def watch(inputs):
        return [inputs["x"]]

    return profile_callable_memory(
        "GroupedQueryAttention",
        build_inputs_fn=build_inputs,
        forward_fn=forward,
        watch_fn=watch,
        params_fn=lambda: layer.count_params(),
    )

###############################################################################
# Transformer
###############################################################################
def run_probe_transformer():
    def build_inputs(dt):
        # ids must be int32, not dt
        ids = tf.random.uniform((batch_size, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32)
        return {"ids": ids}

    def forward(inputs):
        return model(inputs["ids"], start_pos=0, training=True)

    def watch(inputs):
        # watch embeddings input doesn't exist; we watch nothing and instead watch model.trainable_variables
        # But for simplicity and consistency: use GradientTape on trainable vars below.
        return []

    # Special case: backward w.r.t. trainable vars (not input ids)
    def profile_model_trainable():
        res = {"name": "Transformer (full model)", "runs": []}
        for _, dt in dtypes.items():
            inputs = build_inputs(dt)

            @tf.function(jit_compile=False)
            def fwd():
                return forward(inputs)

            # forward warmup
            for _ in range(warmup):
                out = fwd()
                force_compute_any(out)

            reset_peak(device)
            _, peak0 = mem_info(device)
            out = fwd()
            force_compute_any(out)
            fwd_stats = summarize_mem(device, peak0)

            # backward warmup
            for _ in range(warmup):
                with tf.GradientTape() as tape:
                    out2 = fwd()
                    loss = tf.reduce_mean(tf.cast(out2, tf.float32))
                grads = tape.gradient(loss, model.trainable_variables)
                _ = loss.numpy()
                # touch one grad to force exec if exists
                for g in grads:
                    if g is not None:
                        force_compute(g)
                        break

            reset_peak(device)
            _, peak0 = mem_info(device)

            with tf.GradientTape() as tape:
                out2 = fwd()
                loss = tf.reduce_mean(tf.cast(out2, tf.float32))
            grads = tape.gradient(loss, model.trainable_variables)
            _ = loss.numpy()
            for g in grads:
                if g is not None:
                    force_compute(g)
                    break

            bwd_stats = summarize_mem(device, peak0)
            bwd_stats["loss"] = float(loss)

            res["runs"].append({
                "dtype": dt.name,
                "params": model.count_params(),
                "forward": fwd_stats,
                "fwd_bwd": bwd_stats,
            })
        return res

    return profile_model_trainable()

###############################################################################
# Loss function (next-token)
###############################################################################
def compute_loss(ids: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    # logits: [B,T,V]; ids: [B,T]
    y_true = ids[:, 1:]
    y_pred = logits[:, :-1, :]
    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    )
    return tf.cast(loss, tf.float32)

###############################################################################
# Train step (compiled)
###############################################################################
@tf.function(jit_compile=False)
def train_step(model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, ids: tf.Tensor) -> tf.Tensor:
    with tf.GradientTape() as tape:
        logits = model(ids, start_pos=0, training=True)
        loss = compute_loss(ids, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

###############################################################################
# OOM detection
###############################################################################
def is_oom_error(e: Exception) -> bool:
    msg = str(e).lower()
    return (
        isinstance(e, tf.errors.ResourceExhaustedError)
        or "oom" in msg
        or "resourceexhausted" in msg
        or "ran out of memory" in msg
    )

###############################################################################
# Try a batch size once; return (ok, stats)
###############################################################################
def try_batch_size(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    batch_size: int,
):
    ids = tf.random.uniform((batch_size, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32)

    try:
        # warmup
        for _ in range(warmup_steps):
            loss = train_step(model, optimizer, ids)
            _ = float(loss.numpy())

        # measure memory
        reset_peak(device)
        _, peak0 = mem_info(device)

        loss = None
        for _ in range(measure_steps):
            loss = train_step(model, optimizer, ids)
            _ = float(loss.numpy())

        stats = summarize_mem(device, peak0)
        stats["loss"] = float(loss) if loss is not None else float("nan")
        return True, stats

    except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
        if is_oom_error(e):
            return False, {"error": "OOM", "detail": str(e)}
        raise

###############################################################################
# Find max batch (doubling then binary search)
###############################################################################

def find_max_batch_size():
    attempts = []

    def log_attempt(status: str, batch: int, stats: dict | None):
        row = {
            "status": status,          # "OK" or "OOM"
            "batch": batch,
            "delta_peak_mb": None,
            "peak_mb": None,
            "current_mb": None,
            "loss": None,
        }
        if stats:
            row.update({
                "delta_peak_mb": stats.get("delta_peak_mb"),
                "peak_mb": stats.get("peak_mb"),
                "current_mb": stats.get("current_mb"),
                "loss": stats.get("loss"),
            })
        attempts.append(row)

    print("\n" + "=" * 100)
    print(f"Finding max batch size | optimizer={optimizer_name} | lr={learning_rate} | policy={mp_policy}")
    print("=" * 100)

    b = 1
    last_ok = None
    last_ok_stats = None

    # 1) Exponential growth
    while b <= max_cap:
        ok, stats = try_batch_size(model, optimizer, b)
        if ok:
            print(f"[OK ] batch={b:4d}  Δpeak={stats['delta_peak_mb']:.1f} MB  peak={stats['peak_mb']:.1f} MB  loss={stats['loss']:.4f}")
            log_attempt("OK", b, stats)
            last_ok = b
            last_ok_stats = stats
            b *= 2
        else:
            print(f"[OOM] batch={b:4d}  (stopping growth)")
            log_attempt("OOM", b, None)
            break

    if last_ok is None:
        print("Even batch_size=1 OOM. Reduce seq_len / d_model or enable mixed precision.")
        return {"max_batch": None, "stats": None, "attempts": attempts}

    if b > max_cap:
        print(f"Reached max_cap={max_cap} without OOM. Max batch >= {last_ok}.")
        return {"max_batch": last_ok, "stats": last_ok_stats, "attempts": attempts}

    lo = last_ok
    hi = b  # first failing

    # 2) Binary search
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, stats = try_batch_size(model, optimizer, mid)
        if ok:
            print(f"[OK ] batch={mid:4d}  Δpeak={stats['delta_peak_mb']:.1f} MB  peak={stats['peak_mb']:.1f} MB")
            log_attempt("OK", mid, stats)
            lo = mid
            last_ok_stats = stats
        else:
            print(f"[OOM] batch={mid:4d}")
            log_attempt("OOM", mid, None)
            hi = mid

    print("\n" + "-" * 100)
    print(f"MAX BATCH SIZE (fwd+bwd+opt): {lo}")
    if last_ok_stats is not None:
        print(f"At batch={lo}: Δpeak={last_ok_stats['delta_peak_mb']:.1f} MB | peak={last_ok_stats['peak_mb']:.1f} MB | cur={last_ok_stats['current_mb']:.1f} MB")
    print("-" * 100)

    return {"max_batch": lo, "stats": last_ok_stats, "attempts": attempts}

###############################################################################
# Save run profile
###############################################################################
def make_run_dir(root="logs/memory"):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(root) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def save_csv(rows, path):
    if not rows:
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
        
def flatten_probe_result(res):
    rows = []
    name = res.get("name", "")
    for r in res.get("runs", []):
        fwd = r.get("forward") or {}
        fbw = r.get("fwd_bwd") or {}
        rows.append({
            "name": name,
            "dtype": r.get("dtype"),
            "params": r.get("params"),
            "fwd_current_mb": fwd.get("current_mb"),
            "fwd_peak_mb": fwd.get("peak_mb"),
            "fwd_delta_peak_mb": fwd.get("delta_peak_mb"),
            "fwd_bwd_current_mb": fbw.get("current_mb"),
            "fwd_bwd_peak_mb": fbw.get("peak_mb"),
            "fwd_bwd_delta_peak_mb": fbw.get("delta_peak_mb"),
            "loss": fbw.get("loss"),
        })
    return rows

def run_layer_suite():
    results = [
        run_probe_rmsnorm(),
        run_probe_ffn(),
        run_probe_rope(),
        run_probe_attn_block(),
        run_probe_gqa(),
        run_probe_transformer(),
    ]

    all_rows = []
    for res in results:
        print_profile_results(res)  # keep your console output
        all_rows.extend(flatten_probe_result(res))

    return {
        "suite": "layers",
        "rows": all_rows,
    }

def run_profiles(save_root="logs/memory"):
    out_dir = make_run_dir(save_root)

    # Profile 1: layers
    layer_suite = run_layer_suite()
    save_csv(layer_suite["rows"], out_dir / "layers_profile.csv")

    # Profile 2: batch capacity
    cap = find_max_batch_size()
    save_csv(cap["attempts"], out_dir / "batch_capacity_attempts.csv")

    # Also save a 1-row summary
    summary = [{
        "max_batch": cap.get("max_batch"),
        "delta_peak_mb": None if not cap.get("stats") else cap["stats"].get("delta_peak_mb"),
        "peak_mb": None if not cap.get("stats") else cap["stats"].get("peak_mb"),
        "current_mb": None if not cap.get("stats") else cap["stats"].get("current_mb"),
        "loss": None if not cap.get("stats") else cap["stats"].get("loss"),
    }]
    save_csv(summary, out_dir / "batch_capacity_summary.csv")

    print(f"\nSaved logs to: {out_dir}")
    
run_profiles(save_root="logs/memory")