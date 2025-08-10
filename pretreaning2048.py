

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAIRA — Lightweight pretraining script (decoder-only, GPT-2 style) for M1 8GB.

• Uses our SentencePiece tokenizer (.model) directly — no HF tokenizer needed.
• Packs text into fixed-length blocks (no padding except possibly last block).
• Works on CPU or Apple MPS (Metal) if available.

Quick start (from project root):
  python motorlar/pretraning.kaira.py \
    --sp_model models/kaira_tokenizer_forced.model \
    --data_glob "dataset_parts/**/*.txt" \
    --save_dir models/pretrain_runs/kaira_gpt2_small \
    --block_size 1024 --batch_size 6 --accum 8 --max_steps 3000 --lr 3e-4

Tip: Start small (max_steps=300–1000) to validate, then extend.
"""

import os
import re
import sys
import glob
import math
import time
import json
import random
from datetime import datetime
import argparse
from dataclasses import dataclass
from typing import List, Iterator

# ----------------------------
# Progress bar (tqdm) with safe fallback
# ----------------------------
try:
    from tqdm import tqdm as _tqdm
    def get_tqdm(disable=False, **kwargs):
        return _tqdm(disable=disable, **kwargs)
except Exception:
    class _DummyTQDM:
        def __init__(self, *a, **k): pass
        def update(self, *a, **k): pass
        def set_postfix(self, *a, **k): pass
        def close(self): pass
    def get_tqdm(disable=False, **kwargs):
        return _DummyTQDM()

import torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader

try:
    import sentencepiece as spm
except Exception as e:
    print("[ERR] sentencepiece import failed:", e)
    sys.exit(2)

try:
    from transformers import GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup
except Exception as e:
    print("[ERR] transformers import failed. Install: pip install transformers")
    raise

# ----------------------------
# Utils
# ----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def detect_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------
# Dataset — SP packer
# ----------------------------
@dataclass
class SPOptions:
    add_bos: bool = True
    add_eos: bool = True
    sampling: bool = True
    nbest_size: int = -1   # -1 = sample from full
    alpha: float = 0.1
    no_sampling_over: int = 1000  # disable sampling for segments longer than this many chars (0=always sample)
    verbose: bool = False


class PackedSpDataset(IterableDataset):
    """Streams multiple .txt files, tokenizes with SentencePiece and packs into fixed blocks."""
    def __init__(self, files: List[str], sp: spm.SentencePieceProcessor, block_size: int,
                 bos_id: int, eos_id: int, pad_id: int, opts: SPOptions,
                 max_line_chars: int = 1800, drop_last: bool = False,
                 verbose: bool = False, scan_log_every: int = 50000, total_files: int = 0):
        self.files = files
        self.sp = sp
        self.block = block_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.opts = opts
        self.drop_last = drop_last
        self.max_line_chars = max_line_chars
        self.verbose = verbose or getattr(self.opts, "verbose", False)
        self.scan_log_every = scan_log_every
        self.total_files = total_files if total_files else len(files)

    def _encode(self, text: str) -> List[int]:
        # Avoid SentencePiece "Too big agenda size" warnings by disabling sampling
        # for very long segments.
        use_sampling = self.opts.sampling and (
            self.opts.no_sampling_over <= 0 or len(text) <= self.opts.no_sampling_over
        )
        if use_sampling:
            return self.sp.encode(
                text, out_type=int, enable_sampling=True,
                alpha=self.opts.alpha, nbest_size=self.opts.nbest_size
            )
        else:
            return self.sp.encode(text, out_type=int)

    def _split_long_by_ws(self, s: str):
        """
        Yield segments of s, each at most self.max_line_chars chars.
        Prefer to split at the last whitespace before the limit.
        """
        if len(s) <= self.max_line_chars:
            yield s
            return
        n = len(s)
        start = 0
        limit = self.max_line_chars
        while start < n:
            end = min(start + limit, n)
            if end < n:
                # search backward for whitespace near the limit
                cut = -1
                j = end - 1
                # avoid cutting too early; allow search window down to start
                while j > start:
                    if s[j].isspace():
                        cut = j
                        break
                    j -= 1
                if cut == -1 or cut <= start:
                    cut = end
            else:
                cut = n
            seg = s[start:cut]
            if seg:
                yield seg
            start = cut

    def __iter__(self) -> Iterator[dict]:
        buf: List[int] = []
        file_count = 0
        for fp in self.files:
            file_count += 1
            line_count = 0
            char_count = 0
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    # keep inner spaces; only drop trailing newline
                    s = line.rstrip("\n")
                    line_count += 1
                    char_count += len(line)
                    if self.verbose and (line_count % max(1, self.scan_log_every) == 0):
                        print(f"[SCAN] {file_count}/{self.total_files} files | lines={line_count:,} | chars={char_count/1e6:.1f}M", flush=True)
                    if not s.strip():
                        continue
                    for seg in self._split_long_by_ws(s):
                        toks = self._encode(seg)
                        if self.opts.add_bos:
                            toks = [self.bos_id] + toks
                        if self.opts.add_eos:
                            toks = toks + [self.eos_id]
                        buf.extend(toks)
                        # pack
                        while len(buf) >= self.block:
                            chunk = buf[:self.block]
                            buf = buf[self.block:]
                            yield {
                                "input_ids": torch.tensor(chunk, dtype=torch.long),
                                "attention_mask": torch.ones(self.block, dtype=torch.long),
                                "labels": torch.tensor(chunk, dtype=torch.long),  # causal LM: shift inside model
                            }
        # tail
        if not self.drop_last and buf:
            # pad tail to block size
            pad_len = self.block - len(buf)
            buf = buf + [self.pad_id] * pad_len
            yield {
                "input_ids": torch.tensor(buf, dtype=torch.long),
                "attention_mask": torch.tensor([1]* (self.block - pad_len) + [0]*pad_len, dtype=torch.long),
                "labels": torch.tensor(buf, dtype=torch.long),
            }


# ----------------------------
# Dataloader collate — top-level (picklable for multiprocessing)
# ----------------------------
def collate(batch: List[dict]):
    # All blocks are fixed length already; just stack.
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attn = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


# ----------------------------
# Training loop
# ----------------------------

def save_ckpt(model, optimizer, scheduler, step, save_dir, cfg):
    os.makedirs(save_dir, exist_ok=True)
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "cfg": cfg,
    }
    path = os.path.join(save_dir, f"ckpt_step_{step}.pt")
    torch.save(state, path)
    with open(os.path.join(save_dir, "last.json"), "w") as f:
        json.dump({"last": path, "step": step}, f)
    print(f"[CKPT] saved {path}")


def load_last_ckpt(model, optimizer, scheduler, save_dir):
    last_file = os.path.join(save_dir, "last.json")
    if not os.path.exists(last_file):
        return 0
    meta = json.load(open(last_file))
    path = meta.get("last")
    if not path or not os.path.exists(path):
        return 0
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"], strict=False)
    if optimizer and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    print(f"[CKPT] resumed from {path}")
    return int(state.get("step", 0))


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp_model", default="models/kaira_tokenizer_forced.model")
    ap.add_argument("--data_glob", default="dataset_parts/**/*.txt")
    ap.add_argument("--save_dir", default="models/pretrain_runs/kaira_gpt2_small")
    ap.add_argument("--block_size", type=int, default=1024, help="fixed block size for packing (no padding except last block)")
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--accum", type=int, default=8, help="gradient accumulation steps")
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_sampling", action="store_true", help="disable SP subword sampling")
    ap.add_argument("--sp_nbest", type=int, default=64, help="nbest size for SP sampling when enabled")
    ap.add_argument("--sp_alpha", type=float, default=0.1, help="alpha for SP sampling when enabled")
    ap.add_argument("--no_sampling_over", type=int, default=1000,
                    help="disable SP sampling when a segment is longer than this many characters (0=always sample)")
    ap.add_argument("--max_line_chars", type=int, default=1800, help="split lines longer than this many chars before tokenizing")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (0=main thread)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--use_checkpoint", action="store_true", help="Enable gradient checkpointing to save memory")
    ap.add_argument("--grad_clip", type=float, default=1.0, help="Gradient norm clipping")
    ap.add_argument("--grad_mul", type=float, default=1.0, help="Multiply loss before backward to amplify gradients")
    ap.add_argument("--lr_scale_with_accum", action="store_true", help="Linearly scale LR by accumulation steps")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--auto_tune", action="store_true", help="Auto backoff on OOM (batch/block) and continue training")
    ap.add_argument("--keep_total_tokens", action="store_true", help="Keep total tokens per update roughly constant when batch shrinks (scale accum)")
    ap.add_argument("--max_accum", type=int, default=64, help="Upper bound for gradient accumulation when scaling")
    ap.add_argument("--min_bs", type=int, default=1, help="Lower bound for micro-batch size during backoff")
    ap.add_argument("--min_block", type=int, default=256, help="Lower bound for block_size during backoff")
    ap.add_argument("--bs_backoff", type=float, default=0.5, help="Multiply batch_size by this factor on OOM (0.5 halves it)")
    ap.add_argument("--block_backoff", type=float, default=0.75, help="Multiply block_size by this factor on repeated OOM")
    ap.add_argument("--oom_patience", type=int, default=2, help="How many consecutive OOMs before backing off block size")
    ap.add_argument("--teacher_hf", type=str, default="", help="Path to HF dir for teacher model (for KL distillation)")
    ap.add_argument("--distill_alpha", type=float, default=0.0, help="Weight for KL distillation loss (0=off)")
    ap.add_argument("--distill_T", type=float, default=1.0, help="Temperature for distillation")
    ap.add_argument("--ema", action="store_true", help="Enable EMA of weights for stability")
    ap.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay (e.g., 0.999)")
    ap.add_argument("--freeze_layers", type=int, default=0, help="Freeze lowest N transformer blocks to reduce forgetting")
    ap.add_argument("--anchor_reg", type=float, default=0.0, help="L2 anchor regularization strength toward initial weights (EWC-lite)")
    ap.add_argument("--progress", action="store_true", help="Enable tqdm progress bars/logs")
    ap.add_argument("--scan_log_every", type=int, default=50000, help="Log every N lines per file during scanning")
    ap.add_argument("--val_glob", type=str, default="", help="Validation files glob (e.g., 'dataset_parts_val/**/*.txt'). If empty, validation is disabled.")
    ap.add_argument("--val_every", type=int, default=0, help="Run validation every N true updates (0=off)")
    ap.add_argument("--val_max_batches", type=int, default=50, help="Max validation batches to average per eval")
    ap.add_argument("--log_csv", type=str, default="", help="Write metrics to this CSV path (default: save_dir/train_log.csv)")
    ap.add_argument("--save_best", action="store_true", help="Save ckpt_best.pt when validation loss improves")
    ap.add_argument("--autosave_secs", type=int, default=600, help="Time-based autosave interval in seconds (0=off)")
    ap.add_argument("--save_on_interrupt", action="store_true", help="Save checkpoint on KeyboardInterrupt (Ctrl+C)")
    # Memory helpers
    def maybe_empty_cache():
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass
    args = ap.parse_args()

    # Setup metrics CSV path
    if not args.log_csv:
        args.log_csv = os.path.join(args.save_dir, "train_log.csv")
    os.makedirs(args.save_dir, exist_ok=True)
    if not os.path.exists(args.log_csv):
        with open(args.log_csv, "w", encoding="utf-8") as f:
            f.write("time,step,train_loss,train_ppl,val_loss,val_ppl,lr,bs,block,accum\n")

    set_seed(args.seed)
    device = detect_device()

    # Load SP tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.sp_model)
    vocab_size = sp.get_piece_size()
    ids = {
        "unk": sp.piece_to_id("<unk>"),
        "bos": sp.piece_to_id("<s>"),
        "eos": sp.piece_to_id("</s>"),
        "pad": sp.piece_to_id("<pad>"),
    }
    assert ids["unk"] == 0 and ids["bos"] == 1 and ids["eos"] == 2 and ids["pad"] == 3, \
        f"Special IDs mismatch: {ids}"

    # Model config
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=args.block_size,
        n_ctx=args.block_size,
        n_embd=args.d_model,
        n_layer=args.layers,
        n_head=args.n_head,
        resid_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        bos_token_id=ids["bos"],
        eos_token_id=ids["eos"],
        pad_token_id=ids["pad"],
        tie_word_embeddings=True,
    )
    model = GPT2LMHeadModel(cfg)

    # Optionally enable gradient checkpointing and disable use_cache during training
    if args.use_checkpoint:
        try:
            model.gradient_checkpointing_enable()
            print("[CHKPT] Gradient checkpointing enabled")
        except Exception as e:
            print(f"[CHKPT] Could not enable checkpointing: {e}")
    # Disable cache during training to reduce memory
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Optionally freeze lowest N blocks (protect base knowledge)
    if args.freeze_layers > 0:
        try:
            for i in range(min(args.freeze_layers, len(model.transformer.h))):
                for p in model.transformer.h[i].parameters():
                    p.requires_grad = False
            print(f"[FREEZE] Frozen lowest {args.freeze_layers} transformer blocks")
        except Exception as e:
            print(f"[FREEZE] Could not freeze layers: {e}")

    # Teacher for KL distillation
    teacher = None
    if args.distill_alpha > 0.0:
        try:
            if args.teacher_hf:
                teacher = GPT2LMHeadModel.from_pretrained(args.teacher_hf)
                print(f"[DISTILL] Loaded teacher from {args.teacher_hf}")
            else:
                import copy
                teacher = copy.deepcopy(model)
                print("[DISTILL] Using self-snapshot as teacher")
            teacher.to(device)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False
        except Exception as e:
            print(f"[DISTILL] Failed to set up teacher: {e}")
            teacher = None

    # EMA of weights
    ema_state = None
    if args.ema:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"[EMA] Enabled (decay={args.ema_decay})")
    def ema_update():
        if not args.ema:
            return
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in ema_state:
                    ema_state[k].mul_(args.ema_decay).add_(v.detach(), alpha=(1.0 - args.ema_decay))
                else:
                    ema_state[k] = v.detach().clone().to(v.device)

    # Anchor regularization (EWC-lite)
    anchor_params = None
    if args.anchor_reg > 0.0:
        anchor_params = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        print(f"[ANCHOR] Enabled L2-to-initial with lambda={args.anchor_reg}")

    # Device
    model.to(device)
    # Make sure EMA tensors are on the same device as the model (fixes MPS/CPU mismatch)
    if args.ema and ema_state is not None:
        for k in list(ema_state.keys()):
            ema_state[k] = ema_state[k].to(device)
    model.train()

    # Optimizer & scheduler
    base_lr = args.lr * (args.accum if args.lr_scale_with_accum else 1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=args.wd)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )

    # Dataset & loader
    files = sorted(glob.glob(args.data_glob, recursive=True))
    assert files, f"No files matched: {args.data_glob}"

    sp_opts = SPOptions(
        add_bos=True, add_eos=True,
        sampling=(not args.no_sampling),
        alpha=args.sp_alpha, nbest_size=args.sp_nbest,
        no_sampling_over=args.no_sampling_over,
        verbose=args.progress
    )
    ds = PackedSpDataset(
        files, sp, args.block_size, ids["bos"], ids["eos"], ids["pad"], sp_opts,
        max_line_chars=args.max_line_chars, drop_last=False,
        verbose=args.progress, scan_log_every=args.scan_log_every, total_files=len(files)
    )

    # Validation loader (optional)
    val_loader = None
    if args.val_glob:
        val_files = sorted(glob.glob(args.val_glob, recursive=True))
        if val_files:
            val_sp_opts = SPOptions(add_bos=True, add_eos=True, sampling=False, alpha=0.0, nbest_size=1, no_sampling_over=0, verbose=False)
            val_ds = PackedSpDataset(
                val_files, sp, args.block_size, ids["bos"], ids["eos"], ids["pad"], val_sp_opts,
                max_line_chars=args.max_line_chars, drop_last=False, verbose=False, scan_log_every=args.scan_log_every, total_files=len(val_files)
            )
            def make_val_loader(bs:int):
                return DataLoader(val_ds, batch_size=bs, collate_fn=collate, num_workers=0)
            # keep eval micro-batch modest for M1
            val_loader = make_val_loader(min(4, max(1, int(args.batch_size/2))))
        else:
            print("[VAL] No files matched for validation; disabling.")
            args.val_every = 0


    # Dynamic knobs for OOM-safe training
    cur_bs = max(1, int(args.batch_size))
    update_every = max(1, int(args.accum))
    cur_block = int(args.block_size)

    def make_loader(bs:int):
        if args.workers > 0:
            return DataLoader(
                ds, batch_size=bs, collate_fn=collate,
                num_workers=args.workers, persistent_workers=True, prefetch_factor=2
            )
        else:
            return DataLoader(ds, batch_size=bs, collate_fn=collate, num_workers=0)

    loader = make_loader(cur_bs)
    print(f"[DATA] DataLoader ready (workers={args.workers}, sp_sampling={not args.no_sampling}, nbest={args.sp_nbest if not args.no_sampling else 1}, alpha={args.sp_alpha if not args.no_sampling else 0.0})")

    # Resume if requested
    start_step = 0
    if args.resume:
        start_step = load_last_ckpt(model, optimizer, scheduler, args.save_dir)

    print(f"[DEV] {device}, vocab={vocab_size}, block={args.block_size}, bs={args.batch_size} x accum={args.accum}")
    print(f"[SP] ids: {ids}")
    print(f"[DATA] {len(files)} files matched. Streaming...")
    print(f"[CFG] checkpointing={args.use_checkpoint}, grad_clip={args.grad_clip}, grad_mul={args.grad_mul}, lr={base_lr} (scale_with_accum={args.lr_scale_with_accum}), log_every={args.log_every}")
    print(f"[DYN] cur_bs={cur_bs}, update_every={update_every}, cur_block={cur_block}")
    print(f"[REG] distill_alpha={args.distill_alpha}, T={args.distill_T}, ema={args.ema}, freeze_layers={args.freeze_layers}, anchor_reg={args.anchor_reg}")
    print(f"[SAVE] save_dir={args.save_dir}")
    print(f"[SAVE] checkpoint every {args.save_every} true updates (~{args.save_every * max(1, int(args.accum))} micro-steps)")
    if args.autosave_secs > 0:
        print(f"[SAVE] autosave every {args.autosave_secs} seconds")

    # Training progress bar over "steps" (each micro-step before an optimizer update)
    pbar = get_tqdm(total=max(1, args.max_steps - start_step), desc="steps", disable=not args.progress)

    if str(device) == "mps":
        print("[TIP] On MPS, you can export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 before running to reduce OOM risk.")

    scaler = torch.cuda.amp.GradScaler(enabled=False)  # MPS doesn't support autocast well yet

    step = start_step
    # Running accumulators for logging
    log_loss_sum = 0.0
    log_loss_count = 0
    best_val = float("inf")

    def write_csv_row(step_num, train_loss, train_ppl, val_loss, val_ppl, lr, bs, block, accum):
        ts = datetime.now().isoformat(timespec="seconds")
        with open(args.log_csv, "a", encoding="utf-8") as f:
            f.write(f"{ts},{step_num},{'' if train_loss is None else f'{train_loss:.6f}'},{'' if train_ppl is None else f'{train_ppl:.3f}'},{'' if val_loss is None else f'{val_loss:.6f}'},{'' if val_ppl is None else f'{val_ppl:.3f}'},{lr:.6f},{bs},{block},{accum}\n")

    def run_validation():
        if val_loader is None or args.val_every <= 0:
            return None, None
        model.eval()
        losses = []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= max(1, args.val_max_batches):
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                losses.append(out.loss.detach().cpu())
        model.train()
        if not losses:
            return None, None
        val_loss = float(torch.stack(losses).mean().item())
        val_ppl = float(math.exp(min(20.0, val_loss)))
        return val_loss, val_ppl

    running = 0.0
    model.train()

    oom_streak = 0
    last_autosave = time.time()
    try:
        while step < args.max_steps:
            for batch in loader:
                try:
                    # Move to device
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    main_loss = outputs.loss  # cross-entropy LM loss

                    # KL distillation from teacher (optional)
                    if teacher is not None and args.distill_alpha > 0.0:
                        with torch.no_grad():
                            t_out = teacher(input_ids=input_ids, attention_mask=attention_mask)
                            t_logits = t_out.logits
                        T = max(1e-6, float(args.distill_T))
                        s_logp = torch.log_softmax(outputs.logits / T, dim=-1)
                        t_p = torch.softmax(t_logits / T, dim=-1)
                        kl_tok = torch.sum(t_p * (torch.log(t_p + 1e-8) - s_logp), dim=-1)  # KL(P||Q)
                        distill_loss = (kl_tok.mean()) * (T * T)
                        main_loss = main_loss + args.distill_alpha * distill_loss

                    # Anchor L2 regularization toward initial weights (EWC-lite)
                    if anchor_params is not None and args.anchor_reg > 0.0:
                        reg = 0.0
                        cnt = 0
                        for n, p in model.named_parameters():
                            if p.requires_grad and n in anchor_params:
                                ap = anchor_params[n].to(p.device)
                                reg = reg + torch.mean((p - ap) ** 2)
                                cnt += 1
                        if cnt > 0:
                            main_loss = main_loss + args.anchor_reg * reg

                    # Scale & backward
                    loss = main_loss * (args.grad_mul / update_every)
                    loss.backward()
                    running += outputs.loss.item()
                    log_loss_sum += outputs.loss.item()
                    log_loss_count += 1

                    if (step + 1) % update_every == 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        ema_update()

                        true_step = (step + 1) // update_every
                        if true_step % args.log_every == 0:
                            avg = log_loss_sum / max(1, log_loss_count)
                            ppl = math.exp(min(20.0, avg))
                            print(f"[STEP {true_step}] train_loss={avg:.4f} train_ppl={ppl:.2f} | bs={cur_bs} block={cur_block} accum={update_every}")
                            write_csv_row(true_step, avg, ppl, None, None, optimizer.param_groups[0]['lr'], cur_bs, cur_block, update_every)
                            log_loss_sum = 0.0
                            log_loss_count = 0
                            running = 0.0
                        if args.val_every > 0 and (true_step % args.val_every == 0):
                            val_loss, val_ppl = run_validation()
                            if val_loss is not None:
                                print(f"[VAL  {true_step}] val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")
                                write_csv_row(true_step, None, None, val_loss, val_ppl, optimizer.param_groups[0]['lr'], cur_bs, cur_block, update_every)
                                if args.save_best and val_loss < best_val:
                                    best_val = val_loss
                                    save_ckpt(model, optimizer, scheduler, true_step, args.save_dir, cfg.to_dict())
                                    save_ema_ckpt(ema_state, true_step, args.save_dir)
                                    # Also mark as best
                                    best_path = os.path.join(args.save_dir, "ckpt_best.pt")
                                    last_meta = {"best_step": true_step, "best_val_loss": best_val}
                                    torch.save({"step": true_step, "model": model.state_dict(), "cfg": cfg.to_dict()}, best_path)
                                    with open(os.path.join(args.save_dir, "best.json"), "w") as f:
                                        json.dump(last_meta, f)
                                    print(f"[CKPT] saved BEST to {best_path}")
                        if true_step % args.save_every == 0:
                            save_ckpt(model, optimizer, scheduler, true_step, args.save_dir, cfg.to_dict())
                            save_ema_ckpt(ema_state, true_step, args.save_dir)
                        # Time-based autosave (independent of save_every)
                        if args.autosave_secs > 0 and (time.time() - last_autosave) >= args.autosave_secs:
                            save_ckpt(model, optimizer, scheduler, true_step, args.save_dir, cfg.to_dict())
                            save_ema_ckpt(ema_state, true_step, args.save_dir)
                            last_autosave = time.time()

                    step += 1
                    # progress bar update
                    try:
                        pbar.update(1)
                        if (step % args.log_every) == 0:
                            try:
                                cur_loss = outputs.loss.item()
                                cur_ppl = math.exp(min(20.0, float(cur_loss)))
                                pbar.set_postfix(bs=cur_bs, block=cur_block, accum=update_every, loss=f"{cur_loss:.3f}", ppl=f"{cur_ppl:.2f}")
                            except Exception:
                                pbar.set_postfix(bs=cur_bs, block=cur_block, accum=update_every)
                    except Exception:
                        pass
                    oom_streak = 0  # success resets streak
                    # free temporaries
                    del outputs, loss

                    if step >= args.max_steps:
                        break

                except RuntimeError as e:
                    msg = str(e).lower()
                    is_oom = ("out of memory" in msg) or ("mps" in msg and "memory" in msg)
                    if not (args.auto_tune and is_oom):
                        raise
                    print(f"[OOM] Caught OOM: {e}")
                    maybe_empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    # Backoff strategy: reduce batch first
                    prev_bs = cur_bs
                    new_bs = max(args.min_bs, int(max(1, math.floor(cur_bs * args.bs_backoff))))
                    if new_bs < cur_bs:
                        cur_bs = new_bs
                        loader = make_loader(cur_bs)
                        if args.keep_total_tokens:
                            # scale accumulation to keep tokens/update approximately constant
                            tokens_old = prev_bs * cur_block * update_every
                            target = max(cur_block * cur_bs, 1)
                            scaled = max(1, int(round(tokens_old / target)))
                            update_every = min(args.max_accum, scaled)
                            if args.lr_scale_with_accum:
                                # adjust LR on-the-fly if requested
                                new_lr = args.lr * update_every
                                for g in optimizer.param_groups:
                                    g['lr'] = new_lr
                        print(f"[BACKOFF] bs {prev_bs} -> {cur_bs}; accum set to {update_every}")
                    else:
                        # Could not reduce bs further; try reducing block (sequence length)
                        oom_streak += 1
                        if oom_streak >= args.oom_patience and cur_block > args.min_block:
                            prev_block = cur_block
                            cur_block = max(args.min_block, int(math.floor(cur_block * args.block_backoff)))
                            ds.block = cur_block  # change packing length
                            loader = make_loader(cur_bs)
                            if args.keep_total_tokens:
                                # rescale accumulation to keep tokens/update
                                tokens_old = prev_bs * prev_block * update_every
                                target = max(cur_block * cur_bs, 1)
                                update_every = min(args.max_accum, max(1, int(round(tokens_old / target))))
                                if args.lr_scale_with_accum:
                                    new_lr = args.lr * update_every
                                    for g in optimizer.param_groups:
                                        g['lr'] = new_lr
                            print(f"[BACKOFF] block {prev_block} -> {cur_block}; accum now {update_every}")
                            oom_streak = 0
                        else:
                            print("[OOM] Skipping this batch (waiting for patience/backoff)...")
                    continue

    except KeyboardInterrupt:
        print("[INTERRUPT] Caught KeyboardInterrupt.")
        if args.save_on_interrupt:
            curr_true_step = max(1, (step // max(1, int(args.accum))))
            save_ckpt(model, optimizer, scheduler, curr_true_step, args.save_dir, cfg.to_dict())
            save_ema_ckpt(ema_state, curr_true_step, args.save_dir)
            print("[INTERRUPT] Checkpoint saved. Exiting.")
        else:
            print("[INTERRUPT] Exiting without saving (pass --save_on_interrupt to enable).")
    finally:
        try:
            pbar.close()
        except Exception:
            pass
        # Final save (also runs on normal completion)
        final_true_step = max(1, (step // max(1, int(args.accum))) + 1)
        save_ckpt(model, optimizer, scheduler, final_true_step, args.save_dir, cfg.to_dict())
        save_ema_ckpt(ema_state, final_true_step, args.save_dir)
        # Save HF compatible export
        hf_dir = os.path.join(args.save_dir, "hf_export")
        os.makedirs(hf_dir, exist_ok=True)
        model.save_pretrained(hf_dir, safe_serialization=False)
        with open(os.path.join(hf_dir, "config.json"), "w") as f:
            f.write(model.config.to_json_string())
        print(f"[DONE] Saved to {args.save_dir} and HF export at {hf_dir}")


# Save EMA checkpoint helper
def save_ema_ckpt(ema_state, step, save_dir):
    if ema_state is None:
        return
    try:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"ckpt_step_{step}_ema.pt")
        cpu_ema = {k: v.detach().to("cpu") for k, v in ema_state.items()}
        torch.save({"step": step, "ema": cpu_ema}, path)
        print(f"[CKPT] saved EMA {path}")
    except Exception as e:
        print(f"[CKPT] EMA save failed: {e}")


if __name__ == "__main__":
    main()