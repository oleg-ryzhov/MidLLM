"""
MidLLM trainer server for Google Colab (T4 GPU)  -  NO EDITING NEEDED.

1. Colab: Runtime > Change runtime type > T4 GPU.
2. Paste this whole file into ONE cell and run it. Leave the cell running.
3. It prints a "connection code". Paste that code into the dashboard and press Connect.
   Everything else (dataset, settings, Start / Pause / Stop, nonstop mode) is done in the dashboard.

The model mirrors the dashboard's TF.js architecture 1:1, so trained weights are
loaded into the browser for generation, attention maps and inspection.
"""
import base64
import json
import math
import os
import platform
import re
import secrets
import subprocess
import threading
import time
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_COLAB = "COLAB_GPU" in os.environ or os.path.exists("/content")
PORT = 8765
TOKEN = secrets.token_urlsafe(16)
CKPT_PATH = "/content/midllm_model.json" if IN_COLAB else "midllm_model.json"
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}"

DEFAULTS = dict(blockSize=64, embedDim=64, heads=4, layers=4, batchSize=64,
                lr=3e-3, maxSteps=5000, nonstop=False, evalEvery=250, ckptMinutes=10)
EVAL_BATCHES = 20
MAX_HISTORY = 1500      # chart points are thinned automatically so overnight runs stay small

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"
gpu_name = torch.cuda.get_device_name(0) if use_amp else "CPU"
print(f"Device: {gpu_name}")
if not use_amp:
    print("WARNING: no GPU found. Runtime > Change runtime type > T4 GPU, then run again.")


# ----------------------------------------------------------------------------
# Model (mirrors the TF.js architecture)
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.nh = nh
        self.ln1 = nn.LayerNorm(d, eps=1e-5)
        self.c_attn = nn.Linear(d, 3 * d, bias=False)
        self.c_proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d, eps=1e-5)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.nh, C // self.nh).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.c_proj(y)
        return x + self.fc2(F.relu(self.fc1(self.ln2(x))))


class MidLLM(nn.Module):
    def __init__(self, vocab, block, d, nh, nl):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(block, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.ln_f = nn.LayerNorm(d, eps=1e-5)
        self.head = nn.Linear(d, vocab)
        self.apply(self._init)
        for b in self.blocks:  # GPT-2 style scaled init for residual projections
            nn.init.normal_(b.c_proj.weight, std=0.02 / math.sqrt(2 * nl))
            nn.init.normal_(b.fc2.weight, std=0.02 / math.sqrt(2 * nl))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


def loss_fn(logits, y):
    return F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1))


def pack(name, tensor, transpose=False):
    """float32 little-endian base64; Linear weights transposed to the TF.js [in, out] layout."""
    a = tensor.detach().float().cpu()
    if transpose:
        a = a.t()
    a = a.contiguous().numpy().astype("<f4")
    return {"name": name, "shape": list(a.shape), "data": base64.b64encode(a.tobytes()).decode("ascii")}


def export_weights(m):
    w = [pack("token_embed", m.tok_emb.weight), pack("pos_embed", m.pos_emb.weight)]
    for i, b in enumerate(m.blocks):
        w += [pack(f"b{i}.ln1.gamma", b.ln1.weight), pack(f"b{i}.ln1.beta", b.ln1.bias),
              pack(f"b{i}.c_attn", b.c_attn.weight, True), pack(f"b{i}.c_proj", b.c_proj.weight, True),
              pack(f"b{i}.ln2.gamma", b.ln2.weight), pack(f"b{i}.ln2.beta", b.ln2.bias),
              pack(f"b{i}.fc1.kernel", b.fc1.weight, True), pack(f"b{i}.fc1.bias", b.fc1.bias),
              pack(f"b{i}.fc2.kernel", b.fc2.weight, True), pack(f"b{i}.fc2.bias", b.fc2.bias)]
    w += [pack("ln_f.gamma", m.ln_f.weight), pack("ln_f.beta", m.ln_f.bias),
          pack("head.kernel", m.head.weight, True), pack("head.bias", m.head.bias)]
    return w


# ----------------------------------------------------------------------------
# Trainer: one background thread, controlled through start/pause/resume/stop
# ----------------------------------------------------------------------------
class Trainer:
    def __init__(self):
        self.lock = threading.Lock()                 # held during each optimisation step
        self.export_wanted = threading.Event()       # lets exports slip between steps
        self.stop_ev = threading.Event()
        self.pause_ev = threading.Event()
        self.thread = None
        self.state, self.error = "idle", None
        self.run_id = 0
        self.model = self.opt = self.scaler = None
        self.arch = None
        self.chars, self.stoi = [], {}
        self.text = None
        self.data = self.train_data = self.val_data = None
        self.block = self.batch = 0
        self.max_lr, self.max_steps, self.nonstop = 0.0, 0, False
        self.warmup, self.eval_every, self.ckpt_minutes = 100, 250, 10
        self.step = self.session_step = 0
        self.history, self.val_hist, self.samples = [], [], []
        self.log_every = 10
        self.train_loss = self.val_loss = None
        self.tok_s, self.active, self.cur_lr = 0.0, 0.0, 0.0
        self.n_params = 0
        self.last_ckpt = None
        self._rl, self._rn = torch.zeros((), device=device), 0
        self._tok = self._tok_mark = 0
        self._t_mark = time.time()
        self._last_eval_step = 0

    # ---- control ------------------------------------------------------
    def start(self, text, cfg, cont):
        if self.thread and self.thread.is_alive():
            raise ValueError("Training is already running.")
        c = {**DEFAULTS, **{k: v for k, v in (cfg or {}).items() if v is not None}}
        block, d, nh, nl = int(c["blockSize"]), int(c["embedDim"]), int(c["heads"]), int(c["layers"])
        batch, lr = int(c["batchSize"]), float(c["lr"])
        nonstop, max_steps = bool(c["nonstop"]), int(c["maxSteps"])
        if min(block, d, nh, nl, batch) < 1 or lr <= 0 or (not nonstop and max_steps < 1):
            raise ValueError("Invalid settings (all sizes, the learning rate and the step count must be positive).")
        if d % nh:
            raise ValueError(f"Embedding dim ({d}) must be divisible by the number of heads ({nh}).")
        text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        chars = sorted(set(text))
        if len(chars) < 2:
            raise ValueError("The dataset needs at least 2 different characters.")
        if len(text) < block + 2:
            raise ValueError(f"The dataset ({len(text)} chars) is shorter than the context window ({block}).")

        stoi = {ch: i for i, ch in enumerate(chars)}
        data = torch.from_numpy(np.fromiter((stoi[ch] for ch in text), dtype=np.int64, count=len(text)))
        n_val = int(len(data) * 0.1)
        if n_val >= block + 2 and len(data) - n_val >= block + 2:
            train, val = data[:-n_val], data[-n_val:]
        else:
            train = val = data           # tiny corpus: validate on the training data

        compatible = bool(cont) and self.model is not None and self.arch == (block, d, nh, nl) and self.chars == chars
        with self.lock:
            if not compatible:
                self.model, self.opt = None, None
                if use_amp:
                    torch.cuda.empty_cache()
                self.model = MidLLM(len(chars), block, d, nh, nl).to(device)
                decay = [p for p in self.model.parameters() if p.dim() >= 2]
                other = [p for p in self.model.parameters() if p.dim() < 2]
                self.opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                                              {"params": other, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))
                self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
                self.arch, self.chars = (block, d, nh, nl), chars
                self.n_params = sum(p.numel() for p in self.model.parameters())
                self.step = 0
                self.history, self.val_hist, self.samples = [], [], []
                self.log_every = 10
                self.train_loss = self.val_loss = None
                self.active = 0.0
                self.run_id += 1
                self.last_ckpt = None
            self.stoi = stoi
            self.text = text if len(text) <= 1_000_000 else None
            self.data = data
            self.train_data, self.val_data = train.to(device), val.to(device)
            self.block, self.batch = block, batch
            self.max_lr, self.max_steps, self.nonstop = lr, max_steps, nonstop
            self.warmup = max(1, min(100, max_steps // 10)) if not nonstop else 100
            self.eval_every = max(1, int(c["evalEvery"]))
            self.ckpt_minutes = max(1, int(c["ckptMinutes"]))
            self._last_eval_step = self.step
        self.stop_ev.clear()
        self.pause_ev.clear()
        self.error = None
        self.session_step = 0
        self.state = "training"
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return {"continued": compatible, "params": self.n_params}

    def pause(self):
        if self.state != "training":
            raise ValueError("Not training.")
        self.pause_ev.set()

    def resume(self):
        if self.state != "paused" and not self.pause_ev.is_set():
            raise ValueError("Not paused.")
        self.pause_ev.clear()

    def stop(self):
        self.stop_ev.set()
        self.pause_ev.clear()

    # ---- helpers ------------------------------------------------------
    def lr_at(self, i):
        if i < self.warmup:
            return self.max_lr * (i + 1) / self.warmup
        if self.nonstop:
            return self.max_lr
        prog = min(1.0, (i - self.warmup) / max(1, self.max_steps - self.warmup))
        return self.max_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    def get_batch(self, split):
        d = self.train_data if split == "train" else self.val_data
        T = self.block
        ix = torch.randint(0, len(d) - T, (self.batch,), device=device)   # start <= len-T-1
        offs = torch.arange(T, device=device)
        return d[ix[:, None] + offs], d[ix[:, None] + offs + 1]

    @torch.no_grad()
    def estimate_val(self):
        total = 0.0
        for _ in range(EVAL_BATCHES):
            x, y = self.get_batch("val")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                total += loss_fn(self.model(x), y).item()
        return total / EVAL_BATCHES

    @torch.no_grad()
    def sample(self, n=120, temperature=0.8, top_k=10):
        src = self.text if self.text else "".join(self.chars)
        ids = [self.stoi[ch] for ch in src[:10] if ch in self.stoi] or [0]
        for _ in range(n):
            ctx = torch.tensor([ids[-self.block:]], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = self.model(ctx)[0, -1].float() / temperature
            v, i = torch.topk(logits, min(top_k, len(self.chars)))
            ids.append(i[torch.multinomial(F.softmax(v, dim=-1), 1)].item())
        return "".join(self.chars[i] for i in ids)

    def _record(self, force_eval=False):
        if self._rn:
            tl = (self._rl / self._rn).item()
            self._rl, self._rn = torch.zeros((), device=device), 0
            if not math.isfinite(tl):
                raise RuntimeError("Loss became NaN/inf. Lower the learning rate and start again.")
            self.train_loss = tl
            now = time.time()
            self.tok_s = (self._tok - self._tok_mark) / max(now - self._t_mark, 1e-6)
            self._tok_mark, self._t_mark = self._tok, now
            self.history.append({"step": self.step, "train": tl, "val": None})
            if len(self.history) > MAX_HISTORY:
                self.history = self.history[::2]
                self.log_every *= 2
        if self.train_loss is None:
            return
        if force_eval or self.step - self._last_eval_step >= self.eval_every:
            self._last_eval_step = self.step
            self.val_loss = self.estimate_val()
            self.val_hist.append({"step": self.step, "train": None, "val": self.val_loss})
            if len(self.val_hist) > MAX_HISTORY // 3:
                self.val_hist = self.val_hist[::2]
                self.eval_every *= 2
            self.samples.append({"step": self.step, "text": self.sample()})
            self.samples = self.samples[-50:]

    def combined_history(self):
        return sorted(self.history + self.val_hist, key=lambda p: p["step"])

    # ---- training loop ------------------------------------------------
    def _loop(self):
        try:
            self._tok = self._tok_mark = 0
            self._t_mark = prev = time.time()
            last_ckpt = prev
            while not self.stop_ev.is_set():
                if self.pause_ev.is_set():
                    self.state = "paused"
                    while self.pause_ev.is_set() and not self.stop_ev.is_set():
                        time.sleep(0.2)
                    if self.stop_ev.is_set():
                        break
                    self.state = "training"
                    self._t_mark = prev = time.time()
                    self._tok_mark = self._tok
                if not self.nonstop and self.session_step >= self.max_steps:
                    break
                now = time.time()
                self.active += now - prev
                prev = now
                while self.export_wanted.is_set():
                    time.sleep(0.005)
                with self.lock:
                    lr = self.lr_at(self.session_step)
                    for g in self.opt.param_groups:
                        g["lr"] = lr
                    x, y = self.get_batch("train")
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        loss = loss_fn(self.model(x), y)
                    self.opt.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                self._rl += loss.detach()
                self._rn += 1
                self.step += 1
                self.session_step += 1
                self._tok += self.batch * self.block
                self.cur_lr = lr
                if self.step % self.log_every == 0:
                    self._record()
                    if time.time() - last_ckpt > self.ckpt_minutes * 60:
                        self.save_checkpoint()
                        last_ckpt = time.time()
            self._record(force_eval=True)
            self.save_checkpoint()
            self.state = "stopped" if self.stop_ev.is_set() else "finished"
        except Exception as e:  # report to the dashboard instead of dying silently
            traceback.print_exc()
            self.error = f"{type(e).__name__}: {e}"
            self.state = "error"
            try:
                self.save_checkpoint()
            except Exception:
                pass

    # ---- export -------------------------------------------------------
    def export(self):
        if self.model is None:
            return None
        self.export_wanted.set()
        try:
            with self.lock:
                weights = export_weights(self.model)
                ids = self.data[:self.block].tolist()
                with torch.no_grad():
                    logits = self.model(torch.tensor([ids], device=device))[0, -1].float().cpu().tolist()
        finally:
            self.export_wanted.clear()
        block, d, nh, nl = self.arch
        payload = {
            "format": "midllm-pytorch-v1",
            "config": {"blockSize": block, "embedDim": d, "heads": nh, "layers": nl, "vocabSize": len(self.chars)},
            "chars": self.chars,
            "training": {"steps": self.step, "batchSize": self.batch, "lr": self.max_lr, "params": self.n_params,
                         "finalTrainLoss": self.train_loss, "finalValLoss": self.val_loss,
                         "durationSec": self.active, "device": device},
            "history": self.combined_history(),
            "samples": list(self.samples),
            "check": {"ids": ids, "logits": logits},
            "weights": weights,
        }
        if self.text is not None:
            payload["datasetText"] = self.text
        return json.dumps(payload).encode("utf-8")

    def save_checkpoint(self):
        blob = self.export()
        if blob is None:
            return
        tmp = CKPT_PATH + ".tmp"
        with open(tmp, "wb") as f:
            f.write(blob)
        os.replace(tmp, CKPT_PATH)
        self.last_ckpt = {"step": self.step, "path": CKPT_PATH, "time": time.time()}

    def status(self):
        return {
            "state": self.state, "error": self.error, "runId": self.run_id,
            "hasModel": self.model is not None, "step": self.step, "sessionStep": self.session_step,
            "maxSteps": None if self.nonstop else self.max_steps, "nonstop": self.nonstop,
            "trainLoss": self.train_loss, "valLoss": self.val_loss, "tokPerSec": self.tok_s,
            "elapsed": self.active, "lr": self.cur_lr, "params": self.n_params,
            "device": device, "gpu": gpu_name, "lastCheckpoint": self.last_ckpt,
        }


trainer = Trainer()


# ----------------------------------------------------------------------------
# HTTP API (token protected, CORS open so the dashboard can call it from anywhere)
# ----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Token")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode("utf-8"))

    def _authed(self):
        got = self.headers.get("X-Token", "")
        return secrets.compare_digest(got.encode("utf-8"), TOKEN.encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if not self._authed():
            return self._json(401, {"error": "unauthorized"})
        path = self.path.split("?")[0]
        try:
            if path == "/status":
                self._json(200, trainer.status())
            elif path == "/history":
                self._json(200, {"history": trainer.combined_history(), "samples": list(trainer.samples)})
            elif path == "/model":
                blob = trainer.export()
                if blob is None:
                    self._json(404, {"error": "No model yet. Start training first."})
                else:
                    self._send(200, blob)
            else:
                self._json(404, {"error": "not found"})
        except Exception as e:
            traceback.print_exc()
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        if not self._authed():
            return self._json(401, {"error": "unauthorized"})
        path = self.path.split("?")[0]
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if path == "/start":
                self._json(200, trainer.start(body.get("text"), body.get("config"), body.get("continue", False)))
            elif path == "/pause":
                trainer.pause()
                self._json(200, {"ok": True})
            elif path == "/resume":
                trainer.resume()
                self._json(200, {"ok": True})
            elif path == "/stop":
                trainer.stop()
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "not found"})
        except ValueError as e:
            self._json(400, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            self._json(500, {"error": f"{type(e).__name__}: {e}"})


def start_tunnel(port):
    """Public HTTPS URL for the local server through a Cloudflare quick tunnel (no account needed)."""
    arch = "arm64" if platform.machine() in ("aarch64", "arm64") else "amd64"
    exe = "/tmp/cloudflared"
    if not os.path.exists(exe):
        print("Downloading cloudflared...")
        urllib.request.urlretrieve(CLOUDFLARED_URL.format(arch=arch), exe)
        os.chmod(exe, 0o755)
    proc = subprocess.Popen([exe, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    found = {}

    def drain():  # keep reading so the pipe never fills up
        for line in proc.stdout:
            m = pat.search(line)
            if m and "url" not in found:
                found["url"] = m.group(0)

    threading.Thread(target=drain, daemon=True).start()
    deadline = time.time() + 60
    while time.time() < deadline and "url" not in found and proc.poll() is None:
        time.sleep(0.2)
    if "url" not in found:
        proc.terminate()
        raise RuntimeError("cloudflared did not return a public URL")
    return proc, found["url"]


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    proc = None
    try:
        proc, url = start_tunnel(PORT)
        line = "=" * 64
        print(f"\n{line}\nCONNECTION CODE (paste into the dashboard, then press Connect):\n\n{url}#{TOKEN}\n\n{line}")
    except Exception as e:
        print(f"Could not open a public tunnel ({e}).")
        print(f"Local-only code (dashboard running on the same machine): http://127.0.0.1:{PORT}#{TOKEN}")
    print("Leave this cell running. Interrupt it (stop button) to shut the trainer down.")
    try:
        while True:
            time.sleep(300)
            s = trainer.status()
            loss = f"{s['trainLoss']:.4f}" if s["trainLoss"] is not None else "--"
            print(f"[{time.strftime('%H:%M')}] {s['state']} | step {s['step']:,} | loss {loss}")
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        trainer.stop()
        if trainer.thread:
            trainer.thread.join(timeout=60)
        if proc:
            proc.terminate()
        server.shutdown()


if __name__ == "__main__":
    main()
