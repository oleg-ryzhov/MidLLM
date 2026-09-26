# MidLLM

MidLLM is a zero-setup, fully browser-based tool for building, training, and testing custom GPT Transformer models. Powered by TensorFlow.js and WebGL/WebGPU acceleration, it runs full neural network workflows 100% locally inside your web browser—requiring no servers, Python, or command-line setup.

---

## Quick Start

1. **Download:** Save `index.html` to your computer.
2. **Launch:** Double-click `index.html` to open it in Chrome, Edge, Firefox, or Safari.
3. **Initialize:** Select a preset dataset (or load a custom `.txt` file) and click **Initialize Model Weights**.
4. **Train:** Click **Start** under **Execution Controls** to begin training immediately.

---

## How to Train Effectively

Training an AI model efficiently requires matching your settings (hyperparameters) to your target text dataset and computer hardware:

### 1. Structural Settings (Requires Clicking "Initialize Model Weights")
* **Context Window ($T$):** How many characters the AI reads at once to predict the next character.
  * *Simple/Fast Training:* `16` – `32`
  * *Complex Text / Long Code Patterns:* `64` – `128`
* **Embedding Dim ($d_{model}$) & Layers ($n_{layer}$):** Internal brain capacity and network depth.
  * *Lightweight (Fast):* Embedding `32`, Layers `2`
  * *Deep (High Accuracy):* Embedding `64`+, Layers `4`+
* **Batch Size:** How many text chunks are processed per learning step.
  * *Higher (16–32):* Faster, smoother learning if you have a dedicated GPU.
  * *Lower (4–8):* Keeps memory usage safe on laptops or integrated graphics.

### 2. Training Speed & Stability Settings
* **Learning Rate ($\eta$):** Step size used to update weights during training.
  * *Aggressive ($0.005 - 0.01$):* Rapid early progress, but risks breaking training (loss explosion).
  * *Balanced ($0.001 - 0.003$):* Ideal sweet spot for standard runs.
  * *Stable ($0.0003 - 0.0005$):* Slow, steady learning for long sessions or large datasets.

---

## Adjusting Generation Parameters (Inference Playground)

Once trained, tweak these generation parameters to shape how the AI generates text:

| Goal / Use Case | Temperature | Top-K | Description |
| :--- | :--- | :--- | :--- |
| **Code, Math, Exact Patterns** | `0.1` – `0.3` | `3` – `5` | Low temperature forces strict predictability and logical accuracy. |
| **Natural Prose & Stories** | `0.7` – `0.8` | `10` – `20` | Balanced settings produce coherent, fluid language. |
| **Creative / Brainstorming** | `1.0` – `1.4` | `40`+ | High values increase randomness and novel word combinations. |

---

## AFK / Overnight Training Recipe

To leave your model training safely overnight without crashing, loss explosion, or browser freezing, apply the following configuration:

1. **Enable Continuous Non-Stop Mode:** Ensure the continuous training toggle is switched ON so training does not halt after a few epochs.
2. **Lower Learning Rate ($\eta = 0.0005 - 0.001$):** Prevents loss explosions (sudden spikes in prediction errors) while running unattended for thousands of steps.
3. **Conservative Batch & Context Size:** Set **Batch Size** to `8` or `16` and **Context Window** to `32` or `64`. This avoids GPU memory exhaustion over long sessions.
4. **Enable Local Storage Backup:** Make sure **Auto-Backup** is active (or link a local `midllm_backup.json` handle) so progress is saved periodically.
5. **Configure PC & Browser Settings:**
   * Keep the browser window **visible** (don't minimize the window, as browsers throttle GPU tasks in hidden tabs).
   * Disable OS sleep/hibernate mode so your computer stays awake throughout the night.

---

## Monitoring Progress

* **Cross-Entropy Loss:** Represents prediction error. Lower is better.
  * *Starting Loss:* `4.0` – `5.0` (random guessing)
  * *Trained Target:* `< 1.5` (learning structure, grammar, and vocabulary)
* **Perplexity:** Measures character choice uncertainty ($e^{\text{loss}}$). A perplexity under `4.0` means the model is confidently picking accurate characters.
