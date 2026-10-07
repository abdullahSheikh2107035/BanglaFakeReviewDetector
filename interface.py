"""
Bangla Fake Review Detector -- Local & Colab Interface (RNN vs LSTM)
======================================================================

Run this locally or on Google Colab.
Features:
  - GUI mode (Tkinter desktop window)
  - Gradio Web interface (works great in Google Colab & web browsers)
  - CLI mode (interactive terminal loop)

Folder layout expected (same folder as this script):

    interface.py
    saved_rnn_model/
        rnn_model.pt
        vocab.joblib
        config.joblib
        label_map.joblib
    saved_lstm_model/
        lstm_model.pt
        vocab.joblib
        config.joblib
        label_map.joblib

Run options:
    python interface.py             # opens GUI (or Gradio on Colab)
    python interface.py --gradio    # launches Gradio web UI
    python interface.py --cli       # runs terminal mode
"""

import os
import re
import sys
import argparse

import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Preprocessing -- identical to the training notebooks
# ---------------------------------------------------------------------------

EMOJI_PATTERN = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF" "\U00002190-\U000021FF" "]+", flags=re.UNICODE)


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = EMOJI_PATTERN.sub(" ", text)
    text = re.sub(r"([\u0964!?.\-_=~*])\1{1,}", r"\1", text)
    text = re.sub(r"(#\w+)(\s*\1){1,}", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def basic_tokenize(text):
    return re.findall(r"[\u0980-\u09FFa-zA-Z0-9]+", text)


def encode(tokens, vocab, max_len):
    unk = vocab["<unk>"]
    ids = [vocab.get(t, unk) for t in tokens[:max_len]]
    return ids + [vocab["<pad>"]] * (max_len - len(ids))


# ---------------------------------------------------------------------------
# Model definitions -- same architectures as the training notebooks
# ---------------------------------------------------------------------------

class RNNClassifier(nn.Module):
    """Vanilla RNN + masked mean-pooling. Matches 05_rnn.ipynb."""

    def __init__(self, embedding_matrix, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32), padding_idx=0)
        embed_dim = embedding_matrix.shape[1]
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        mask = (x != 0).unsqueeze(-1).float()
        pooled = torch.sum(rnn_out * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-8)
        return self.fc(pooled).squeeze(-1)


class LSTMAttentionClassifier(nn.Module):
    """LSTM + additive attention. Matches 03_lstm_attention.ipynb."""

    def __init__(self, embedding_matrix, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32), padding_idx=0)
        embed_dim = embedding_matrix.shape[1]
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        attn_scores = self.attention(lstm_out).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~(x != 0), float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)
        return self.fc(context).squeeze(-1)


# ---------------------------------------------------------------------------
# Loading + prediction
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "RNN":  {"dir": os.path.join(BASE_DIR, "saved_rnn_model"),  "weights": "rnn_model.pt",  "cls": RNNClassifier},
    "LSTM": {"dir": os.path.join(BASE_DIR, "saved_lstm_model"), "weights": "lstm_model.pt", "cls": LSTMAttentionClassifier},
}
_cache = {}


class ModelLoadError(Exception):
    pass


def load_model(name):
    if name in _cache:
        return _cache[name]
    spec = MODELS[name]
    if not os.path.isdir(spec["dir"]):
        raise ModelLoadError(
            f"Couldn't find '{spec['dir']}'.\n\n"
            f"Make sure the '{os.path.basename(spec['dir'])}' folder "
            f"(unzipped) sits next to interface.py."
        )
    try:
        vocab = joblib.load(os.path.join(spec["dir"], "vocab.joblib"))
        config = joblib.load(os.path.join(spec["dir"], "config.joblib"))
        label_map = joblib.load(os.path.join(spec["dir"], "label_map.joblib"))
        dummy_embeddings = np.zeros((len(vocab), config["embed_dim"]), dtype=np.float32)
        model = spec["cls"](dummy_embeddings, hidden_dim=config["hidden_dim"])
        
        weights_path = os.path.join(spec["dir"], spec["weights"])
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(weights_path, map_location="cpu")
            
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        raise ModelLoadError(f"Failed to load '{name}' model: {e}")
    _cache[name] = (model, vocab, config, label_map)
    return _cache[name]


def predict(model_name, text):
    model, vocab, config, label_map = load_model(model_name)
    tokens = basic_tokenize(clean_text(text))
    if not tokens:
        raise ValueError("No recognizable Bangla/alphanumeric text found in the input.")
    ids = encode(tokens, vocab, config["max_seq_len"])
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        prob_fake = torch.sigmoid(model(x)).item()
    verdict = "FAKE" if prob_fake > 0.5 else "AUTHENTIC"
    confidence = prob_fake if verdict == "FAKE" else 1 - prob_fake
    return verdict, confidence


# ---------------------------------------------------------------------------
# Gradio Mode (Web UI / Google Colab)
# ---------------------------------------------------------------------------

def run_gradio(share=False):
    try:
        import gradio as gr
    except ImportError:
        print("Gradio is not installed. Install it using: pip install gradio")
        print("Falling back to CLI mode...")
        run_cli()
        return

    def gradio_predict(model_name, review_text):
        if not review_text or not review_text.strip():
            return "⚠️ Please enter a Bangla review text.", "0%", "N/A"
        try:
            verdict, confidence = predict(model_name, review_text)
            icon = "🚨 FAKE" if verdict == "FAKE" else "✅ AUTHENTIC"
            conf_str = f"{confidence * 100:.2f}%"
            return icon, conf_str, model_name
        except Exception as e:
            return f"❌ Error: {str(e)}", "0%", model_name

    theme = gr.themes.Soft()
    
    with gr.Blocks(title="Bangla Fake Review Detector", theme=theme) as demo:
        gr.Markdown(
            """
            # 🔍 Bangla Fake Review Detector (বাংলা ফেক রিভিউ ডিটেক্টর)
            Select a deep learning model (**LSTM+Attention** or **RNN**), type or paste a Bangla review, and get an authentic vs fake prediction with confidence score.
            """
        )
        with gr.Row():
            with gr.Column(scale=2):
                model_dropdown = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value="LSTM",
                    label="Select Model / মডেল নির্বাচন করুন",
                    info="LSTM with Attention generally achieves higher precision."
                )
                review_input = gr.Textbox(
                    lines=5,
                    placeholder="এখানে একটি বাংলা রিভিউ লিখুন / Paste a Bangla review here...",
                    label="Bangla Review Text / বাংলা রিভিউ টেক্সট"
                )
                with gr.Row():
                    clear_btn = gr.Button("Clear / পরিষ্কার করুন", variant="secondary")
                    submit_btn = gr.Button("Predict / ডিটেক্ট করুন", variant="primary")
            
            with gr.Column(scale=2):
                verdict_output = gr.Textbox(label="Verdict / ফলাফল", interactive=False)
                confidence_output = gr.Textbox(label="Confidence Score / আত্মবিশ্বাসের মাত্রা", interactive=False)
                model_used_output = gr.Textbox(label="Model Used / ব্যবহৃত মডেল", interactive=False)
                
                gr.Examples(
                    examples=[
                        ["LSTM", "পণ্যটি অত্যন্ত চমৎকার এবং কোয়ালিটি খুব ভালো। দ্রুত ডেলিভারি পেয়েছি।"],
                        ["RNN", "ফালতু প্রোডাক্ট কেউ কিনবেন না, টাকা নষ্ট একদম বাজে সার্ভিস।"],
                        ["LSTM", "অসাধারণ একটি মোবাইল phone, ক্যামেরা এবং ব্যাটারি ব্যাকআপ দারুণ!"],
                    ],
                    inputs=[model_dropdown, review_input],
                    label="Sample Reviews / উদাহরণ"
                )
        
        submit_btn.click(
            fn=gradio_predict,
            inputs=[model_dropdown, review_input],
            outputs=[verdict_output, confidence_output, model_used_output]
        )
        clear_btn.click(
            fn=lambda: ("", "", "", ""),
            outputs=[review_input, verdict_output, confidence_output, model_used_output]
        )

    demo.launch(share=share)


# ---------------------------------------------------------------------------
# CLI mode
# ---------------------------------------------------------------------------

def run_cli():
    print("=" * 60)
    print(" Bangla Fake Review Detector -- CLI mode")
    print("=" * 60)
    print("Type 'quit' or 'exit' at any prompt to stop.\n")

    while True:
        try:
            model_name = input("Model [RNN/LSTM] (default LSTM): ").strip().upper() or "LSTM"
        except (KeyboardInterrupt, EOFError):
            break
        if model_name in ("QUIT", "EXIT"):
            break
        if model_name not in MODELS:
            print(f"  Unknown model '{model_name}'. Choose RNN or LSTM.\n")
            continue

        try:
            text = input("Review text: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if text.upper() in ("QUIT", "EXIT"):
            break
        if not text:
            print("  Enter a review first.\n")
            continue

        try:
            verdict, confidence = predict(model_name, text)
        except ModelLoadError as e:
            print(f"  [Model error] {e}\n")
            continue
        except ValueError as e:
            print(f"  [Input error] {e}\n")
            continue

        icon = "🚨" if verdict == "FAKE" else "✅"
        print(f"  {icon} {verdict}  (confidence: {confidence:.1%})  -- model: {model_name}\n")


# ---------------------------------------------------------------------------
# GUI mode (Tkinter)
# ---------------------------------------------------------------------------

def run_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox

    root = tk.Tk()
    root.title("Bangla Fake Review Detector")
    root.geometry("640x520")
    root.minsize(560, 480)

    PAD = 14
    FAKE_COLOR = "#c0392b"
    AUTH_COLOR = "#1e8449"
    NEUTRAL_BG = "#f4f4f4"

    main = ttk.Frame(root, padding=PAD)
    main.pack(fill="both", expand=True)

    ttk.Label(main, text="Bangla Fake Review Detector", font=("Segoe UI", 16, "bold")).pack(anchor="w")
    ttk.Label(main, text="Pick a model, paste a review, and get a verdict.",
              font=("Segoe UI", 10)).pack(anchor="w", pady=(0, PAD))

    # --- Model selector ---
    row1 = ttk.Frame(main)
    row1.pack(fill="x", pady=(0, 10))
    ttk.Label(row1, text="Model:", font=("Segoe UI", 10, "bold")).pack(side="left")
    model_var = tk.StringVar(value="LSTM")
    model_combo = ttk.Combobox(row1, textvariable=model_var, values=list(MODELS.keys()),
                                state="readonly", width=12)
    model_combo.pack(side="left", padx=(8, 0))

    status_var = tk.StringVar(value="")
    ttk.Label(row1, textvariable=status_var, foreground="#666").pack(side="left", padx=(12, 0))

    # --- Review input ---
    ttk.Label(main, text="Review text:", font=("Segoe UI", 10, "bold")).pack(anchor="w")
    text_frame = ttk.Frame(main)
    text_frame.pack(fill="both", expand=True, pady=(4, 10))

    review_box = tk.Text(text_frame, height=8, wrap="word", font=("Segoe UI", 11))
    review_box.pack(side="left", fill="both", expand=True)
    scroll = ttk.Scrollbar(text_frame, command=review_box.yview)
    scroll.pack(side="right", fill="y")
    review_box.configure(yscrollcommand=scroll.set)
    review_box.insert("1.0", "এখানে একটি বাংলা রিভিউ লিখুন / paste a Bangla review here...")
    review_box.bind("<FocusIn>", lambda e: (
        review_box.delete("1.0", "end")
        if review_box.get("1.0", "end").strip().startswith("এখানে একটি বাংলা রিভিউ")
        else None
    ))

    # --- Buttons ---
    btn_row = ttk.Frame(main)
    btn_row.pack(fill="x", pady=(0, 10))

    def clear_all():
        review_box.delete("1.0", "end")
        result_var.set("")
        result_frame.configure(style="Neutral.TFrame")

    def on_predict():
        text = review_box.get("1.0", "end").strip()
        if not text or text.startswith("এখানে একটি বাংলা রিভিউ"):
            messagebox.showinfo("Input needed", "Please enter a review first.")
            return

        predict_btn.configure(state="disabled")
        status_var.set("Loading model / predicting...")
        root.update_idletasks()
        try:
            verdict, confidence = predict(model_var.get(), text)
        except ModelLoadError as e:
            status_var.set("")
            messagebox.showerror("Model error", str(e))
            predict_btn.configure(state="normal")
            return
        except ValueError as e:
            status_var.set("")
            messagebox.showwarning("Input error", str(e))
            predict_btn.configure(state="normal")
            return
        except Exception as e:
            status_var.set("")
            messagebox.showerror("Unexpected error", str(e))
            predict_btn.configure(state="normal")
            return

        status_var.set("")
        predict_btn.configure(state="normal")
        icon = "🚨 FAKE" if verdict == "FAKE" else "✅ AUTHENTIC"
        color = FAKE_COLOR if verdict == "FAKE" else AUTH_COLOR
        result_var.set(f"{icon}   (confidence: {confidence:.1%})   -- model: {model_var.get()}")
        result_label.configure(foreground=color)

    predict_btn = ttk.Button(btn_row, text="Predict", command=on_predict)
    predict_btn.pack(side="left")
    ttk.Button(btn_row, text="Clear", command=clear_all).pack(side="left", padx=(8, 0))

    # --- Result ---
    style = ttk.Style()
    style.configure("Neutral.TFrame", background=NEUTRAL_BG)
    result_frame = ttk.Frame(main, padding=PAD, style="Neutral.TFrame")
    result_frame.pack(fill="x")
    result_var = tk.StringVar(value="")
    result_label = ttk.Label(result_frame, textvariable=result_var, font=("Segoe UI", 13, "bold"),
                              background=NEUTRAL_BG)
    result_label.pack(anchor="w")

    root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bangla Fake Review Detector -- local/Colab interface")
    parser.add_argument("--cli", action="store_true", help="Run in terminal instead of opening a GUI window")
    parser.add_argument("--gradio", action="store_true", help="Run Gradio web application (recommended for Colab)")
    parser.add_argument("--share", action="store_true", help="Create a public shareable Gradio link")
    args = parser.parse_args()

    in_colab = 'google.colab' in sys.modules or os.environ.get("COLAB_GPU") is not None

    if args.cli:
        run_cli()
    elif args.gradio or in_colab:
        run_gradio(share=args.share or in_colab)
    else:
        try:
            run_gui()
        except Exception as e:
            print(f"GUI launch failed ({e}) -- falling back to Gradio / CLI mode.\n")
            try:
                run_gradio(share=args.share)
            except Exception:
                run_cli()
