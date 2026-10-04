import time
import json
import torch
import math
import tiktoken
from model import TinyLM
from data import get_dataloader

def generate_text(model, enc, prompt="The ", max_new_tokens=15, device="cpu"):
    """Periodically generates text from a fixed prompt to monitor model sanity."""
    model.eval()
    idx = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(idx)
            # Take the logits for the final token in the sequence
            logits = logits[:, -1, :] 
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # Greedily pick the most likely next word
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)
            # Append it to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Crop to block size if exceeding max length
            if idx.size(1) > model.max_seq_len:
                idx = idx[:, -model.max_seq_len:]
    
    out = enc.decode(idx[0].tolist())
    model.train()
    return out

@torch.no_grad()
def estimate_loss(model, val_loader, eval_steps=5, device="cpu"):
    """Quickly estimates validation loss over a few batches."""
    model.eval()
    losses = []
    for i, (X, Y) in enumerate(val_loader):
        if i >= eval_steps:
            break
        X, Y = X.to(device), Y.to(device)
        _, loss = model(X, Y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else 0.0

def main():
    # Setup device
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon MPS acceleration!")
    elif torch.cuda.is_available():
        device = "cuda"
    print(f"Executing on device: {device}")
    
    # === Acceptance Check constraints ===
    # "Overfit deliberately on a tiny subset. Training loss should fall sharply."
    batch_size = 4
    seq_length = 32
    max_steps = 250
    learning_rate = 3e-4
    eval_interval = 50
    
    # Get loaders
    train_loader = get_dataloader(split='train', batch_size=batch_size, seq_length=seq_length)
    val_loader = get_dataloader(split='val', batch_size=batch_size, seq_length=seq_length)
    
    # Init model & Optimizer (AdamW is standard for Transformers)
    model = TinyLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    enc = tiktoken.get_encoding("gpt2")
    log_file = open("training_log.jsonl", "w")
    
    model.train()
    start_time = time.time()
    
    # To artificially OVERFIT to pass the Acceptance Check, 
    # we literally yank one single batch out and force the model to memorize it.
    X_overfit, Y_overfit = next(iter(train_loader))
    X_overfit, Y_overfit = X_overfit.to(device), Y_overfit.to(device)
    
    for step in range(max_steps):
        t0 = time.time()
        
        # 1. Zero out previous gradients
        optimizer.zero_grad()
        # 2. Forward pass
        logits, loss = model(X_overfit, Y_overfit)
        # 3. Backward Pass
        loss.backward()
        # 4. Gradient Clipping (prevents explode)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 5. Step!
        optimizer.step()
        
        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = (batch_size * seq_length) / dt
        
        # Periodic Reporting
        if step % eval_interval == 0 or step == max_steps - 1:
            val_loss = estimate_loss(model, val_loader, eval_steps=5, device=device)
            elapsed = time.time() - start_time
            
            gen_text = generate_text(model, enc, prompt="\nFirst Citizen:\nBefore we proceed", device=device)
            
            # Progress logging to JSONL (Syllabus constraint)
            log_entry = {
                "step": step,
                "train_loss": loss.item(),
                "validation_loss": val_loss,
                "learning_rate": learning_rate,
                "tokens_per_second": tokens_per_sec,
                "elapsed_seconds": elapsed
            }
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()
            
            print(f"Step {step:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Tok/s: {tokens_per_sec:.0f}")
            print(f"Gen Output: {gen_text!r}\n")
    
    log_file.close()

if __name__ == "__main__":
    main()
