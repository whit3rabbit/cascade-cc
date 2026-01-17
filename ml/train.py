import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import os
import sys
import random
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
from vectorize import CodeFingerprinter, flatten_ast
from constants import NODE_TYPES, MAX_NODES, MAX_LITERALS

def generate_synthetic_obfuscation(ast_node):
    """
    Creates a 'mangled' version of the AST.
    Teaches the NN that names are junk, but structure is signal.
    """
    if isinstance(ast_node, list):
        return [generate_synthetic_obfuscation(n) for n in ast_node]
    if not isinstance(ast_node, dict):
        return ast_node
        
    new_node = ast_node.copy()
    if new_node.get("type") == "Identifier":
        # Simulate minification: rename variables to 1-2 random chars
        new_node["name"] = random.choice("abcdefghijklmnopqrstuvwxyz") + random.choice("0123456789")
        
    if "children" in new_node:
        new_node["children"] = [generate_synthetic_obfuscation(c) for c in new_node["children"]]
    return new_node

class TripletDataset(Dataset):
    def __init__(self, patterns):
        self.keys = list(patterns.keys())
        self.patterns = patterns
        self.max_nodes = MAX_NODES
        
        # Pre-calculate structural hashes for isomorphism check
        self.structural_hashes = {}
        print(f"[*] Pre-calculating structural hashes for {len(self.keys)} chunks...")
        for i, k in enumerate(self.keys):
            if i % 100 == 0: print(f"    - {i}/{len(self.keys)} hashed...")
            seq = []
            flatten_ast(self.patterns[k], seq, [], [], {"total_nodes": 0, "unknown_nodes": 0})
            self.structural_hashes[k] = hash(tuple(seq))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        anchor_key = self.keys[idx]
        anchor_ast = self.patterns[anchor_key]
        
        # 1. Anchor
        anchor_seq = []
        anchor_lits = []
        flatten_ast(anchor_ast, anchor_seq, [], anchor_lits, {"total_nodes": 0, "unknown_nodes": 0})
        anchor_t = torch.tensor(anchor_seq[:self.max_nodes])
        if anchor_t.size(0) < self.max_nodes:
            anchor_t = F.pad(anchor_t, (0, self.max_nodes - anchor_t.size(0)))
            
        anchor_lit_t = torch.tensor(anchor_lits[:MAX_LITERALS])
        if anchor_lit_t.size(0) < MAX_LITERALS:
            anchor_lit_t = F.pad(anchor_lit_t, (0, MAX_LITERALS - anchor_lit_t.size(0)), value=-1.0)
            
        # 2. Positive: Synthetically mangled
        pos_ast = generate_synthetic_obfuscation(anchor_ast)
        pos_seq = []
        pos_lits = []
        flatten_ast(pos_ast, pos_seq, [], pos_lits, {"total_nodes": 0, "unknown_nodes": 0})
        positive_t = torch.tensor(pos_seq[:self.max_nodes])
        if positive_t.size(0) < self.max_nodes:
            positive_t = F.pad(positive_t, (0, self.max_nodes - positive_t.size(0)))
            
        pos_lit_t = torch.tensor(pos_lits[:MAX_LITERALS])
        if pos_lit_t.size(0) < MAX_LITERALS:
            pos_lit_t = F.pad(pos_lit_t, (0, MAX_LITERALS - pos_lit_t.size(0)), value=-1.0)
            
        return anchor_t, anchor_lit_t, positive_t, pos_lit_t, anchor_key

def evaluate_model(model, dataloader, device, dataset):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for anchors, a_lits, positives, p_lits, anchor_keys in dataloader:
            anchors, a_lits = anchors.to(device), a_lits.to(device)
            positives, p_lits = positives.to(device), p_lits.to(device)
            
            a_vec = model(anchors, a_lits)
            p_vec = model(positives, p_lits)
            
            # For each anchor, pick a hard negative from the batch
            for i in range(len(anchors)):
                anchor_struct_hash = dataset.structural_hashes[anchor_keys[i]]
                
                # Simple negative: pick any other non-isomorphic item in batch
                neg_idx = -1
                for j in range(len(anchors)):
                    if i == j: continue
                    if dataset.structural_hashes[anchor_keys[j]] != anchor_struct_hash:
                        neg_idx = j
                        break
                
                if neg_idx == -1: continue # Skip if no suitable negative in batch
                
                n_vec = model(anchors[neg_idx].unsqueeze(0), a_lits[neg_idx].unsqueeze(0))
                
                d_pos = torch.norm(a_vec[i] - p_vec[i], p=2)
                d_neg = torch.norm(a_vec[i] - n_vec, p=2)
                
                if d_pos < d_neg:
                    correct += 1
                total += 1
    
    return (correct / total) * 100 if total > 0 else 0

def train_brain(bootstrap_dir, epochs=5, batch_size=16, force=False, lr=0.001, margin=0.2, embed_dim=32, hidden_dim=64, is_sweep=False, device_name="cuda"):
    # Device discovery (CUDA -> MPS -> CPU)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    elif device_name == "cuda" and not torch.cuda.is_available():
        print("[!] Warning: CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    elif device_name == "mps" and not torch.backends.mps.is_available():
        print("[!] Warning: MPS requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    
    if not is_sweep: print(f"[*] Training on device: {device}")

    current_vocab_size = len(NODE_TYPES) + 5
    if not is_sweep: print(f"[*] Current Vocabulary: {len(NODE_TYPES)} types (Total with specials: {current_vocab_size})")

    if not is_sweep: print(f"[*] Starting 'Brain' Training on Bootstrap Data: {bootstrap_dir}")
    
    ast_files = [f for f in os.listdir(bootstrap_dir) if f.endswith('_gold_asts.json')]
    if not ast_files:
        print("[!] Error: No gold ASTs found. Run 'npm run bootstrap' first.")
        return 0, None

    patterns = {}
    for f_name in ast_files:
        with open(os.path.join(bootstrap_dir, f_name), 'r') as f:
            data = json.load(f)
            lib_prefix = f_name.replace('_gold_asts.json', '')
            for chunk_name, ast in data.items():
                patterns[f"{lib_prefix}_{chunk_name}"] = ast

    full_dataset = TripletDataset(patterns)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    use_cuda = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_cuda)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=use_cuda)

    model = CodeFingerprinter(vocab_size=current_vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    
    # Robust Checkpoint Loading (Skip if sweep unless specified)
    if not is_sweep:
        model_path = os.path.join(os.path.dirname(__file__), "model.pth")
        if os.path.exists(model_path):
            print(f"[*] Found existing model at {model_path}. Attempting to load...")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                checkpoint_vocab_size = checkpoint.get('transformer_encoder.embedding.weight', checkpoint.get('embedding.weight', torch.zeros(0))).shape[0]
                if checkpoint_vocab_size == current_vocab_size:
                    model.load_state_dict(checkpoint)
                    print("[+] Successfully loaded checkpoint weights.")
                elif force:
                    print("[!] Vocabulary mismatch, but --force used. Partially loading...")
                    state_dict = model.state_dict()
                    for name, param in checkpoint.items():
                        if name in state_dict:
                            if param.shape == state_dict[name].shape:
                                state_dict[name].copy_(param)
                            elif 'embedding.weight' in name:
                                # Handle partial embedding load if sizes differ
                                min_size = min(checkpoint_vocab_size, current_vocab_size)
                                state_dict[name][:min_size].copy_(param[:min_size])
                                print(f"    [+] Resized {name} (mapped {min_size} types)")
                    model.load_state_dict(state_dict)
                else:
                    print(f"[!] Vocabulary mismatch (Checkpoint: {checkpoint_vocab_size}, Current: {current_vocab_size}). Use --force to load.")
            except Exception as e:
                print(f"[!] Error loading checkpoint: {e}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for anchors, a_lits, positives, p_lits, anchor_keys in train_loader:
            anchors, a_lits = anchors.to(device), a_lits.to(device)
            positives, p_lits = positives.to(device), p_lits.to(device)

            with torch.no_grad():
                anchor_vecs = model(anchors, a_lits)
            
            negatives = []
            neg_lits = []
            for i in range(len(anchors)):
                max_sim = -1.0
                hardest_neg_idx = -1
                anchor_struct_hash = full_dataset.structural_hashes[anchor_keys[i]]
                
                for j in range(len(anchors)):
                    if i == j: continue
                    if full_dataset.structural_hashes[anchor_keys[j]] == anchor_struct_hash: continue
                    
                    sim = torch.dot(anchor_vecs[i], anchor_vecs[j]).item()
                    if sim > max_sim:
                        max_sim = sim
                        hardest_neg_idx = j
                
                if hardest_neg_idx == -1:
                    hardest_neg_idx = (i + 1) % len(anchors)
                    
                negatives.append(anchors[hardest_neg_idx])
                neg_lits.append(a_lits[hardest_neg_idx])
            
            negatives = torch.stack(negatives).to(device)
            neg_lits = torch.stack(neg_lits).to(device)

            a_vec = model(anchors, a_lits)
            p_vec = model(positives, p_lits)
            n_vec = model(negatives, neg_lits)

            loss = criterion(a_vec, p_vec, n_vec)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate_model(model, val_loader, device, full_dataset)
        if not is_sweep:
            print(f"    Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f} - Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if not is_sweep:
                torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "model.pth"))

    return best_val_acc, model.state_dict()

def run_sweep(bootstrap_dir, epochs=5):
    print("[*] Starting Hyperparameter Sweep...")
    results = []
    
    margins = [0.2, 0.3, 0.5]
    lrs = [0.001, 0.0005]
    embed_dims = [32, 64]
    
    best_overall_acc = 0
    best_params = None
    best_state = None

    for m in margins:
        for lr in lrs:
            for ed in embed_dims:
                print(f"    - Testing Margin: {m}, LR: {lr}, Embed: {ed}...")
                acc, state = train_brain(bootstrap_dir, epochs=epochs, is_sweep=True, margin=m, lr=lr, embed_dim=ed, device_name="auto")
                print(f"      Result: {acc:.2f}%")
                results.append({"margin": m, "lr": lr, "embed": ed, "acc": acc})
                
                if acc > best_overall_acc:
                    best_overall_acc = acc
                    best_params = {"margin": m, "lr": lr, "embed": ed}
                    best_state = state

    print(f"\n[+] Sweep Complete! Best Accuracy: {best_overall_acc:.2f}%")
    print(f"[+] Best Params: {best_params}")
    
    if best_state:
        model_path = os.path.join(os.path.dirname(__file__), "model.pth")
        torch.save(best_state, model_path)
        print(f"[*] Best model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Code Fingerprinting Neural Network")
    parser.add_argument("bootstrap_dir", nargs="?", default="./ml/bootstrap_data", help="Directory containing gold ASTs")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--force", action="store_true", help="Force loading weights even if vocabulary size mismatches")
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda, mps, cpu, or auto")
    
    args = parser.parse_args()
    
    if args.sweep:
        run_sweep(args.bootstrap_dir, epochs=args.epochs)
    else:
        train_brain(args.bootstrap_dir, epochs=args.epochs, batch_size=args.batch_size, force=args.force, device_name=args.device)
