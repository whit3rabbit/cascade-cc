import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import os
import sys
import random
from vectorize import CodeFingerprinter, NODE_TYPES, flatten_ast

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

def to_tensor(ast):
    seq = []
    syms = []
    flatten_ast(ast, seq, syms)
    t = torch.tensor(seq[:256]).unsqueeze(0)
    if t.size(1) < 256:
        t = F.pad(t, (0, 256 - t.size(1)))
    return t

def train_brain(bootstrap_dir, epochs=10):
    print(f"[*] Starting 'Brain' Training on Bootstrap Data: {bootstrap_dir}")
    
    ast_files = [f for f in os.listdir(bootstrap_dir) if f.endswith('_gold_asts.json')]
    if not ast_files:
        print("[!] Error: No gold ASTs found. Run 'npm run bootstrap' first.")
        return

    # Load all unique structural patterns
    patterns = {}
    for f_name in ast_files:
        with open(os.path.join(bootstrap_dir, f_name), 'r') as f:
            data = json.load(f)
            lib_prefix = f_name.replace('_gold_asts.json', '')
            for chunk_name, ast in data.items():
                patterns[f"{lib_prefix}_{chunk_name}"] = ast

    pattern_keys = list(patterns.keys())
    print(f"[*] Loaded {len(pattern_keys)} logic patterns from {len(ast_files)} libraries.")

    model = CodeFingerprinter(vocab_size=len(NODE_TYPES))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(pattern_keys)
        
        # We use a simple Triplet Strategy for the Cold Start
        for i in range(len(pattern_keys)):
            anchor_key = pattern_keys[i]
            
            # 1. Anchor: Clean AST from library
            anchor_t = to_tensor(patterns[anchor_key])
            
            # 2. Positive: Synthetically mangled version of the same AST
            pos_ast = generate_synthetic_obfuscation(patterns[anchor_key])
            positive_t = to_tensor(pos_ast)
            
            # 3. Negative: A completely different AST from another library
            neg_key = pattern_keys[(i + 1) % len(pattern_keys)]
            negative_t = to_tensor(patterns[neg_key])

            # Get Embeddings
            a_vec = model(anchor_t)
            p_vec = model(positive_t)
            n_vec = model(negative_t)

            # Triplet Loss: Minimize distance(a,p), Maximize distance(a,n)
            pos_dist = (a_vec - p_vec).pow(2).sum(1)
            neg_dist = (a_vec - n_vec).pow(2).sum(1)
            loss = F.relu(pos_dist - neg_dist + 0.5).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"    Epoch {epoch+1}/{epochs} - Avg Loss: {total_loss/len(pattern_keys):.4f}")

    # Save the trained brain
    model_path = os.path.join(os.path.dirname(__file__), "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[+] Brain successfully trained and saved to {model_path}")

if __name__ == "__main__":
    bootstrap_path = sys.argv[1] if len(sys.argv) > 1 else "./ml/bootstrap_data"
    train_brain(bootstrap_path)
