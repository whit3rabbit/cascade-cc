import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import os
import sys
import random
from torch.utils.data import Dataset, DataLoader
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

class TripletDataset(Dataset):
    def __init__(self, patterns):
        self.keys = list(patterns.keys())
        self.patterns = patterns
        self.max_nodes = 1024
        
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
            
        anchor_lit_t = torch.tensor(anchor_lits[:32])
        if anchor_lit_t.size(0) < 32:
            anchor_lit_t = F.pad(anchor_lit_t, (0, 32 - anchor_lit_t.size(0)), value=-1.0)
            
        # 2. Positive: Synthetically mangled
        pos_ast = generate_synthetic_obfuscation(anchor_ast)
        pos_seq = []
        pos_lits = []
        flatten_ast(pos_ast, pos_seq, [], pos_lits, {"total_nodes": 0, "unknown_nodes": 0})
        positive_t = torch.tensor(pos_seq[:self.max_nodes])
        if positive_t.size(0) < self.max_nodes:
            positive_t = F.pad(positive_t, (0, self.max_nodes - positive_t.size(0)))
            
        pos_lit_t = torch.tensor(pos_lits[:32])
        if pos_lit_t.size(0) < 32:
            pos_lit_t = F.pad(pos_lit_t, (0, 32 - pos_lit_t.size(0)), value=-1.0)
            
        return anchor_t, anchor_lit_t, positive_t, pos_lit_t, anchor_key

def train_brain(bootstrap_dir, epochs=10, batch_size=32):
    # Device discovery (CUDA -> MPS -> CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    print(f"[*] Starting 'Brain' Training on Bootstrap Data: {bootstrap_dir}")
    
    ast_files = [f for f in os.listdir(bootstrap_dir) if f.endswith('_gold_asts.json')]
    if not ast_files:
        print("[!] Error: No gold ASTs found. Run 'npm run bootstrap' first.")
        return

    patterns = {}
    for f_name in ast_files:
        with open(os.path.join(bootstrap_dir, f_name), 'r') as f:
            data = json.load(f)
            lib_prefix = f_name.replace('_gold_asts.json', '')
            for chunk_name, ast in data.items():
                patterns[f"{lib_prefix}_{chunk_name}"] = ast

    dataset = TripletDataset(patterns)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CodeFingerprinter(vocab_size=len(NODE_TYPES) + 5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.TripletMarginLoss(margin=0.5, p=2)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for anchors, a_lits, positives, p_lits, anchor_keys in dataloader:
            anchors, a_lits = anchors.to(device), a_lits.to(device)
            positives, p_lits = positives.to(device), p_lits.to(device)

            # 1. Get initial embeddings for mining
            with torch.no_grad():
                anchor_vecs = model(anchors, a_lits)
            
            # 2. Hard Negative Mining (Batch-level) with Isomorphism Check
            negatives = []
            neg_lits = []
            for i in range(len(anchors)):
                max_sim = -1
                hardest_neg_idx = -1
                
                anchor_struct_hash = dataset.structural_hashes[anchor_keys[i]]
                
                for j in range(len(anchors)):
                    if i == j: continue
                    
                    # Structural Isomorphism Check
                    neg_struct_hash = dataset.structural_hashes[anchor_keys[j]]
                    if anchor_struct_hash == neg_struct_hash:
                        continue # Skip identical structures to avoid empty function trap
                    
                    sim = torch.dot(anchor_vecs[i], anchor_vecs[j]).item()
                    if sim > max_sim:
                        max_sim = sim
                        hardest_neg_idx = j
                
                if hardest_neg_idx == -1:
                    # Fallback if no non-isomorphic negative found in batch
                    hardest_neg_idx = (i + 1) % len(anchors)
                    
                negatives.append(anchors[hardest_neg_idx])
                neg_lits.append(a_lits[hardest_neg_idx])
            
            negatives = torch.stack(negatives)
            neg_lits = torch.stack(neg_lits)

            # 3. Model Forward Pass
            a_vec = model(anchors, a_lits)
            p_vec = model(positives, p_lits)
            n_vec = model(negatives, neg_lits)

            # 4. Loss and Optimization
            loss = criterion(a_vec, p_vec, n_vec)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"    Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
        
    print("[*] Training complete.")

    # Save the trained brain
    model_path = os.path.join(os.path.dirname(__file__), "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[+] Brain successfully trained and saved to {model_path}")

if __name__ == "__main__":
    bootstrap_path = sys.argv[1] if len(sys.argv) > 1 else "./ml/bootstrap_data"
    train_brain(bootstrap_path)
