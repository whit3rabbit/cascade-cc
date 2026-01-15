import torch
import torch.nn as nn
import json
import sys
import os

from .constants import NODE_TYPES, TYPE_TO_ID

class CodeFingerprinter(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 64) # Final fingerprint size

    def forward(self, x):
        # x is a sequence of Node Type IDs
        embeds = self.embedding(x)
        _, (hidden, _) = self.lstm(embeds)
        # Combine bi-directional hidden states
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return torch.nn.functional.normalize(self.fc(hidden_cat), p=2, dim=1)

def flatten_ast(node, sequence, symbols, path="Program"):
    """Convert AST tree into a linear sequence of type IDs (DFS) and extract symbols with structural keys"""
    if isinstance(node, list):
        for i, item in enumerate(node):
            flatten_ast(item, sequence, symbols, f"{path}[{i}]")
        return

    node_type = node.get("type", "UNKNOWN")
    
    # Built-in detection
    if node_type == "CallExpression" and "call" in node:
        call_name = node["call"]
        if call_name == "require":
            node_type = "Builtin_require"
        elif call_name == "defineProperty":
            node_type = "Builtin_defineProperty"
    elif node_type == "Identifier" and "name" in node:
        name = node["name"]
        if name == "exports":
            node_type = "Builtin_exports"
        elif name == "module":
            node_type = "Builtin_module"

    sequence.append(TYPE_TO_ID.get(node_type, 0)) # 0 for UNKNOWN
    
    if node_type == "Identifier" and "name" in node:
        symbols.append({
            "name": node["name"],
            "key": path
        })
        
    for i, child in enumerate(node.get("children", [])):
        child_type = child.get("type", "Node")
        flatten_ast(child, sequence, symbols, f"{path}.{child_type}[{i}]")

def run_vectorization(version_path):
    # Set seed for deterministic fingerprints across runs
    torch.manual_seed(42)
    
    # Initialize Model with full vocab size
    model = CodeFingerprinter(vocab_size=len(NODE_TYPES))
    
    # Try to load pre-trained weights if they exist
    model_path = os.path.join(os.path.dirname(__file__), "model.pth")
    if os.path.exists(model_path):
        print(f"[*] Loading pre-trained weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print("[!] Warning: No pre-trained weights found. Using seeded random weights.")
    
    model.eval()

    # Load simplified ASTs from metadata
    ast_path = os.path.join(version_path, "metadata", "simplified_asts.json")
    if not os.path.exists(ast_path):
        print(f"Error: Could not find simplified ASTs at {ast_path}", file=sys.stderr)
        sys.exit(1)

    with open(ast_path, 'r') as f:
        chunks = json.load(f)

    results = []
    total_nodes = 0
    unknown_nodes = 0
    
    with torch.no_grad():
        for chunk_name, ast_root in chunks.items():
            sequence = []
            symbols = []
            flatten_ast(ast_root, sequence, symbols)
            
            if not sequence:
                continue
            
            total_nodes += len(sequence)
            unknown_nodes += sequence.count(0)

            # Truncate or pad to fixed length (e.g., 256 nodes)
            seq_tensor = torch.tensor(sequence[:256]).unsqueeze(0)
            if seq_tensor.size(1) < 256:
                seq_tensor = torch.nn.functional.pad(seq_tensor, (0, 256 - seq_tensor.size(1)))
            
            fingerprint = model(seq_tensor).squeeze().tolist()
            results.append({
                "name": chunk_name,
                "vector": fingerprint,
                "symbols": symbols # Exported for alignment: list of {name, key}
            })

    # Vocabulary health check
    if total_nodes > 0:
        unknown_pct = (unknown_nodes / total_nodes) * 100
        print(f"[*] Vocabulary Health Check: {unknown_pct:.2f}% UNKNOWN nodes")
        if unknown_pct > 5.0:
            print(f"[!] WARNING: High percentage of UNKNOWN nodes ({unknown_pct:.2f}%). ML performance may be degraded.")

    output_path = os.path.join(version_path, "metadata", "logic_db.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[+] logic_db.json saved (Deterministic weights, {len(results)} chunks)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 vectorize.py <version_path>")
        sys.exit(1)
    run_vectorization(sys.argv[1])
