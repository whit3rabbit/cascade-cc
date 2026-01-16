import torch
import torch.nn as nn
import json
import sys
import os
import hashlib
import argparse
from constants import NODE_TYPES, TYPE_TO_ID, MAX_NODES, MAX_LITERALS

class CodeFingerprinter(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Permutation-invariant literal channel: Process each hash independently then pool
        self.literal_fc = nn.Linear(1, 16) 
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2 + 16, 64) 

    def forward(self, x, literal_vectors=None):
        lengths = (x != 0).sum(dim=1).cpu()
        embeds = self.embedding(x)
        
        packed_embeds = nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (hn, cn) = self.lstm(packed_embeds)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        
        mask = (x != 0).float().unsqueeze(-1)
        masked_out = lstm_out * mask
        sum_out = torch.sum(masked_out, dim=1)
        count = torch.clamp(torch.sum(mask, dim=1), min=1e-9)
        avg_pool = sum_out / count
        
        # Literal Channel: Order-independent Pooling
        if literal_vectors is not None:
            # literal_vectors: (batch, 32)
            # mask out padding (-1.0)
            lit_mask = (literal_vectors != -1.0).float().unsqueeze(-1)
            # Process each literal: (batch, 32, 1) -> (batch, 32, 16)
            lit_embeds = torch.relu(self.literal_fc(literal_vectors.unsqueeze(-1)))
            # Average pool over literals
            masked_lits = lit_embeds * lit_mask
            lit_feat = torch.sum(masked_lits, dim=1) / torch.clamp(torch.sum(lit_mask, dim=1), min=1e-9)
            combined = torch.cat([avg_pool, lit_feat], dim=1)
        else:
            lit_feat = torch.zeros(x.size(0), 16).to(x.device)
            combined = torch.cat([avg_pool, lit_feat], dim=1)
            
        return torch.nn.functional.normalize(self.fc(combined), p=2, dim=1)

def get_literal_hash(value):
    """Generate a numeric hash for a literal value (String/Number)"""
    if value is None: return -1.0
    s = str(value)
    # Range 0.0 to 1.0 (float)
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 10000000 / 10000000.0

def flatten_ast(node, sequence, symbols, literals, stats, path="Root"):
    """Convert AST tree into a linear sequence of type IDs (DFS) and extract symbols with relative context keys"""
    stats["total_nodes"] += 1
    if isinstance(node, list):
        for i, item in enumerate(node):
            flatten_ast(item, sequence, symbols, literals, stats, f"{path}[{i}]")
        return

    node_type = node.get("type", "UNKNOWN")
    
    # AST Skeletonization: Skip redundant/noisy nodes
    if node_type in ["EmptyStatement", "DebuggerStatement", "CommentLine", "CommentBlock"]:
        return

    # Literal Extraction
    if "valHash" in node:
        val_hash_int = int(node["valHash"], 16)
        literals.append((val_hash_int % 10000000) / 10000000.0)
    
    # Built-in detection
    if node_type == "CallExpression" and "call" in node:
        call_name = node["call"]
        if call_name == "require": node_type = "Builtin_require"
        elif call_name == "defineProperty": node_type = "Builtin_defineProperty"
    elif node_type == "Identifier" and "name" in node:
        name = node["name"]
        if name == "exports": node_type = "Builtin_exports"
        elif name == "module": node_type = "Builtin_module"

    type_id = TYPE_TO_ID.get(node_type, 1)
    if type_id == 1: stats["unknown_nodes"] += 1
    sequence.append(type_id)
    
    # Relative Context Keys: Reset path at major logic boundaries (Functions, Blocks)
    # This prevents absolute path shifts from breaking alignment.
    if node_type in ["FunctionDeclaration", "FunctionExpression", "ArrowFunctionExpression", "BlockStatement", "ClassBody"]:
        path = node_type # Reset context

    if node_type == "Identifier" and "name" in node:
        symbols.append({
            "name": node["name"],
            "key": path
        })
        
    for child in node.get("children", []):
        slot = child.get("slot", "child")
        flatten_ast(child, sequence, symbols, literals, stats, f"{path}/{slot}")

def run_vectorization(version_path, force=False):
    torch.manual_seed(42)
    current_vocab_size = len(NODE_TYPES) + 5
    model = CodeFingerprinter(vocab_size=current_vocab_size)
    model_path = os.path.join(os.path.dirname(__file__), "model.pth")
    
    if os.path.exists(model_path):
        print(f"[*] Loading pre-trained weights from {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            checkpoint_vocab_size = checkpoint.get('embedding.weight', torch.zeros(0)).shape[0]
            
            if checkpoint_vocab_size != current_vocab_size:
                print(f"[!] Warning: Vocabulary mismatch! Checkpoint: {checkpoint_vocab_size}, Current: {current_vocab_size}")
                if force:
                    print("[!] --force used. Partially loading weights...")
                    state_dict = model.state_dict()
                    for name, param in checkpoint.items():
                        if name in state_dict:
                            if param.shape == state_dict[name].shape:
                                state_dict[name].copy_(param)
                            elif name == 'embedding.weight':
                                min_size = min(checkpoint_vocab_size, current_vocab_size)
                                state_dict[name][:min_size].copy_(param[:min_size])
                                print(f"    [+] Resized embedding weights (mapped {min_size} types)")
                            else:
                                print(f"    [!] Skipping layer {name} due to shape mismatch: {param.shape} vs {state_dict[name].shape}")
                    model.load_state_dict(state_dict)
                else:
                    print("[!] Error: Model vocabulary mismatch. Use --force to load matching layers anyway.")
                    sys.exit(1)
            else:
                model.load_state_dict(checkpoint)
                print("[+] Successfully loaded weights.")
        except Exception as e:
            print(f"[!] Error loading weights: {e}")
            sys.exit(1)
    else:
        print("[!] Warning: No pre-trained weights found. Using seeded random weights.")
    
    model.eval()

    ast_path = os.path.join(version_path, "metadata", "simplified_asts.json")
    if not os.path.exists(ast_path):
        print(f"Error: Could not find simplified ASTs at {ast_path}", file=sys.stderr)
        sys.exit(1)

    with open(ast_path, 'r') as f:
        chunks = json.load(f)

    results = []
    
    stats = {"total_nodes": 0, "unknown_nodes": 0}
    with torch.no_grad():
        for chunk_name, ast_root in chunks.items():
            sequence = []
            symbols = []
            literals = []
            flatten_ast(ast_root, sequence, symbols, literals, stats)
            
            if not sequence: continue
            
            # Truncate or pad to fixed length
            seq_tensor = torch.tensor(sequence[:MAX_NODES]).unsqueeze(0)
            if seq_tensor.size(1) < MAX_NODES:
                seq_tensor = torch.nn.functional.pad(seq_tensor, (0, MAX_NODES - seq_tensor.size(1)))
            
            # Literal Hashing Channel (Pad with -1.0)
            lit_tensor = torch.tensor(literals[:MAX_LITERALS]).unsqueeze(0)
            if lit_tensor.size(1) < MAX_LITERALS:
                lit_tensor = torch.nn.functional.pad(lit_tensor, (0, MAX_LITERALS - lit_tensor.size(1)), value=-1.0)
            
            try:
                fingerprint = model(seq_tensor, lit_tensor).squeeze().tolist()
            except Exception as e:
                print(f"[!] Warning: Model execution failed for {chunk_name}: {e}")
                # Log first 5 node types in the offending sequence for debugging
                debug_types = [NODE_TYPES[id] if id < len(NODE_TYPES) else f"OUT_OF_BOUNDS({id})" for id in sequence[:5]]
                print(f"    Sample sequence types: {debug_types}")
                continue
            results.append({
                "name": chunk_name,
                "vector": fingerprint,
                "symbols": symbols
            })

    # Vocabulary health check
    if stats["total_nodes"] > 0:
        unknown_pct = (stats["unknown_nodes"] / stats["total_nodes"]) * 100
        print(f"[*] Vocabulary Health Check: {unknown_pct:.2f}% UNKNOWN nodes")

    output_path = os.path.join(version_path, "metadata", "logic_db.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[+] logic_db.json saved ({len(results)} chunks)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorize simplified ASTs using the brain model")
    parser.add_argument("version_path", help="Path to the version directory (e.g., claude-analysis/v3.5)")
    parser.add_argument("--force", action="store_true", help="Force loading weights even if vocabulary size mismatches")
    
    args = parser.parse_args()
    run_vectorization(args.version_path, force=args.force)
