import torch
import torch.nn as nn
import json
import sys
import os
import hashlib
import argparse
from constants import NODE_TYPES, TYPE_TO_ID, MAX_NODES, MAX_LITERALS

def get_auto_max_nodes(device):
    """Infers optimal context window based on device memory."""
    if device.type == "cuda":
        try:
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_vram > 15: return 2048 # A100 / H100 / 3090+
            if total_vram > 7: return 1024  # T4 / 3080 / 4070
            return 512
        except:
            return 512
    elif device.type == "mps":
        return 256 # Conservative for shared memory architectures
    return 512 # Safe default for CPU

from encoder import TransformerCodeEncoder

class CodeFingerprinter(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=32, hidden_dim=64, max_nodes=MAX_NODES):
        super().__init__()
        self.transformer_encoder = TransformerCodeEncoder(vocab_size, embed_dim=embed_dim, max_nodes=max_nodes)
        
        # Permutation-invariant literal channel: Process each hash independently then pool
        self.literal_fc = nn.Linear(1, 16) 
        
        # Combined projection: Transformer out (64) + Literal out (16)
        self.fc = nn.Linear(64 + 16, 64) 

    def forward(self, x, literal_vectors=None):
        # x: (batch, seq_len)
        # Transformer output is already normalized and 64-dim from TransformerCodeEncoder
        struct_feat = self.transformer_encoder(x)
        
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
            combined = torch.cat([struct_feat, lit_feat], dim=1)
        else:
            lit_feat = torch.zeros(x.size(0), 16).to(x.device)
            combined = torch.cat([struct_feat, lit_feat], dim=1)
            
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

def run_vectorization(version_path, force=False, device_name="cuda", max_nodes_override=None):
    torch.manual_seed(42)
    
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
    
    # Dynamic Context Window Adjustment
    effective_max_nodes = max_nodes_override if max_nodes_override else get_auto_max_nodes(device)
    source = "Manual override" if max_nodes_override else f"Auto-detected for {device.type}"
    print(f"[*] Context Window: {effective_max_nodes} nodes ({source})")

    node_type_count = len(NODE_TYPES)
    if node_type_count < 2:
        print("[!] Error: NODE_TYPES is unexpectedly small; run 'npm run sync-vocab' before vectorizing.")
        sys.exit(1)
    current_vocab_size = node_type_count + 5
    model = CodeFingerprinter(vocab_size=current_vocab_size, max_nodes=effective_max_nodes).to(device)
    model_path = os.path.join(os.path.dirname(__file__), "model.pth")
    
    if os.path.exists(model_path):
        print(f"[*] Loading pre-trained weights from {model_path} onto {device}")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if 'transformer_encoder.embedding.weight' in checkpoint:
                embedding_key = 'transformer_encoder.embedding.weight'
            elif 'embedding.weight' in checkpoint:
                embedding_key = 'embedding.weight'
            else:
                print("[!] Error: Checkpoint missing embedding weights. Re-train or re-export model.pth.")
                sys.exit(1)
            checkpoint_vocab_size = checkpoint[embedding_key].shape[0]
            
            # Robust loading: Check shapes for all parameters
            state_dict = model.state_dict()
            loaded_count = 0
            resized_count = 0
            
            for name, param in checkpoint.items():
                if name in state_dict:
                    if param.shape == state_dict[name].shape:
                        state_dict[name].copy_(param)
                        loaded_count += 1
                    elif 'embedding.weight' in name:
                        min_size = min(param.shape[0], state_dict[name].shape[0])
                        state_dict[name][:min_size].copy_(param[:min_size])
                        resized_count += 1
                    elif 'pos_encoder' in name:
                        min_seq = min(param.shape[1], state_dict[name].shape[1])
                        state_dict[name][:, :min_seq, :].copy_(param[:, :min_seq, :])
                        resized_count += 1
            
            model.load_state_dict(state_dict)
            print(f"[+] Loaded weights: {loaded_count} exact, {resized_count} resized.")
        except Exception as e:
            print(f"[!] Error loading weights: {e}")
            if not force: sys.exit(1)
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
                # Move tensors to device
                seq_tensor = seq_tensor.to(device)
                lit_tensor = lit_tensor.to(device)
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
    parser.add_argument("--max_nodes", type=int, default=0, help="Override context window size (0 for auto)")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda, mps, cpu, or auto")
    args = parser.parse_args()

    m_nodes = args.max_nodes if args.max_nodes > 0 else None
    run_vectorization(args.version_path, force=args.force, device_name=args.device, max_nodes_override=m_nodes)
