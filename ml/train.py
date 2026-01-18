import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import os
import sys
import random
import argparse
import warnings
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
from vectorize import CodeFingerprinter, flatten_ast, get_auto_max_nodes
from constants import NODE_TYPES, MAX_NODES, MAX_LITERALS
# removed get_auto_max_nodes definition (imported from vectorize)

# Suppress transformer nested tensor prototype warnings for cleaner sweep logs.
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

def generate_synthetic_obfuscation(ast_node):
    """
    Creates a 'mangled' version of the AST.
    Teaches the NN that names are junk, but structure is signal.
    Includes aggressive structural noise to prevent trivial 100% accuracy.
    """
    if isinstance(ast_node, list):
        # Statement Shuffling: If children don't seem like they have strict order (e.g. declarations)
        if len(ast_node) > 1 and random.random() < 0.15:
            new_list = ast_node.copy()
            # Pick two random indices and swap
            i, j = random.sample(range(len(new_list)), 2)
            new_list[i], new_list[j] = new_list[j], new_list[i]
            return [generate_synthetic_obfuscation(n) for n in new_list]
        return [generate_synthetic_obfuscation(n) for n in ast_node]
        
    if not isinstance(ast_node, dict):
        return ast_node
        
    new_node = ast_node.copy()
    node_type = new_node.get("type")

    # 1. Rename Identifiers (Minification)
    if node_type == "Identifier":
        new_node["name"] = random.choice("abcdefghijklmnopqrstuvwxyz") + random.choice("0123456789")
        
    # 2. Dead Code Injection: Randomly insert if(false){...}
    if node_type == "BlockStatement" and random.random() < 0.15:
        dead_node = {
            "type": "IfStatement",
            "children": [
                {"type": "BooleanLiteral", "value": False, "slot": "test"},
                {"type": "BlockStatement", "children": [], "slot": "consequent"}
            ]
        }
        if "children" in new_node:
            new_node["children"] = new_node["children"].copy()
            insert_pos = random.randint(0, len(new_node["children"]))
            new_node["children"].insert(insert_pos, dead_node)

    # 3. Constant Unfolding: Replace true with !0, etc.
    if node_type == "BooleanLiteral" and random.random() < 0.2:
        val = new_node.get("value")
        # Simulate !0 or !1 structure
        new_node = {
            "type": "UnaryExpression",
            "operator": "!",
            "children": [
                {"type": "NumericLiteral", "value": 0 if val else 1, "slot": "argument"}
            ]
        }
        return new_node

    # 4. IIFE Wrapping: Wrap a block or expression in (function(){...})()
    if node_type in ["BlockStatement", "CallExpression"] and random.random() < 0.05:
        # Simplified representation of an IIFE in our AST logic
        new_node = {
            "type": "CallExpression",
            "children": [
                {
                    "type": "FunctionExpression",
                    "children": [new_node],
                    "slot": "callee"
                }
            ]
        }

    # 5. Structural Noise: IfStatement Swapping (Existing)
    if node_type == "IfStatement" and random.random() < 0.2:
        children = new_node.get("children", [])
        test_idx, cons_idx, alt_idx = -1, -1, -1
        for i, c in enumerate(children):
            slot = c.get("slot")
            if slot == "test": test_idx = i
            elif slot == "consequent": cons_idx = i
            elif slot == "alternate": alt_idx = i
        
        if test_idx != -1 and cons_idx != -1 and alt_idx != -1:
            new_children = children.copy()
            new_children[cons_idx], new_children[alt_idx] = children[alt_idx], children[cons_idx]
            new_children[test_idx] = {
                "type": "UnaryExpression",
                "children": [children[test_idx]],
                "slot": "test"
            }
            new_node["children"] = new_children

    # 6. Commutative Swapping: BinaryExpression / LogicalExpression (Existing)
    if node_type in ["BinaryExpression", "LogicalExpression"] and random.random() < 0.2:
        children = new_node.get("children", [])
        left_idx, right_idx = -1, -1
        for i, c in enumerate(children):
            slot = c.get("slot")
            if slot == "left": left_idx = i
            elif slot == "right": right_idx = i
        
        if left_idx != -1 and right_idx != -1:
            new_children = children.copy()
            new_children[left_idx], new_children[right_idx] = children[right_idx], children[left_idx]
            new_node["children"] = new_children

    if "children" in new_node:
        new_node["children"] = [generate_synthetic_obfuscation(c) for c in new_node["children"]]
    return new_node

class TripletDataset(Dataset):
    def __init__(self, patterns, max_nodes=MAX_NODES):
        self.keys = list(patterns.keys())
        self.patterns = patterns
        self.max_nodes = max_nodes
        self.lib_to_keys = {}
        
        # Pre-calculate structural hashes for isomorphism check
        self.structural_hashes = {}
        print(f"[*] Pre-calculating structural hashes for {len(self.keys)} chunks...")
        for i, k in enumerate(self.keys):
            if i % 100 == 0: print(f"    - {i}/{len(self.keys)} hashed...")
            lib_key = k.split("_", 1)[0]
            self.lib_to_keys.setdefault(lib_key, []).append(k)
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
        
        # --- NUCLEAR OPTION: Node Type Masking (15%) ---
        for i in range(len(anchor_seq)):
            if random.random() < 0.15:
                anchor_seq[i] = 1 # UNKNOWN
                
        anchor_t = torch.tensor(anchor_seq[:self.max_nodes])
        if anchor_t.size(0) < self.max_nodes:
            anchor_t = F.pad(anchor_t, (0, self.max_nodes - anchor_t.size(0)))
            
        anchor_lit_t = torch.tensor(anchor_lits[:MAX_LITERALS])
        if anchor_lit_t.size(0) < MAX_LITERALS:
            anchor_lit_t = F.pad(anchor_lit_t, (0, MAX_LITERALS - anchor_lit_t.size(0)), value=-1.0)
            
        # --- NUCLEAR OPTION: Literal Dropout (50%) Phase 1: Determine ---
        dropout_active = random.random() < 0.5
        if dropout_active:
            anchor_lit_t = torch.full((MAX_LITERALS,), -1.0)

        # 2. Positive: usually a synthetic obfuscation of the anchor,
        # but occasionally (10%) a different chunk from the same library
        # to encourage clustering by vendor/library "family."
        pos_ast = None
        if random.random() < 0.10:
            anchor_lib = anchor_key.split("_", 1)[0]
            lib_keys = [k for k in self.lib_to_keys.get(anchor_lib, []) if k != anchor_key]
            if lib_keys:
                pos_key = random.choice(lib_keys)
                pos_ast = self.patterns[pos_key]
        if pos_ast is None:
            pos_ast = generate_synthetic_obfuscation(anchor_ast)
        pos_seq = []
        pos_lits = []
        flatten_ast(pos_ast, pos_seq, [], pos_lits, {"total_nodes": 0, "unknown_nodes": 0})
        
        # --- NUCLEAR OPTION: Node Type Masking (15%) ---
        for i in range(len(pos_seq)):
            if random.random() < 0.15:
                pos_seq[i] = 1 # UNKNOWN

        # Add 20% random "junk" nodes (PAD or UNKNOWN variant)
        if len(pos_seq) > 0:
            junk_count = int(len(pos_seq) * 0.2)
            for _ in range(junk_count):
                pos_seq.insert(random.randint(0, len(pos_seq)), random.choice([0, 1])) # 0=PAD, 1=UNKNOWN

        # --- NUCLEAR OPTION: Sequence Jittering (Random Cropping/Padding) ---
        if len(pos_seq) > 10 and random.random() < 0.3:
            # Crop up to 10% from start or end
            crop_size = int(len(pos_seq) * 0.1)
            if random.random() < 0.5:
                pos_seq = pos_seq[random.randint(0, crop_size):]
            else:
                pos_seq = pos_seq[:-random.randint(1, crop_size)]

        # Re-apply padding/truncation AFTER jittering to ensure consistent tensor size
        positive_t = torch.tensor(pos_seq[:self.max_nodes])
        if positive_t.size(0) < self.max_nodes:
            positive_t = F.pad(positive_t, (0, self.max_nodes - positive_t.size(0)))
            
        pos_lit_t = torch.tensor(pos_lits[:MAX_LITERALS])
        if pos_lit_t.size(0) < MAX_LITERALS:
            pos_lit_t = F.pad(pos_lit_t, (0, MAX_LITERALS - pos_lit_t.size(0)), value=-1.0)
        
        # --- NUCLEAR OPTION: Literal Dropout (50%) Phase 2: Align ---
        if dropout_active: 
            pos_lit_t = torch.full((MAX_LITERALS,), -1.0)
            
        return anchor_t, anchor_lit_t, positive_t, pos_lit_t, anchor_key

def evaluate_model(model, dataloader, device, dataset, mask_same_library=False):
    model.eval()
    correct = 0
    total = 0
    
    total_pos_dist = 0
    total_neg_dist = 0
    count = 0
    total_rr = 0.0
    rr_count = 0
    per_lib_stats = {}

    with torch.no_grad():
        for anchors, a_lits, positives, p_lits, anchor_keys in dataloader:
            anchors, a_lits = anchors.to(device), a_lits.to(device)
            positives, p_lits = positives.to(device), p_lits.to(device)
            
            a_vec = model(anchors, a_lits)
            p_vec = model(positives, p_lits)
            
            # Vectorized hard-negative selection for evaluation.
            # Similarity matrix uses cosine similarity because embeddings are L2-normalized.
            sims = torch.matmul(a_vec, a_vec.T)
            
            for i in range(len(anchors)):
                anchor_hash = dataset.structural_hashes[anchor_keys[i]]
                anchor_lib = anchor_keys[i].split("_", 1)[0] if mask_same_library else None
                positive_lib = anchor_lib
                
                # Mask out self and isomorphs; optionally mask same-library candidates
                # when evaluating across multiple libraries.
                batch_sims = sims[i].clone()
                batch_sims[i] = -2 # Lower than any possible cosine sim
                
                for j in range(len(anchors)):
                    candidate_key = anchor_keys[j]
                    candidate_lib = candidate_key.split("_", 1)[0] if mask_same_library else None
                    if dataset.structural_hashes[candidate_key] == anchor_hash:
                        batch_sims[j] = -2
                    elif mask_same_library and candidate_lib == anchor_lib:
                        batch_sims[j] = -2
                
                hardest_idx = torch.argmax(batch_sims)
                if batch_sims[hardest_idx] == -2:
                    continue # No valid negative in batch
                
                n_vec = a_vec[hardest_idx]
                
                d_pos = torch.norm(a_vec[i] - p_vec[i], p=2)
                d_neg = torch.norm(a_vec[i] - n_vec, p=2)
                
                if d_pos < d_neg:
                    correct += 1
                total += 1
                
                total_pos_dist += d_pos.item()
                total_neg_dist += d_neg.item()
                count += 1

                # Mean Reciprocal Rank (MRR): how high the correct match ranks.
                pos_dists = torch.norm(a_vec[i].unsqueeze(0) - p_vec, p=2, dim=1)
                if mask_same_library:
                    for j in range(len(anchor_keys)):
                        if j == i:
                            continue
                        candidate_lib = anchor_keys[j].split("_", 1)[0]
                        if candidate_lib == positive_lib:
                            pos_dists[j] = float("inf")
                rank = int(torch.argsort(pos_dists).tolist().index(i)) + 1
                total_rr += 1.0 / rank
                rr_count += 1

                lib_key = anchor_keys[i].split("_", 1)[0]
                lib_stats = per_lib_stats.setdefault(
                    lib_key,
                    {"pos_sum": 0.0, "neg_sum": 0.0, "count": 0, "rr_sum": 0.0, "rr_count": 0},
                )
                lib_stats["pos_sum"] += d_pos.item()
                lib_stats["neg_sum"] += d_neg.item()
                lib_stats["count"] += 1
                lib_stats["rr_sum"] += 1.0 / rank
                lib_stats["rr_count"] += 1
    
    avg_pos = total_pos_dist / count if count > 0 else 0
    avg_neg = total_neg_dist / count if count > 0 else 0
    mrr = total_rr / rr_count if rr_count > 0 else 0
    per_lib_margins = {}
    per_lib_mrr = {}
    for lib, stats in per_lib_stats.items():
        if stats["count"] > 0:
            avg_pos_lib = stats["pos_sum"] / stats["count"]
            avg_neg_lib = stats["neg_sum"] / stats["count"]
            per_lib_margins[lib] = avg_neg_lib - avg_pos_lib
        if stats["rr_count"] > 0:
            per_lib_mrr[lib] = stats["rr_sum"] / stats["rr_count"]
    
    # Worst-case (min) and average MRR across libraries to measure generalization.
    min_lib_mrr = min(per_lib_mrr.values()) if per_lib_mrr else 0
    avg_lib_mrr = sum(per_lib_mrr.values()) / len(per_lib_mrr) if per_lib_mrr else 0
    
    return (
        (correct / total) * 100 if total > 0 else 0,
        avg_pos,
        avg_neg,
        mrr,
        per_lib_margins,
        per_lib_mrr,
        min_lib_mrr,
        avg_lib_mrr,
    )

def train_brain(bootstrap_dir, epochs=5, batch_size=16, force=False, lr=0.001, margin=0.2, embed_dim=32, hidden_dim=64, is_sweep=False, device_name="cuda", max_nodes_override=None, val_library=None):
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

    effective_max_nodes = max_nodes_override if max_nodes_override else get_auto_max_nodes(device)
    if not is_sweep:
        source = "Manual override" if max_nodes_override else f"Auto-detected for {device.type}"
        print(f"[*] Context Window: {effective_max_nodes} nodes ({source})")

    node_type_count = len(NODE_TYPES)
    if node_type_count < 2:
        print("[!] Error: NODE_TYPES is unexpectedly small; run 'npm run sync-vocab' before training.")
        return (0, 0, 0, 0), None
    current_vocab_size = node_type_count + 5

    if not is_sweep: print(f"[*] Starting 'Brain' Training on Bootstrap Data: {bootstrap_dir}")
    
    ast_files = [f for f in os.listdir(bootstrap_dir) if f.endswith('_gold_asts.json')]
    if not ast_files:
        print("[!] Error: No gold ASTs found. Run 'npm run bootstrap' first.")
        return (0, 0, 0, 0), None

    patterns = {}
    libraries = set()
    for f_name in ast_files:
        lib_name = f_name.replace('_gold_asts.json', '')
        libraries.add(lib_name)
        with open(os.path.join(bootstrap_dir, f_name), 'r') as f:
            data = json.load(f)
            for chunk_name, ast in data.items():
                patterns[f"{lib_name}_{chunk_name}"] = ast

    # 4. Leave-Library-Out Validation (single or multiple libraries).
    # Multiple validation libraries allow same-library masking during eval.
    if val_library:
        if isinstance(val_library, (list, tuple, set)):
            val_libraries = list(dict.fromkeys(val_library))
        else:
            val_libraries = [val_library]
    else:
        val_libraries = [random.choice(list(libraries))]
        if not is_sweep:
            print(f"[*] Leave-One-Library-Out: Validating on '{val_libraries[0]}'")

    if not is_sweep and len(val_libraries) > 1:
        joined_libs = ", ".join(val_libraries)
        print(f"[*] Leave-Multi-Library-Out: Validating on [{joined_libs}]")

    val_prefixes = tuple(f"{lib}_" for lib in val_libraries)
    train_patterns = {k: v for k, v in patterns.items() if not k.startswith(val_prefixes)}
    val_patterns = {k: v for k, v in patterns.items() if k.startswith(val_prefixes)}

    if not val_patterns:
        print(f"[!] Warning: Validation libraries {val_libraries} have no patterns. Falling back to 80/20 split.")
        full_dataset = TripletDataset(patterns, max_nodes=effective_max_nodes)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    else:
        train_dataset = TripletDataset(train_patterns, max_nodes=effective_max_nodes)
        val_dataset = TripletDataset(val_patterns, max_nodes=effective_max_nodes)

    use_cuda = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_cuda)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=use_cuda)

    model = CodeFingerprinter(vocab_size=current_vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, max_nodes=effective_max_nodes).to(device)
    
    # Robust Checkpoint Loading (Skip if sweep unless specified)
    if not is_sweep:
        model_path = os.path.join(os.path.dirname(__file__), "model.pth")
        if os.path.exists(model_path):
            print(f"[*] Found existing model at {model_path}. Attempting to load...")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                if 'transformer_encoder.embedding.weight' in checkpoint:
                    embedding_key = 'transformer_encoder.embedding.weight'
                elif 'embedding.weight' in checkpoint:
                    embedding_key = 'embedding.weight'
                else:
                    print("[!] Error: Checkpoint missing embedding weights; skipping load.")
                    embedding_key = None

                # Robust loading: Check shapes for all parameters (Vocab and Max Nodes)
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
                print(f"[!] Error loading checkpoint: {e}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    
    best_val_acc = 0
    best_val_margin = -1.0
    best_val_mrr = 0.0
    best_val_min_lib_margin = 0.0
    best_val_min_lib_mrr = 0.0
    best_val_avg_lib_mrr = 0.0
    best_state = None
    early_stop_patience = 3
    early_stop_min_delta = 0.01
    early_stop_bad_epochs = 0

    print(f"[*] Starting training loop for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (anchors, a_lits, positives, p_lits, anchor_keys) in enumerate(train_loader):
            anchors, a_lits = anchors.to(device), a_lits.to(device)
            positives, p_lits = positives.to(device), p_lits.to(device)

            with torch.no_grad():
                anchor_vecs = model(anchors, a_lits)
            
            # Vectorized Hard Negative Mining
            sims = torch.matmul(anchor_vecs, anchor_vecs.T)
            negatives = []
            neg_lits = []
            
            for i in range(len(anchors)):
                batch_sims = sims[i].clone()
                batch_sims[i] = -2  # Mask self
                
                anchor_hash = train_dataset.structural_hashes[anchor_keys[i]]
                anchor_lib = anchor_keys[i].split("_", 1)[0]
                for j in range(len(anchors)):
                    candidate_key = anchor_keys[j]
                    candidate_lib = candidate_key.split("_", 1)[0]
                    if train_dataset.structural_hashes[candidate_key] == anchor_hash or candidate_lib == anchor_lib:
                        batch_sims[j] = -2 # Mask isomorphs
                
                hardest_idx = torch.argmax(batch_sims)
                if batch_sims[hardest_idx] == -2:
                    # Fallback: prefer a different library, then any other item.
                    anchor_lib = anchor_keys[i].split("_", 1)[0]
                    fallback_idx = None
                    for j in range(len(anchors)):
                        if j == i:
                            continue
                        candidate_key = anchor_keys[j]
                        candidate_lib = candidate_key.split("_", 1)[0]
                        if candidate_lib == anchor_lib:
                            continue
                        if train_dataset.structural_hashes[candidate_key] == anchor_hash:
                            continue
                        fallback_idx = j
                        break
                    if fallback_idx is None:
                        for j in range(len(anchors)):
                            if j == i:
                                continue
                            candidate_key = anchor_keys[j]
                            candidate_lib = candidate_key.split("_", 1)[0]
                            if candidate_lib != anchor_lib:
                                fallback_idx = j
                                break
                    if fallback_idx is None:
                        fallback_idx = (i + 1) % len(anchors)
                    hardest_idx = fallback_idx
                    
                negatives.append(anchors[hardest_idx])
                neg_lits.append(a_lits[hardest_idx])
            
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

            if not is_sweep and batch_idx % 10 == 0:
                print(f"      Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        mask_same_library = len(val_libraries) > 1
        val_acc, avg_pos, avg_neg, val_mrr, per_lib_margins, per_lib_mrr, min_lib_mrr, avg_lib_mrr = evaluate_model(
            model,
            val_loader,
            device,
            val_dataset,
            mask_same_library=mask_same_library,
        )
        val_margin = avg_neg - avg_pos
        min_lib_margin = min(per_lib_margins.values()) if per_lib_margins else 0.0
        
        if not is_sweep:
            print(f"    Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")
            print(f"    Validation Match Accuracy: {val_acc:.2f}%")
            print(f"    Similarity Spread - Pos Dist: {avg_pos:.4f}, Neg Dist: {avg_neg:.4f} (Margin: {val_margin:.4f})")
            print(f"    Validation MRR: {val_mrr:.4f}")
            if mask_same_library and per_lib_margins:
                print(f"    Validation Min-Library Margin: {min_lib_margin:.4f}")
            if mask_same_library and per_lib_mrr:
                print(f"    Validation Min-Library MRR: {min_lib_mrr:.4f}")
                print(f"    Validation Avg-Library MRR: {avg_lib_mrr:.4f}")
        
        # Optimize for margin during training; sweep scoring happens in run_sweep.
        if val_margin > best_val_margin:
            best_val_margin = val_margin
            best_val_acc = val_acc
            best_val_mrr = val_mrr
            best_val_min_lib_margin = min_lib_margin
            best_val_min_lib_mrr = min_lib_mrr
            best_val_avg_lib_mrr = avg_lib_mrr
            best_state = model.state_dict()
            if not is_sweep:
                torch.save(best_state, os.path.join(os.path.dirname(__file__), "model.pth"))
            early_stop_bad_epochs = 0
        elif val_margin <= best_val_margin + early_stop_min_delta:
            early_stop_bad_epochs += 1

        if early_stop_bad_epochs >= early_stop_patience:
            if not is_sweep:
                print(f"    Early stopping after {early_stop_bad_epochs} stagnant epochs.")
            break

    return (best_val_margin, best_val_acc, best_val_mrr, best_val_min_lib_margin, best_val_min_lib_mrr, best_val_avg_lib_mrr), best_state

def run_sweep(bootstrap_dir, epochs=5, device_name="auto", max_nodes_override=None):
    print("[*] Starting Hyperparameter Sweep (Targeting Maximal Margin)...")
    results = []
    
    # Loss margins to explore; higher margins demand larger separations.
    margins = [0.2, 0.5, 0.8, 1.0]
    lrs = [0.001, 0.0005, 0.0001]
    embed_dims = [32, 64, 128]
    hidden_dims = [64, 128]
    sweep_batch_size = 64
    
    ast_files = [f for f in os.listdir(bootstrap_dir) if f.endswith("_gold_asts.json")]
    if not ast_files:
        print("[!] Error: No gold ASTs found. Run 'npm run bootstrap' first.")
        return
    libraries = [f.replace("_gold_asts.json", "") for f in ast_files]
    # Use up to 3 validation libraries to stress generalization.
    val_lib_count = min(3, len(libraries))
    fixed_val_libs = random.sample(libraries, val_lib_count) if libraries else []
    if fixed_val_libs:
        joined_libs = ", ".join(fixed_val_libs)
        print(f"[*] Sweep validating across: {joined_libs}")

    best_overall_score = -1.0
    best_overall_margin = -1.0
    best_params = None
    best_state = None

    for m in margins:
        for lr in lrs:
            for ed in embed_dims:
                for hd in hidden_dims:
                    print(f"    - Testing Margin: {m}, LR: {lr}, Embed: {ed}, Hidden: {hd}...")
                    result, state = train_brain(
                        bootstrap_dir,
                        epochs=epochs,
                        batch_size=sweep_batch_size,
                        is_sweep=True,
                        margin=m,
                        lr=lr,
                        embed_dim=ed,
                        hidden_dim=hd,
                        device_name=device_name,
                        max_nodes_override=max_nodes_override,
                        val_library=fixed_val_libs if fixed_val_libs else None,
                    )
                    val_margin, val_acc, val_mrr, min_lib_margin, min_lib_mrr, avg_lib_mrr = result
                    if min_lib_mrr > 0:
                        print(f"      Result: MinLib MRR {min_lib_mrr:.4f}, AvgLib MRR {avg_lib_mrr:.4f}, Margin {val_margin:.4f}")
                    elif min_lib_margin > 0:
                        print(f"      Result: Acc {val_acc:.2f}%, Margin {val_margin:.4f}, MRR {val_mrr:.4f}, MinLib {min_lib_margin:.4f}")
                    else:
                        print(f"      Result: Acc {val_acc:.2f}%, Margin {val_margin:.4f}, MRR {val_mrr:.4f}")
                    results.append({
                        "margin": m,
                        "lr": lr,
                        "embed": ed,
                        "hidden": hd,
                        "acc": val_acc,
                        "val_margin": val_margin,
                        "mrr": val_mrr,
                        "min_lib_margin": min_lib_margin,
                        "min_lib_mrr": min_lib_mrr,
                        "avg_lib_mrr": avg_lib_mrr,
                    })
                    
                    # Selection score favors worst-case (min) library MRR,
                    # then average MRR, with margin as a mild confidence tie-breaker.
                    score = (min_lib_mrr * 0.7) + (avg_lib_mrr * 0.2) + (min(val_margin, 1.0) * 0.1)
                    if val_acc <= 0 or min_lib_mrr <= 0:
                        continue
                    if score > best_overall_score:
                        best_overall_score = score
                        best_overall_margin = val_margin
                        best_params = {"margin": m, "lr": lr, "embed": ed, "hidden": hd, "mrr": min_lib_mrr}
                        best_state = state

    print(f"\n[+] Sweep Complete!")
    if best_params:
        print(f"[+] Best Similarity Margin: {best_overall_margin:.4f} (Acc: {best_params['acc']:.2f}%)")
        print(f"[+] Best Params: {best_params}")
        if best_state:
            model_path = os.path.join(os.path.dirname(__file__), "model.pth")
            torch.save(best_state, model_path)
            print(f"[*] Best model saved to {model_path}")
    else:
        print("[!] Sweep failed to find a valid model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Code Fingerprinting Neural Network")
    parser.add_argument("bootstrap_dir", nargs="?", default="./ml/bootstrap_data", help="Directory containing gold ASTs")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--force", action="store_true", help="Force loading weights even if vocabulary size mismatches")
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")
    parser.add_argument("--max_nodes", type=int, default=0, help="Override context window size (0 for auto)")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda, mps, cpu, or auto")
    
    args = parser.parse_args()
    
    # Handle max_nodes override
    m_nodes = args.max_nodes if args.max_nodes > 0 else None

    if args.sweep:
        run_sweep(args.bootstrap_dir, epochs=args.epochs, device_name=args.device, max_nodes_override=m_nodes)
    else:
        train_brain(args.bootstrap_dir, epochs=args.epochs, batch_size=args.batch_size, force=args.force, device_name=args.device, max_nodes_override=m_nodes)
