import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import NODE_TYPES, TYPE_TO_ID, VOCAB_SIZE, MAX_NODES

class TransformerCodeEncoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=128, nhead=8, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Positional Encoding is required for Transformers to know order
        # MAX_NODES + 1 for safety
        self.pos_encoder = nn.Parameter(torch.zeros(1, MAX_NODES + 1, embed_dim)) 
        
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, nhead, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(embed_dim, 64)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer(x)
        
        # Global average pooling (masking out PAD/0 nodes if necessary, 
        # but simple average over seq dimension is often sufficient for code structure)
        x = x.mean(dim=1)
        return F.normalize(self.fc(x), p=2, dim=1)

class ASTPreprocessor:
    def __init__(self):
        # Common Babel node types. This list can be expanded.
        self.node_types = NODE_TYPES
        self.type_to_id = TYPE_TO_ID
        self.vocab_size = VOCAB_SIZE

    def get_type_id(self, type_name):
        return self.type_to_id.get(type_name, self.type_to_id["UNKNOWN"])

    def flatten_ast(self, node, depth=0, max_depth=10):
        """Recursively flattens the structural AST into a sequence of type IDs."""
        if not node or depth > max_depth:
            return []
        
        type_id = self.get_type_id(node.get("type", "UNKNOWN"))
        sequence = [type_id]
        
        children = node.get("children", [])
        for child in children:
            sequence.extend(self.flatten_ast(child, depth + 1, max_depth))
            
        return sequence

    def process_chunk(self, structural_ast_nodes, max_seq_len=MAX_NODES):
        """Converts a list of structural AST nodes into a single padded tensor."""
        full_sequence = []
        for node in structural_ast_nodes:
            full_sequence.extend(self.flatten_ast(node))
            
        # Truncate or pad
        if len(full_sequence) > max_seq_len:
            full_sequence = full_sequence[:max_seq_len]
        else:
            full_sequence.extend([0] * (max_seq_len - len(full_sequence)))
            
        return torch.tensor([full_sequence], dtype=torch.long)
