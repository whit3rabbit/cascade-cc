import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import NODE_TYPES, TYPE_TO_ID, VOCAB_SIZE

class CodeStructureEncoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=64, hidden_dim=128):
        super(CodeStructureEncoder, self).__init__()
        # vocab_size is the number of unique AST Node types
        self.nodes_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM to process structural sequences
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Final projection to a fixed-size fingerprint
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.nodes_embedding(x)
        
        # Process structural sequence
        _, (hidden, _) = self.lstm(embedded)
        
        # Concatenate forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Return the normalized Logic Vector
        return F.normalize(self.fc(hidden_cat), p=2, dim=1)

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

    def process_chunk(self, structural_ast_nodes, max_seq_len=256):
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
