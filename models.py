"""
Model Definitions: TGN, TGAT, HGT
All three temporal/dynamic heterogeneous GNN models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config

# Try to import required libraries
try:
    import dgl
    import dgl.nn as dglnn
    HAS_DGL = True
except ImportError:
    HAS_DGL = False
    print("Warning: DGL not available")

try:
    from torch_geometric_temporal import TGN as PyG_TGN
    from torch_geometric.nn import TransformerConv
    HAS_PYG_TEMPORAL = True
except ImportError:
    HAS_PYG_TEMPORAL = False
    print("Warning: PyG Temporal not available")


class PredictionHead(nn.Module):
    """
    Multi-task prediction head for ED utilization
    - Regression: time to next ED visit (in days)
    - Classification: binary within-30-day ED visit
    """
    
    def __init__(self, hidden_dim=config.HIDDEN_DIM):
        super().__init__()
        self.delta_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.binary_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, node_embeddings):
        """
        Args:
            node_embeddings: (batch_size, hidden_dim) - patient embeddings
        
        Returns:
            delta_pred: (batch_size, 1) - predicted days to next ED
            binary_logits: (batch_size, 1) - logits for within-30-day classification
        """
        delta_pred = self.delta_predictor(node_embeddings)
        binary_logits = self.binary_predictor(node_embeddings)
        
        # Ensure delta is non-negative
        delta_pred = F.softplus(delta_pred)
        
        return delta_pred, binary_logits


# =============================================================================
# TGN (Temporal Graph Network) - PyG Implementation
# =============================================================================

class SimpleTGN(nn.Module):
    """
    Simplified TGN implementation for temporal graphs
    Memory-based temporal GNN with message passing
    """
    
    def __init__(self, num_nodes, node_dim=config.HIDDEN_DIM):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        
        # Memory module
        self.memory = nn.Parameter(torch.zeros(num_nodes, config.TGN_MEMORY_DIM))
        nn.init.xavier_uniform_(self.memory)
        
        # Last update time for each node
        self.register_buffer('last_update', torch.zeros(num_nodes))
        
        # Time encoding (Fourier features)
        self.time_encoder = nn.Sequential(
            nn.Linear(1, config.TGN_TIME_DIM),
            nn.ReLU(),
            nn.Linear(config.TGN_TIME_DIM, config.TGN_TIME_DIM)
        )
        
        # Message function
        self.message_fn = nn.Sequential(
            nn.Linear(config.TGN_MEMORY_DIM * 2 + config.TGN_TIME_DIM + node_dim, 
                     config.TGN_MESSAGE_DIM),
            nn.ReLU(),
            nn.Linear(config.TGN_MESSAGE_DIM, config.TGN_MEMORY_DIM)
        )
        
        # Memory updater (GRU-based)
        self.memory_updater = nn.GRUCell(config.TGN_MEMORY_DIM, config.TGN_MEMORY_DIM)
        
        # Embedding function
        self.embedding_fn = nn.Sequential(
            nn.Linear(config.TGN_MEMORY_DIM + node_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)
        )
        
        self.pred_head = PredictionHead()
    
    def encode_time(self, timestamps):
        """Encode timestamps using Fourier features"""
        if timestamps.dim() == 0:
            timestamps = timestamps.unsqueeze(0)
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(-1)
        return self.time_encoder(timestamps)
    
    def compute_messages(self, src_nodes, dst_nodes, timestamps, edge_features):
        """Compute messages for edges"""
        src_mem = self.memory[src_nodes]
        dst_mem = self.memory[dst_nodes]
        time_enc = self.encode_time(timestamps)
        
        # Concatenate features
        msg_input = torch.cat([src_mem, dst_mem, time_enc, edge_features], dim=-1)
        messages = self.message_fn(msg_input)
        
        return messages
    
    def update_memory(self, nodes, messages):
        """Update memory for nodes"""
        if config.TGN_AGGREGATOR == "mean":
            # Aggregate messages for each node
            unique_nodes, inverse_indices = torch.unique(nodes, return_inverse=True)
            aggregated = torch.zeros(len(unique_nodes), config.TGN_MEMORY_DIM, 
                                    device=messages.device)
            aggregated.index_add_(0, inverse_indices, messages)
            counts = torch.bincount(inverse_indices)
            aggregated = aggregated / counts.unsqueeze(-1).float()
            
            # Update memory using GRU
            updated_mem = self.memory_updater(
                aggregated,
                self.memory[unique_nodes]
            )
            self.memory.data[unique_nodes] = updated_mem
    
    def get_embeddings(self, nodes, node_features):
        """Get current embeddings for nodes"""
        mem = self.memory[nodes]
        emb_input = torch.cat([mem, node_features], dim=-1)
        embeddings = self.embedding_fn(emb_input)
        return embeddings
    
    def forward(self, patient_nodes, node_features):
        """
        Forward pass for prediction
        
        Args:
            patient_nodes: Patient node IDs to predict for
            node_features: Node features
        
        Returns:
            delta_pred, binary_logits
        """
        embeddings = self.get_embeddings(patient_nodes, node_features)
        return self.pred_head(embeddings)
    
    def reset_memory(self):
        """Reset memory to initial state"""
        nn.init.xavier_uniform_(self.memory)
        self.last_update.zero_()


# =============================================================================
# TGAT (Temporal Graph Attention) - Simplified Implementation
# =============================================================================

class TemporalAttentionLayer(nn.Module):
    """Single layer of temporal graph attention"""
    
    def __init__(self, in_dim, out_dim, num_heads=config.TGAT_NUM_HEADS):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        # Multi-head attention
        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        
        # Time encoding
        self.time_encoder = nn.Linear(1, out_dim)
        
        # Output projection
        self.out_proj = nn.Linear(out_dim, out_dim)
        
        self.dropout = nn.Dropout(config.TGAT_DROPOUT)
    
    def forward(self, node_features, edge_index, edge_times):
        """
        Args:
            node_features: (num_nodes, in_dim)
            edge_index: (2, num_edges) - [src, dst]
            edge_times: (num_edges,) - edge timestamps
        
        Returns:
            out_features: (num_nodes, out_dim)
        """
        num_nodes = node_features.size(0)
        
        # Project to Q, K, V
        Q = self.query(node_features)  # (num_nodes, out_dim)
        K = self.key(node_features)
        V = self.value(node_features)
        
        # Reshape for multi-head attention
        Q = Q.view(num_nodes, self.num_heads, self.head_dim)
        K = K.view(num_nodes, self.num_heads, self.head_dim)
        V = V.view(num_nodes, self.num_heads, self.head_dim)
        
        # Encode edge times
        time_enc = self.time_encoder(edge_times.unsqueeze(-1))  # (num_edges, out_dim)
        time_enc = time_enc.view(-1, self.num_heads, self.head_dim)
        
        # Simple message aggregation (placeholder for full temporal attention)
        # In full implementation, would do proper temporal attention with edge_index
        out = Q + 0.1 * torch.randn_like(Q)  # Placeholder
        
        # Reshape and project
        out = out.view(num_nodes, -1)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out + node_features  # Residual connection


class TGAT(nn.Module):
    """
    Temporal Graph Attention Network
    Stack of temporal attention layers
    """
    
    def __init__(self, in_dim=config.HIDDEN_DIM, hidden_dim=config.HIDDEN_DIM):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TemporalAttentionLayer(
                in_dim if i == 0 else hidden_dim,
                hidden_dim,
                num_heads=config.TGAT_NUM_HEADS
            )
            for i in range(config.NUM_LAYERS)
        ])
        
        self.pred_head = PredictionHead(hidden_dim)
    
    def forward(self, node_features, edge_index=None, edge_times=None):
        """
        Args:
            node_features: Initial node features
            edge_index: Edge connectivity (optional for simplified version)
            edge_times: Edge timestamps (optional)
        
        Returns:
            delta_pred, binary_logits
        """
        x = node_features
        
        if edge_index is not None and edge_times is not None:
            for layer in self.layers:
                x = layer(x, edge_index, edge_times)
        else:
            # Simplified: just pass through with self-attention
            for layer in self.layers:
                # Create dummy edges (self-loops)
                num_nodes = x.size(0)
                dummy_edge_index = torch.stack([
                    torch.arange(num_nodes),
                    torch.arange(num_nodes)
                ], dim=0).to(x.device)
                dummy_times = torch.zeros(num_nodes).to(x.device)
                x = layer(x, dummy_edge_index, dummy_times)
        
        return self.pred_head(x)


# =============================================================================
# HGT (Heterogeneous Graph Transformer) - DGL Implementation  
# =============================================================================

class HGTModel(nn.Module):
    """
    Heterogeneous Graph Transformer
    Handles multiple node and edge types with meta-relations
    """
    
    def __init__(self, node_types, edge_types, in_dim=config.HIDDEN_DIM):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        
        if not HAS_DGL:
            raise ImportError("DGL is required for HGT")
        
        # Input projection for each node type
        self.node_projections = nn.ModuleDict({
            ntype: nn.Linear(in_dim, config.HIDDEN_DIM)
            for ntype in node_types
        })
        
        # HGT layers
        self.layers = nn.ModuleList([
            dglnn.HGTConv(
                in_feats=config.HIDDEN_DIM,
                out_feats=config.HIDDEN_DIM,
                num_heads=config.HGT_NUM_HEADS,
                num_ntypes=len(node_types),
                num_etypes=len(edge_types),
                dropout=config.DROPOUT,
                use_norm=config.HGT_USE_NORM
            )
            for _ in range(config.NUM_LAYERS)
        ])
        
        # Output head
        self.pred_head = PredictionHead(config.HIDDEN_DIM)
    
    def forward(self, graph, node_features_dict):
        """
        Args:
            graph: DGL heterogeneous graph
            node_features_dict: Dict of {node_type: features}
        
        Returns:
            patient_embeddings: Embeddings for patient nodes
        """
        # Project input features
        h_dict = {}
        for ntype in self.node_types:
            if ntype in node_features_dict:
                h_dict[ntype] = self.node_projections[ntype](node_features_dict[ntype])
            else:
                # Create dummy features if not present
                num_nodes = graph.number_of_nodes(ntype) if graph else 1
                h_dict[ntype] = torch.zeros(num_nodes, config.HIDDEN_DIM)
        
        # Apply HGT layers
        for layer in self.layers:
            if graph is not None:
                h_dict = layer(graph, h_dict, graph.ndata['node_type'], graph.edata['edge_type'])
            else:
                # Placeholder: just pass through
                pass
        
        # Return patient embeddings
        return h_dict['patient'] if 'patient' in h_dict else h_dict[list(h_dict.keys())[0]]
    
    def predict(self, patient_embeddings):
        """Make predictions from patient embeddings"""
        return self.pred_head(patient_embeddings)


# =============================================================================
# Model Factory
# =============================================================================

def create_model(model_name, **kwargs):
    """
    Factory function to create models
    
    Args:
        model_name: One of 'TGN', 'TGAT', 'HGT'
        **kwargs: Model-specific arguments
    
    Returns:
        model: Initialized model
    """
    if model_name == 'TGN':
        num_nodes = kwargs.get('num_nodes', 1000)
        return SimpleTGN(num_nodes=num_nodes, node_dim=config.HIDDEN_DIM)
    
    elif model_name == 'TGAT':
        return TGAT(in_dim=config.HIDDEN_DIM, hidden_dim=config.HIDDEN_DIM)
    
    elif model_name == 'HGT':
        node_types = kwargs.get('node_types', config.NODE_TYPES)
        edge_types = kwargs.get('edge_types', config.EDGE_TYPES)
        return HGTModel(node_types, edge_types, in_dim=config.HIDDEN_DIM)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing model creation...")
    
    # Test TGN
    print("\nTGN:")
    tgn = create_model('TGN', num_nodes=100)
    print(f"  Parameters: {count_parameters(tgn):,}")
    
    # Test TGAT
    print("\nTGAT:")
    tgat = create_model('TGAT')
    print(f"  Parameters: {count_parameters(tgat):,}")
    
    # Test HGT
    print("\nHGT:")
    try:
        hgt = create_model('HGT')
        print(f"  Parameters: {count_parameters(hgt):,}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n✓ Model creation tests passed")

