"""
Heterogeneous Graph Transformer (HGT) Model
For ED Utilization Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from typing import Dict, List, Tuple

import config_hgt as config


class HGTClassifier(nn.Module):
    """
    Heterogeneous Graph Transformer for multi-class classification
    
    Architecture:
    1. Input projection layers for each node type
    2. Multiple HGT convolution layers
    3. Prediction head for patient nodes
    """
    
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        node_feature_dims: Dict[str, int],
        hidden_dim: int = config.HGT_HIDDEN_DIM,
        num_layers: int = config.HGT_NUM_LAYERS,
        num_heads: int = config.HGT_NUM_HEADS,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.HGT_DROPOUT,
        use_norm: bool = config.HGT_USE_NORM
    ):
        super(HGTClassifier, self).__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Input projection layers for each node type
        self.input_projections = nn.ModuleDict()
        for node_type in node_types:
            input_dim = node_feature_dims[node_type]
            self.input_projections[node_type] = Linear(input_dim, hidden_dim)
        
        # HGT convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=(node_types, edge_types),
                heads=num_heads,
                dropout=dropout
            )
            self.convs.append(conv)
        
        # Layer normalization (optional)
        self.use_norm = use_norm
        if use_norm:
            self.norms = nn.ModuleList()
            for _ in range(num_layers):
                self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Prediction head (only for patient nodes)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, config.PREDICTION_HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.PREDICTION_HEAD_DROPOUT),
            nn.Linear(config.PREDICTION_HEAD_HIDDEN_DIM, num_classes)
        )
    
    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass
        
        Args:
            x_dict: Dictionary of node features {node_type: tensor}
            edge_index_dict: Dictionary of edge indices {edge_type: tensor}
        
        Returns:
            logits: Classification logits for patient nodes (num_patients, num_classes)
        """
        # Project input features to hidden dimension
        h_dict = {}
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.input_projections[node_type](x)
            h_dict[node_type] = F.relu(h_dict[node_type])
            h_dict[node_type] = F.dropout(
                h_dict[node_type], 
                p=self.dropout, 
                training=self.training
            )
        
        # Apply HGT convolution layers
        for i, conv in enumerate(self.convs):
            # HGT convolution
            h_dict = conv(h_dict, edge_index_dict)
            
            # Apply layer normalization
            if self.use_norm:
                for node_type in h_dict.keys():
                    h_dict[node_type] = self.norms[i](h_dict[node_type])
            
            # Apply ReLU and dropout (except for last layer)
            if i < len(self.convs) - 1:
                for node_type in h_dict.keys():
                    h_dict[node_type] = F.relu(h_dict[node_type])
                    h_dict[node_type] = F.dropout(
                        h_dict[node_type],
                        p=self.dropout,
                        training=self.training
                    )
        
        # Extract patient embeddings
        patient_h = h_dict['patient']
        
        # Apply prediction head
        logits = self.prediction_head(patient_h)
        
        return logits
    
    def get_embeddings(self, x_dict, edge_index_dict):
        """
        Get node embeddings without prediction head
        
        Returns:
            h_dict: Dictionary of node embeddings
        """
        # Project input features
        h_dict = {}
        for node_type, x in x_dict.items():
            h_dict[node_type] = self.input_projections[node_type](x)
            h_dict[node_type] = F.relu(h_dict[node_type])
        
        # Apply HGT layers
        for i, conv in enumerate(self.convs):
            h_dict = conv(h_dict, edge_index_dict)
            
            if self.use_norm:
                for node_type in h_dict.keys():
                    h_dict[node_type] = self.norms[i](h_dict[node_type])
            
            if i < len(self.convs) - 1:
                for node_type in h_dict.keys():
                    h_dict[node_type] = F.relu(h_dict[node_type])
        
        return h_dict


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights (tensor of size num_classes)
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_loss_function(class_weights=None):
    """
    Create loss function with optional class weights
    
    Args:
        class_weights: Tensor of class weights or None
    
    Returns:
        Loss function
    """
    if config.USE_CLASS_WEIGHTS and class_weights is not None:
        if config.LABEL_SMOOTHING > 0:
            return nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=config.LABEL_SMOOTHING
            )
        else:
            return FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        return nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)


def compute_class_weights(labels, num_classes=config.NUM_CLASSES):
    """
    Compute class weights inversely proportional to class frequencies
    
    Args:
        labels: Tensor or array of class labels
        num_classes: Number of classes
    
    Returns:
        Tensor of class weights
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Count class frequencies
    class_counts = torch.zeros(num_classes)
    for i in range(num_classes):
        class_counts[i] = (labels == i).sum()
    
    # Compute weights (inverse frequency)
    total = class_counts.sum()
    class_weights = total / (num_classes * class_counts)
    
    # Normalize so that min weight is 1.0
    class_weights = class_weights / class_weights.min()
    
    return class_weights


def count_parameters(model):
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test the HGT model"""
    print("Testing HGT Model...")
    
    # Create dummy data
    batch_size = 100
    node_feature_dims = {
        'patient': 20,
        'diagnosis': 32,
        'procedure': 32,
        'provider': 8,
        'hospital': 8
    }
    
    # Create model
    model = HGTClassifier(
        node_types=['patient', 'diagnosis', 'procedure', 'provider', 'hospital'],
        edge_types=[
            ('patient', 'has_diagnosis', 'diagnosis'),
            ('diagnosis', 'diagnosed_in', 'patient'),
            ('patient', 'has_procedure', 'procedure'),
            ('procedure', 'performed_on', 'patient'),
            ('procedure', 'performed_by', 'provider'),
            ('provider', 'performs', 'procedure'),
            ('provider', 'works_at', 'hospital'),
            ('hospital', 'employs', 'provider'),
            ('patient', 'visits', 'hospital'),
            ('hospital', 'visited_by', 'patient')
        ],
        node_feature_dims=node_feature_dims,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        num_classes=3,
        dropout=0.3
    )
    
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Create dummy input
    x_dict = {
        'patient': torch.randn(batch_size, node_feature_dims['patient']),
        'diagnosis': torch.randn(50, node_feature_dims['diagnosis']),
        'procedure': torch.randn(80, node_feature_dims['procedure']),
        'provider': torch.randn(30, node_feature_dims['provider']),
        'hospital': torch.randn(10, node_feature_dims['hospital'])
    }
    
    edge_index_dict = {
        ('patient', 'has_diagnosis', 'diagnosis'): torch.randint(0, 50, (2, 200)),
        ('diagnosis', 'diagnosed_in', 'patient'): torch.randint(0, 50, (2, 200)),
        ('patient', 'has_procedure', 'procedure'): torch.randint(0, 80, (2, 300)),
        ('procedure', 'performed_on', 'patient'): torch.randint(0, 80, (2, 300)),
        ('procedure', 'performed_by', 'provider'): torch.randint(0, 30, (2, 150)),
        ('provider', 'performs', 'procedure'): torch.randint(0, 30, (2, 150)),
        ('provider', 'works_at', 'hospital'): torch.randint(0, 10, (2, 50)),
        ('hospital', 'employs', 'provider'): torch.randint(0, 10, (2, 50)),
        ('patient', 'visits', 'hospital'): torch.randint(0, 10, (2, 120)),
        ('hospital', 'visited_by', 'patient'): torch.randint(0, 10, (2, 120))
    }
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(x_dict, edge_index_dict)
    
    print(f"Output shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, 3)")
    
    # Test loss
    labels = torch.randint(0, 3, (batch_size,))
    class_weights = compute_class_weights(labels)
    print(f"Class weights: {class_weights}")
    
    loss_fn = create_loss_function(class_weights)
    loss = loss_fn(logits, labels)
    print(f"Loss: {loss.item():.4f}")
    
    print("\nModel test completed successfully!")
    
    return model


if __name__ == "__main__":
    test_model()
