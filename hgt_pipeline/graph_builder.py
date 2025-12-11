"""
Heterogeneous Graph Construction Module
Builds a heterogeneous graph with multiple node and edge types
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
from torch_geometric.data import HeteroData
import warnings
warnings.filterwarnings('ignore')

import config_hgt as config
from data_loader import DataLoader


class HeterogeneousGraphBuilder:
    """Build heterogeneous graph for HGT model"""
    
    def __init__(self, data_loader: DataLoader, verbose=True):
        self.data_loader = data_loader
        self.verbose = verbose
        
        # Mappings from original IDs to graph node indices
        self.patient_id_map = {}
        self.diagnosis_id_map = {}
        self.procedure_id_map = {}
        self.provider_id_map = {}
        self.hospital_id_map = {}
        
        # Reverse mappings
        self.patient_idx_to_id = {}
        self.diagnosis_idx_to_id = {}
        self.procedure_idx_to_id = {}
        self.provider_idx_to_id = {}
        self.hospital_idx_to_id = {}
        
        # Code frequency for filtering rare codes
        self.diagnosis_freq = Counter()
        self.procedure_freq = Counter()
        
    def build_node_mappings(self, patient_ids: List) -> Dict:
        """
        Build mappings from original IDs to node indices
        
        Args:
            patient_ids: List of patient IDs to include in the graph
        
        Returns:
            Dictionary with node counts
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("Building node mappings...")
            print("=" * 80)
        
        # Map patients (only those with labels)
        for idx, patient_id in enumerate(sorted(patient_ids)):
            self.patient_id_map[patient_id] = idx
            self.patient_idx_to_id[idx] = patient_id
        
        # Get diagnosis and procedure data
        dx_data = self.data_loader.get_diagnosis_data()
        proc_data = self.data_loader.get_procedure_data()
        
        # Filter to only include data from labeled patients
        dx_data = dx_data[dx_data['patient_id'].isin(patient_ids)]
        proc_data = proc_data[proc_data['patient_id'].isin(patient_ids)]
        
        # Count code frequencies
        self.diagnosis_freq = Counter(dx_data['diagnosis_code'].dropna())
        self.procedure_freq = Counter(proc_data['procedure_code'].dropna())
        
        # Filter rare codes
        frequent_dx = {
            code for code, count in self.diagnosis_freq.items() 
            if count >= config.MIN_CODE_FREQUENCY
        }
        frequent_proc = {
            code for code, count in self.procedure_freq.items() 
            if count >= config.MIN_CODE_FREQUENCY
        }
        
        # Add UNK tokens
        frequent_dx.add(config.UNK_TOKEN)
        frequent_proc.add(config.UNK_TOKEN)
        
        # Map diagnosis codes
        for idx, dx_code in enumerate(sorted(frequent_dx)):
            self.diagnosis_id_map[dx_code] = idx
            self.diagnosis_idx_to_id[idx] = dx_code
        
        # Map procedure codes
        for idx, proc_code in enumerate(sorted(frequent_proc)):
            self.procedure_id_map[proc_code] = idx
            self.procedure_idx_to_id[idx] = proc_code
        
        # Map providers
        unique_providers = proc_data['provider_npi'].dropna().unique()
        for idx, provider_npi in enumerate(sorted(unique_providers)):
            self.provider_id_map[provider_npi] = idx
            self.provider_idx_to_id[idx] = provider_npi
        
        # Map hospitals
        unique_hospitals = proc_data['hospital_id'].dropna().unique()
        for idx, hospital_id in enumerate(sorted(unique_hospitals)):
            self.hospital_id_map[hospital_id] = idx
            self.hospital_idx_to_id[idx] = hospital_id
        
        # Print statistics
        if self.verbose:
            print(f"  Patients: {len(self.patient_id_map)}")
            print(f"  Diagnosis codes: {len(self.diagnosis_id_map)} "
                  f"(filtered from {len(self.diagnosis_freq)})")
            print(f"  Procedure codes: {len(self.procedure_id_map)} "
                  f"(filtered from {len(self.procedure_freq)})")
            print(f"  Providers: {len(self.provider_id_map)}")
            print(f"  Hospitals: {len(self.hospital_id_map)}")
        
        return {
            'n_patients': len(self.patient_id_map),
            'n_diagnosis': len(self.diagnosis_id_map),
            'n_procedure': len(self.procedure_id_map),
            'n_provider': len(self.provider_id_map),
            'n_hospital': len(self.hospital_id_map)
        }
    
    def create_node_features(self, patient_features: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Create feature tensors for all node types
        
        Args:
            patient_features: DataFrame with patient features
        
        Returns:
            Dictionary mapping node type to feature tensor
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("Creating node features...")
            print("=" * 80)
        
        node_features = {}
        
        # Patient features
        patient_feat_list = []
        for patient_idx in range(len(self.patient_id_map)):
            patient_id = self.patient_idx_to_id[patient_idx]
            patient_row = patient_features[
                patient_features['patient_id'] == patient_id
            ]
            
            if len(patient_row) > 0:
                # Extract numerical features
                feat_cols = [col for col in patient_features.columns 
                            if col != 'patient_id']
                features = patient_row[feat_cols].values[0]
            else:
                # Default features if patient not found
                features = np.zeros(len(patient_features.columns) - 1)
            
            patient_feat_list.append(features)
        
        patient_feat_array = np.array(patient_feat_list, dtype=np.float32)
        
        # Normalize patient features
        if config.NORMALIZE_FEATURES:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            patient_feat_array = scaler.fit_transform(patient_feat_array)
        
        node_features['patient'] = torch.tensor(
            patient_feat_array, dtype=torch.float
        )
        
        # Diagnosis features (learnable embeddings initialized randomly)
        if config.USE_CODE_EMBEDDINGS:
            node_features['diagnosis'] = torch.randn(
                len(self.diagnosis_id_map),
                config.CODE_EMBEDDING_DIM
            )
        else:
            node_features['diagnosis'] = torch.eye(len(self.diagnosis_id_map))
        
        # Procedure features (learnable embeddings initialized randomly)
        if config.USE_CODE_EMBEDDINGS:
            node_features['procedure'] = torch.randn(
                len(self.procedure_id_map),
                config.CODE_EMBEDDING_DIM
            )
        else:
            node_features['procedure'] = torch.eye(len(self.procedure_id_map))
        
        # Provider features (simple embeddings)
        node_features['provider'] = torch.randn(
            len(self.provider_id_map),
            config.PROVIDER_FEATURE_DIM
        )
        
        # Hospital features (simple embeddings)
        node_features['hospital'] = torch.randn(
            len(self.hospital_id_map),
            config.HOSPITAL_FEATURE_DIM
        )
        
        if self.verbose:
            for node_type, features in node_features.items():
                print(f"  {node_type}: {features.shape}")
        
        return node_features
    
    def build_edges(self) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """
        Build edge index tensors for all edge types
        
        Returns:
            Dictionary mapping edge type to edge_index tensor
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("Building edges...")
            print("=" * 80)
        
        edge_indices = {}
        
        # Get data
        dx_data = self.data_loader.get_diagnosis_data()
        proc_data = self.data_loader.get_procedure_data()
        
        # Filter to only patients in our graph
        patient_ids = set(self.patient_id_map.keys())
        dx_data = dx_data[dx_data['patient_id'].isin(patient_ids)]
        proc_data = proc_data[proc_data['patient_id'].isin(patient_ids)]
        
        # 1. Patient -> Diagnosis edges
        patient_diagnosis_edges = []
        for _, row in dx_data.iterrows():
            patient_id = row['patient_id']
            dx_code = row['diagnosis_code']
            
            # Map rare codes to UNK
            if dx_code not in self.diagnosis_id_map:
                dx_code = config.UNK_TOKEN
            
            if patient_id in self.patient_id_map and dx_code in self.diagnosis_id_map:
                patient_idx = self.patient_id_map[patient_id]
                dx_idx = self.diagnosis_id_map[dx_code]
                patient_diagnosis_edges.append([patient_idx, dx_idx])
        
        if len(patient_diagnosis_edges) > 0:
            edge_indices[('patient', 'has_diagnosis', 'diagnosis')] = torch.tensor(
                patient_diagnosis_edges, dtype=torch.long
            ).t().contiguous()
            
            # Reverse edges
            edge_indices[('diagnosis', 'diagnosed_in', 'patient')] = torch.tensor(
                [[e[1], e[0]] for e in patient_diagnosis_edges], dtype=torch.long
            ).t().contiguous()
        
        # 2. Patient -> Procedure edges
        patient_procedure_edges = []
        for _, row in proc_data.iterrows():
            patient_id = row['patient_id']
            proc_code = row['procedure_code']
            
            # Map rare codes to UNK
            if proc_code not in self.procedure_id_map:
                proc_code = config.UNK_TOKEN
            
            if patient_id in self.patient_id_map and proc_code in self.procedure_id_map:
                patient_idx = self.patient_id_map[patient_id]
                proc_idx = self.procedure_id_map[proc_code]
                patient_procedure_edges.append([patient_idx, proc_idx])
        
        if len(patient_procedure_edges) > 0:
            edge_indices[('patient', 'has_procedure', 'procedure')] = torch.tensor(
                patient_procedure_edges, dtype=torch.long
            ).t().contiguous()
            
            # Reverse edges
            edge_indices[('procedure', 'performed_on', 'patient')] = torch.tensor(
                [[e[1], e[0]] for e in patient_procedure_edges], dtype=torch.long
            ).t().contiguous()
        
        # 3. Procedure -> Provider edges
        procedure_provider_edges = []
        for _, row in proc_data.iterrows():
            proc_code = row['procedure_code']
            provider_npi = row['provider_npi']
            
            # Map rare codes to UNK
            if proc_code not in self.procedure_id_map:
                proc_code = config.UNK_TOKEN
            
            if (proc_code in self.procedure_id_map and 
                pd.notna(provider_npi) and 
                provider_npi in self.provider_id_map):
                proc_idx = self.procedure_id_map[proc_code]
                provider_idx = self.provider_id_map[provider_npi]
                procedure_provider_edges.append([proc_idx, provider_idx])
        
        if len(procedure_provider_edges) > 0:
            edge_indices[('procedure', 'performed_by', 'provider')] = torch.tensor(
                procedure_provider_edges, dtype=torch.long
            ).t().contiguous()
            
            # Reverse edges
            edge_indices[('provider', 'performs', 'procedure')] = torch.tensor(
                [[e[1], e[0]] for e in procedure_provider_edges], dtype=torch.long
            ).t().contiguous()
        
        # 4. Provider -> Hospital edges
        provider_hospital_map = {}
        for _, row in proc_data.iterrows():
            provider_npi = row['provider_npi']
            hospital_id = row['hospital_id']
            
            if (pd.notna(provider_npi) and pd.notna(hospital_id) and
                provider_npi in self.provider_id_map and
                hospital_id in self.hospital_id_map):
                provider_hospital_map[provider_npi] = hospital_id
        
        provider_hospital_edges = []
        for provider_npi, hospital_id in provider_hospital_map.items():
            provider_idx = self.provider_id_map[provider_npi]
            hospital_idx = self.hospital_id_map[hospital_id]
            provider_hospital_edges.append([provider_idx, hospital_idx])
        
        if len(provider_hospital_edges) > 0:
            edge_indices[('provider', 'works_at', 'hospital')] = torch.tensor(
                provider_hospital_edges, dtype=torch.long
            ).t().contiguous()
            
            # Reverse edges
            edge_indices[('hospital', 'employs', 'provider')] = torch.tensor(
                [[e[1], e[0]] for e in provider_hospital_edges], dtype=torch.long
            ).t().contiguous()
        
        # 5. Patient -> Hospital edges (through procedures)
        patient_hospital_map = defaultdict(set)
        for _, row in proc_data.iterrows():
            patient_id = row['patient_id']
            hospital_id = row['hospital_id']
            
            if (patient_id in self.patient_id_map and 
                pd.notna(hospital_id) and
                hospital_id in self.hospital_id_map):
                patient_hospital_map[patient_id].add(hospital_id)
        
        patient_hospital_edges = []
        for patient_id, hospitals in patient_hospital_map.items():
            patient_idx = self.patient_id_map[patient_id]
            for hospital_id in hospitals:
                hospital_idx = self.hospital_id_map[hospital_id]
                patient_hospital_edges.append([patient_idx, hospital_idx])
        
        if len(patient_hospital_edges) > 0:
            edge_indices[('patient', 'visits', 'hospital')] = torch.tensor(
                patient_hospital_edges, dtype=torch.long
            ).t().contiguous()
            
            # Reverse edges
            edge_indices[('hospital', 'visited_by', 'patient')] = torch.tensor(
                [[e[1], e[0]] for e in patient_hospital_edges], dtype=torch.long
            ).t().contiguous()
        
        # Print edge statistics
        if self.verbose:
            for edge_type, edge_index in edge_indices.items():
                print(f"  {edge_type}: {edge_index.shape[1]} edges")
        
        return edge_indices
    
    def build_hetero_data(
        self, 
        patient_ids: List,
        patient_features: pd.DataFrame,
        patient_labels: pd.DataFrame
    ) -> HeteroData:
        """
        Build complete HeteroData object
        
        Args:
            patient_ids: List of patient IDs to include
            patient_features: DataFrame with patient features
            patient_labels: DataFrame with patient labels
        
        Returns:
            HeteroData object
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("Building HeteroData object...")
            print("=" * 80)
        
        # Build node mappings
        self.build_node_mappings(patient_ids)
        
        # Create node features
        node_features = self.create_node_features(patient_features)
        
        # Build edges
        edge_indices = self.build_edges()
        
        # Create HeteroData object
        data = HeteroData()
        
        # Add node features
        for node_type, features in node_features.items():
            data[node_type].x = features
        
        # Add edges
        for edge_type, edge_index in edge_indices.items():
            data[edge_type].edge_index = edge_index
        
        # Add labels (only for patients)
        patient_label_list = []
        for patient_idx in range(len(self.patient_id_map)):
            patient_id = self.patient_idx_to_id[patient_idx]
            label_row = patient_labels[patient_labels['patient_id'] == patient_id]
            
            if len(label_row) > 0:
                label = label_row['ed_utilization_class'].values[0]
            else:
                label = -1  # Unknown label
            
            patient_label_list.append(label)
        
        data['patient'].y = torch.tensor(patient_label_list, dtype=torch.long)
        
        # Add train/val/test masks (will be set during cross-validation)
        data['patient'].train_mask = torch.zeros(len(self.patient_id_map), dtype=torch.bool)
        data['patient'].val_mask = torch.zeros(len(self.patient_id_map), dtype=torch.bool)
        data['patient'].test_mask = torch.zeros(len(self.patient_id_map), dtype=torch.bool)
        
        if self.verbose:
            print(f"\n✓ HeteroData object created successfully")
            print(f"  Node types: {data.node_types}")
            print(f"  Edge types: {len(data.edge_types)}")
            print(f"  Patient labels shape: {data['patient'].y.shape}")
        
        return data


def test_graph_builder():
    """Test the graph builder"""
    print("Testing HeterogeneousGraphBuilder...")
    
    # Load data
    loader = DataLoader(verbose=True)
    loader.load_all_data()
    patient_labels = loader.create_patient_labels()
    patient_features = loader.prepare_patient_features()
    
    # Build graph
    builder = HeterogeneousGraphBuilder(loader, verbose=True)
    
    # Use first 1000 patients for testing
    patient_ids = patient_labels['patient_id'].values[:1000]
    
    hetero_data = builder.build_hetero_data(
        patient_ids=patient_ids,
        patient_features=patient_features,
        patient_labels=patient_labels
    )
    
    print("\n" + "=" * 80)
    print("Graph Summary")
    print("=" * 80)
    print(hetero_data)
    
    print("\nGraph builder test completed successfully!")
    
    return builder, hetero_data


if __name__ == "__main__":
    test_graph_builder()
