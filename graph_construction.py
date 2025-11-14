"""
Heterogeneous Temporal Graph Construction
Builds PyTorch Geometric heterogeneous graphs with temporal edges
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, Tuple
from collections import defaultdict
import config

try:
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    print("Warning: PyTorch Geometric not installed. Please install: pip install torch-geometric")
    HAS_PYG = False


class HeteroGraphBuilder:
    """Build heterogeneous temporal graphs for ED prediction"""
    
    def __init__(self, features: Dict):
        self.features = features
        # Use nested defaultdicts for proper initialization
        self.node_id_maps = defaultdict(lambda: defaultdict(dict))  # Map original IDs to graph node IDs
        self.reverse_node_maps = defaultdict(lambda: defaultdict(dict))  # Reverse mapping
        
    def build_graph(self, split_data: Dict, labels_df: pd.DataFrame, split_name: str):
        """
        Build heterogeneous graph for one split
        
        Node types: patient, visit, dx_code, proc_code, provider, hospital, sdoh
        Edge types: See config.EDGE_TYPES
        """
        print(f"\nBuilding {split_name} graph...")
        
        # Initialize node ID counters
        node_counts = {ntype: 0 for ntype in config.NODE_TYPES}
        
        # Edge lists for each edge type
        edge_dict = {}
        edge_timestamps = {}
        
        # =====================================================================
        # 1. Create node ID mappings
        # =====================================================================
        
        # Patient nodes
        patient_ids = sorted(split_data['patient_ids'])
        for idx, pid in enumerate(patient_ids):
            self.node_id_maps[split_name]['patient'][pid] = idx
            self.reverse_node_maps[split_name]['patient'][idx] = pid
        node_counts['patient'] = len(patient_ids)
        
        # Diagnosis code nodes
        dx_codes = set()
        for _, row in split_data['diagnosis'].iterrows():
            code = f"ICD_{row['icd9_diagnosis_cd']}"
            dx_codes.add(code)
        dx_codes.add(config.UNK_TOKEN)
        for idx, code in enumerate(sorted(dx_codes)):
            self.node_id_maps[split_name]['dx_code'][code] = idx
            self.reverse_node_maps[split_name]['dx_code'][idx] = code
        node_counts['dx_code'] = len(dx_codes)
        
        # Procedure code nodes
        proc_codes = set()
        for _, row in split_data['procedures'].iterrows():
            code = f"CPT_{row['cpt_cd']}"
            proc_codes.add(code)
        proc_codes.add(config.UNK_TOKEN)
        for idx, code in enumerate(sorted(proc_codes)):
            self.node_id_maps[split_name]['proc_code'][code] = idx
            self.reverse_node_maps[split_name]['proc_code'][idx] = code
        node_counts['proc_code'] = len(proc_codes)
        
        # Provider nodes
        providers = set(split_data['procedures']['prov_npi_full_nm'].dropna().unique())
        providers.add(config.UNK_TOKEN)
        for idx, provider in enumerate(sorted(providers)):
            self.node_id_maps[split_name]['provider'][provider] = idx
            self.reverse_node_maps[split_name]['provider'][idx] = provider
        node_counts['provider'] = len(providers)
        
        # Hospital nodes
        hospitals = set(split_data['nyu_edu']['billing_provider_name'].dropna().unique())
        hospitals.add(config.UNK_TOKEN)
        for idx, hospital in enumerate(sorted(hospitals)):
            self.node_id_maps[split_name]['hospital'][hospital] = idx
            self.reverse_node_maps[split_name]['hospital'][idx] = hospital
        node_counts['hospital'] = len(hospitals)
        
        # Visit nodes (one per ED visit + implicit visits from dx/proc)
        visit_counter = 0
        visit_map = {}
        
        # Create unique visit IDs for each event
        for _, row in split_data['diagnosis'].iterrows():
            visit_id = f"visit_dx_{row.name}_{row['clm_sys_mbr_sk']}_{row['timestamp']}"
            if visit_id not in visit_map:
                visit_map[visit_id] = visit_counter
                visit_counter += 1
        
        for _, row in split_data['procedures'].iterrows():
            visit_id = f"visit_proc_{row.name}_{row['sys_mbr_sk']}_{row['timestamp']}"
            if visit_id not in visit_map:
                visit_map[visit_id] = visit_counter
                visit_counter += 1
        
        for _, row in split_data['nyu_edu'].iterrows():
            visit_id = f"visit_ed_{row.name}_{row['sys_mbr_sk']}_{row['timestamp']}"
            if visit_id not in visit_map:
                visit_map[visit_id] = visit_counter
                visit_counter += 1
        
        node_counts['visit'] = visit_counter
        
        # SDOH nodes (one per patient)
        sdoh_counter = 0
        sdoh_map = {}
        for empi in split_data['sdoh']['lumeris_empi'].unique():
            sdoh_map[empi] = sdoh_counter
            sdoh_counter += 1
        node_counts['sdoh'] = sdoh_counter
        
        print(f"  Node counts: {node_counts}")
        
        # =====================================================================
        # 2. Build edges
        # =====================================================================
        
        # (patient)-[has_visit]->(visit)
        has_visit_src, has_visit_dst, has_visit_ts = [], [], []
        
        # From diagnoses
        for _, row in split_data['diagnosis'].iterrows():
            patient_id = row['clm_sys_mbr_sk']
            if patient_id not in self.node_id_maps[split_name]['patient']:
                continue
            
            visit_id = f"visit_dx_{row.name}_{patient_id}_{row['timestamp']}"
            patient_nid = self.node_id_maps[split_name]['patient'][patient_id]
            visit_nid = visit_map[visit_id]
            
            has_visit_src.append(patient_nid)
            has_visit_dst.append(visit_nid)
            has_visit_ts.append(row['timestamp'].timestamp())
        
        # From procedures
        for _, row in split_data['procedures'].iterrows():
            patient_id = row['sys_mbr_sk']
            if patient_id not in self.node_id_maps[split_name]['patient']:
                continue
            
            visit_id = f"visit_proc_{row.name}_{patient_id}_{row['timestamp']}"
            patient_nid = self.node_id_maps[split_name]['patient'][patient_id]
            visit_nid = visit_map[visit_id]
            
            has_visit_src.append(patient_nid)
            has_visit_dst.append(visit_nid)
            has_visit_ts.append(row['timestamp'].timestamp())
        
        # From ED visits
        for _, row in split_data['nyu_edu'].iterrows():
            patient_id = row['sys_mbr_sk']
            if patient_id not in self.node_id_maps[split_name]['patient']:
                continue
            
            visit_id = f"visit_ed_{row.name}_{patient_id}_{row['timestamp']}"
            patient_nid = self.node_id_maps[split_name]['patient'][patient_id]
            visit_nid = visit_map[visit_id]
            
            has_visit_src.append(patient_nid)
            has_visit_dst.append(visit_nid)
            has_visit_ts.append(row['timestamp'].timestamp())
        
        edge_dict[('patient', 'has_visit', 'visit')] = (
            torch.tensor(has_visit_src, dtype=torch.long),
            torch.tensor(has_visit_dst, dtype=torch.long)
        )
        edge_timestamps[('patient', 'has_visit', 'visit')] = torch.tensor(has_visit_ts, dtype=torch.float32)
        
        # (visit)-[has_diagnosis]->(dx_code)
        has_diag_src, has_diag_dst = [], []
        for _, row in split_data['diagnosis'].iterrows():
            patient_id = row['clm_sys_mbr_sk']
            if patient_id not in self.node_id_maps[split_name]['patient']:
                continue
            
            visit_id = f"visit_dx_{row.name}_{patient_id}_{row['timestamp']}"
            visit_nid = visit_map[visit_id]
            
            dx_code = f"ICD_{row['icd9_diagnosis_cd']}"
            if dx_code not in self.node_id_maps[split_name]['dx_code']:
                dx_code = config.UNK_TOKEN
            dx_nid = self.node_id_maps[split_name]['dx_code'][dx_code]
            
            has_diag_src.append(visit_nid)
            has_diag_dst.append(dx_nid)
        
        edge_dict[('visit', 'has_diagnosis', 'dx_code')] = (
            torch.tensor(has_diag_src, dtype=torch.long),
            torch.tensor(has_diag_dst, dtype=torch.long)
        )
        
        # (visit)-[has_procedure]->(proc_code)
        has_proc_src, has_proc_dst = [], []
        for _, row in split_data['procedures'].iterrows():
            patient_id = row['sys_mbr_sk']
            if patient_id not in self.node_id_maps[split_name]['patient']:
                continue
            
            visit_id = f"visit_proc_{row.name}_{patient_id}_{row['timestamp']}"
            visit_nid = visit_map[visit_id]
            
            proc_code = f"CPT_{row['cpt_cd']}"
            if proc_code not in self.node_id_maps[split_name]['proc_code']:
                proc_code = config.UNK_TOKEN
            proc_nid = self.node_id_maps[split_name]['proc_code'][proc_code]
            
            has_proc_src.append(visit_nid)
            has_proc_dst.append(proc_nid)
        
        edge_dict[('visit', 'has_procedure', 'proc_code')] = (
            torch.tensor(has_proc_src, dtype=torch.long),
            torch.tensor(has_proc_dst, dtype=torch.long)
        )
        
        # (proc_code)-[performed_by]->(provider)
        performed_src, performed_dst = [], []
        for _, row in split_data['procedures'].iterrows():
            proc_code = f"CPT_{row['cpt_cd']}"
            if proc_code not in self.node_id_maps[split_name]['proc_code']:
                proc_code = config.UNK_TOKEN
            proc_nid = self.node_id_maps[split_name]['proc_code'][proc_code]
            
            provider = row['prov_npi_full_nm']
            if pd.isna(provider) or provider not in self.node_id_maps[split_name]['provider']:
                provider = config.UNK_TOKEN
            provider_nid = self.node_id_maps[split_name]['provider'][provider]
            
            performed_src.append(proc_nid)
            performed_dst.append(provider_nid)
        
        if performed_src:
            edge_dict[('proc_code', 'performed_by', 'provider')] = (
                torch.tensor(performed_src, dtype=torch.long),
                torch.tensor(performed_dst, dtype=torch.long)
            )
        
        # (provider)-[works_at]->(hospital)
        works_src, works_dst = [], []
        for _, row in split_data['nyu_edu'].iterrows():
            # Simplified: connect billing provider (as hospital) - in real data, map providers to hospitals
            hospital = row['billing_provider_name']
            if pd.isna(hospital) or hospital not in self.node_id_maps[split_name]['hospital']:
                hospital = config.UNK_TOKEN
            hospital_nid = self.node_id_maps[split_name]['hospital'][hospital]
            
            # For simplicity, create dummy provider->hospital edges
            # In real implementation, you'd have actual provider-hospital mappings
            for provider in list(self.node_id_maps[split_name]['provider'].keys())[:5]:
                provider_nid = self.node_id_maps[split_name]['provider'][provider]
                works_src.append(provider_nid)
                works_dst.append(hospital_nid)
        
        if works_src:
            edge_dict[('provider', 'works_at', 'hospital')] = (
                torch.tensor(works_src, dtype=torch.long),
                torch.tensor(works_dst, dtype=torch.long)
            )
        
        # (patient)-[has_sdoh]->(sdoh) - static edges
        has_sdoh_src, has_sdoh_dst = [], []
        for _, patient_row in split_data['demographics'].iterrows():
            patient_id = patient_row['sys_mbr_sk']
            empi = patient_row['empi']
            
            if patient_id not in self.node_id_maps[split_name]['patient']:
                continue
            if empi not in sdoh_map:
                continue
            
            patient_nid = self.node_id_maps[split_name]['patient'][patient_id]
            sdoh_nid = sdoh_map[empi]
            
            has_sdoh_src.append(patient_nid)
            has_sdoh_dst.append(sdoh_nid)
        
        if has_sdoh_src:
            edge_dict[('patient', 'has_sdoh', 'sdoh')] = (
                torch.tensor(has_sdoh_src, dtype=torch.long),
                torch.tensor(has_sdoh_dst, dtype=torch.long)
            )
        
        print(f"  Edge counts:")
        for etype, (src, dst) in edge_dict.items():
            print(f"    {etype}: {len(src)}")
        
        # =====================================================================
        # 3. Create node features
        # =====================================================================
        
        node_features = {}
        
        # Patient features
        patient_feat_list = []
        for pid in patient_ids:
            if split_name == 'train':
                feat = self.features['train_patient_features'].get(pid)
            elif split_name == 'val':
                feat = self.features['val_patient_features'].get(pid)
            else:
                feat = self.features['test_patient_features'].get(pid)
            
            if feat is None:
                feat = torch.zeros(config.PROJECTED_DIM)
            patient_feat_list.append(feat)
        
        node_features['patient'] = torch.stack(patient_feat_list)
        
        # Dx code features
        dx_feat_list = []
        for code in sorted(dx_codes):
            feat = self.features['code_embeddings'].get(code)
            if feat is None:
                feat = self.features['code_embeddings'][config.UNK_TOKEN]
            dx_feat_list.append(feat)
        node_features['dx_code'] = torch.stack(dx_feat_list)
        
        # Proc code features
        proc_feat_list = []
        for code in sorted(proc_codes):
            feat = self.features['code_embeddings'].get(code)
            if feat is None:
                feat = self.features['code_embeddings'][config.UNK_TOKEN]
            proc_feat_list.append(feat)
        node_features['proc_code'] = torch.stack(proc_feat_list)
        
        # Provider features
        provider_feat_list = []
        for provider in sorted(providers):
            feat = self.features['provider_embeddings'].get(provider)
            if feat is None:
                feat = self.features['provider_embeddings'][config.UNK_TOKEN]
            provider_feat_list.append(feat)
        node_features['provider'] = torch.stack(provider_feat_list)
        
        # Hospital features
        hospital_feat_list = []
        for hospital in sorted(hospitals):
            feat = self.features['hospital_embeddings'].get(hospital)
            if feat is None:
                feat = self.features['hospital_embeddings'][config.UNK_TOKEN]
            hospital_feat_list.append(feat)
        node_features['hospital'] = torch.stack(hospital_feat_list)
        
        # Visit features (aggregate from connected nodes - simple zero init for now)
        node_features['visit'] = torch.zeros((node_counts['visit'], config.PROJECTED_DIM))
        
        # SDOH features (simple zero init for now)
        node_features['sdoh'] = torch.zeros((node_counts['sdoh'], config.PROJECTED_DIM))
        
        print(f"  Node feature shapes:")
        for ntype, feat in node_features.items():
            print(f"    {ntype}: {feat.shape}")
        
        # =====================================================================
        # 4. Build PyTorch Geometric HeteroData graph
        # =====================================================================
        
        if not HAS_PYG:
            print("  ✗ PyTorch Geometric not available, skipping graph construction")
            return None, node_features, edge_timestamps
        
        graph = HeteroData()
        
        # Add node features
        for ntype, feat in node_features.items():
            graph[ntype].x = feat
            graph[ntype].num_nodes = node_counts[ntype]
        
        # Add edges and edge attributes
        for etype, (src, dst) in edge_dict.items():
            # PyG expects edge_index as (2, num_edges)
            edge_index = torch.stack([src, dst], dim=0)
            graph[etype].edge_index = edge_index
            
            # Add timestamps as edge attributes if available
            if etype in edge_timestamps:
                graph[etype].edge_attr = edge_timestamps[etype].unsqueeze(-1)
        
        print(f"  ✓ Graph created with {len(graph.node_types)} node types and {len(graph.edge_types)} edge types")
        
        return graph, node_features, edge_timestamps


def build_all_graphs():
    """Build graphs for all splits"""
    print("="*80)
    print("GRAPH CONSTRUCTION PIPELINE")
    print("="*80)
    
    # Load data
    output_dir = Path(config.OUTPUT_DIR)
    
    with open(output_dir / 'train_data.pkl', 'rb') as f:
        train_data, train_labels = pickle.load(f)
    with open(output_dir / 'val_data.pkl', 'rb') as f:
        val_data, val_labels = pickle.load(f)
    with open(output_dir / 'test_data.pkl', 'rb') as f:
        test_data, test_labels = pickle.load(f)
    with open(output_dir / 'features.pkl', 'rb') as f:
        features = pickle.load(f)
    
    # Build graphs
    builder = HeteroGraphBuilder(features)
    
    train_graph, train_node_feats, train_edge_ts = builder.build_graph(
        train_data, train_labels, 'train'
    )
    val_graph, val_node_feats, val_edge_ts = builder.build_graph(
        val_data, val_labels, 'val'
    )
    test_graph, test_node_feats, test_edge_ts = builder.build_graph(
        test_data, test_labels, 'test'
    )
    
    # Save graphs and mappings
    print("\nSaving graphs...")
    graphs = {
        'train_graph': train_graph,
        'val_graph': val_graph,
        'test_graph': test_graph,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'test_labels': test_labels,
        'node_id_maps': dict(builder.node_id_maps),
        'reverse_node_maps': dict(builder.reverse_node_maps),
    }
    
    with open(output_dir / 'graphs.pkl', 'wb') as f:
        pickle.dump(graphs, f)
    
    print("  ✓ Graphs saved")
    print("\n" + "="*80)
    print("GRAPH CONSTRUCTION COMPLETE")
    print("="*80)
    
    return graphs


if __name__ == "__main__":
    graphs = build_all_graphs()


