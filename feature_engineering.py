"""
Feature Engineering with LLM Embeddings
Creates node features using BioGPT embeddings for medical codes
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, Tuple
from tqdm import tqdm
import config

# Try to import transformers, use fallback if not available
try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    print("Warning: transformers not installed. Using random embeddings.")
    HAS_TRANSFORMERS = False


class FeatureExtractor:
    """Extract and manage features for all node types"""
    
    def __init__(self, use_llm=True):
        self.use_llm = use_llm and HAS_TRANSFORMERS
        self.device = config.DEVICE
        self.projection = nn.Linear(config.LLM_EMBEDDING_DIM, config.PROJECTED_DIM)
        nn.init.kaiming_normal_(self.projection.weight)
        self.projection = self.projection.to(self.device)
        
        # Cache for embeddings
        self.cache_dir = Path(config.OUTPUT_DIR) / 'embeddings_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM if available
        if self.use_llm:
            print(f"Loading LLM: {config.LLM_MODEL}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)
                self.model = AutoModel.from_pretrained(config.LLM_MODEL).eval().to(self.device)
                print("  ✓ LLM loaded successfully")
            except Exception as e:
                print(f"  ✗ Failed to load LLM: {e}")
                print("  Using random embeddings instead")
                self.use_llm = False
        
        # Storage for all embeddings
        self.code_embeddings = {}
        self.provider_embeddings = {}
        self.hospital_embeddings = {}
        
    def get_cls_embedding(self, text: str) -> torch.Tensor:
        """Get CLS token embedding from LLM"""
        if not self.use_llm or text is None or str(text).strip() == '':
            # Return random embedding
            return torch.randn(config.LLM_EMBEDDING_DIM)
        
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    str(text), 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512
                ).to(self.device)
                outputs = self.model(**inputs)
                # Get CLS token (first token)
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().squeeze()
            return cls_embedding
        except Exception as e:
            print(f"Error getting embedding for '{text}': {e}")
            return torch.randn(config.LLM_EMBEDDING_DIM)
    
    def project_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Project embedding to target dimension"""
        with torch.no_grad():
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            projected = self.projection(embedding.to(self.device)).cpu().squeeze()
        return projected
    
    def extract_code_embeddings(self, proc_mapping_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings for all diagnosis and procedure codes
        """
        print("\nExtracting code embeddings from procMapping...")
        
        cache_file = self.cache_dir / 'code_embeddings.pkl'
        
        if config.USE_CACHED_EMBEDDINGS and cache_file.exists():
            print("  Loading cached embeddings...")
            with open(cache_file, 'rb') as f:
                self.code_embeddings = pickle.load(f)
            print(f"  ✓ Loaded {len(self.code_embeddings)} cached code embeddings")
            return self.code_embeddings
        
        code_embeddings = {}
        
        # Add UNK token
        unk_emb = self.project_embedding(torch.randn(config.LLM_EMBEDDING_DIM))
        code_embeddings[config.UNK_TOKEN] = unk_emb
        
        # Process CPT codes
        cpt_codes = proc_mapping_df[proc_mapping_df['CODE_TP_NM'] == 'CPT']
        print(f"  Processing {len(cpt_codes)} CPT codes...")
        for _, row in tqdm(cpt_codes.iterrows(), total=len(cpt_codes), desc="CPT"):
            code = str(row['Code'])
            desc = str(row['CODE_DESC'])
            
            # Create text: "CPT code: description"
            text = f"CPT {code}: {desc}"
            emb = self.get_cls_embedding(text)
            code_embeddings[f"CPT_{code}"] = self.project_embedding(emb)
        
        # Process CVX codes (vaccine codes)
        cvx_codes = proc_mapping_df[proc_mapping_df['CODE_TP_NM'] == 'CVX']
        print(f"  Processing {len(cvx_codes)} CVX codes...")
        for _, row in tqdm(cvx_codes.iterrows(), total=len(cvx_codes), desc="CVX"):
            code = str(row['Code'])
            desc = str(row['CODE_DESC'])
            
            text = f"CVX {code}: {desc}"
            emb = self.get_cls_embedding(text)
            code_embeddings[f"CVX_{code}"] = self.project_embedding(emb)
        
        # Add common diagnosis codes (from placeholder data)
        print("  Processing diagnosis codes...")
        common_dx_codes = {
            '54.1': 'Genital herpes',
            '1': 'Cholera',
            '401.9': 'Essential hypertension',
            '250.00': 'Diabetes mellitus',
            '272.4': 'Hyperlipidemia',
            'V76.12': 'Screening mammogram',
            '486': 'Pneumonia',
        }
        
        for code, desc in common_dx_codes.items():
            text = f"ICD-9 {code}: {desc}"
            emb = self.get_cls_embedding(text)
            code_embeddings[f"ICD_{code}"] = self.project_embedding(emb)
        
        self.code_embeddings = code_embeddings
        
        # Cache embeddings
        print(f"  Caching {len(code_embeddings)} embeddings...")
        with open(cache_file, 'wb') as f:
            pickle.dump(code_embeddings, f)
        
        print(f"  ✓ Extracted {len(code_embeddings)} code embeddings")
        return code_embeddings
    
    def extract_provider_embeddings(self, procedures_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Extract embeddings for providers"""
        print("\nExtracting provider embeddings...")
        
        cache_file = self.cache_dir / 'provider_embeddings.pkl'
        
        if config.USE_CACHED_EMBEDDINGS and cache_file.exists():
            print("  Loading cached embeddings...")
            with open(cache_file, 'rb') as f:
                self.provider_embeddings = pickle.load(f)
            print(f"  ✓ Loaded {len(self.provider_embeddings)} cached provider embeddings")
            return self.provider_embeddings
        
        provider_embeddings = {}
        
        # Add UNK token
        provider_embeddings[config.UNK_TOKEN] = self.project_embedding(
            torch.randn(config.LLM_EMBEDDING_DIM)
        )
        
        unique_providers = procedures_df['prov_npi_full_nm'].dropna().unique()
        print(f"  Processing {len(unique_providers)} providers...")
        
        for provider in tqdm(unique_providers, desc="Providers"):
            text = f"Healthcare provider: {provider}"
            emb = self.get_cls_embedding(text)
            provider_embeddings[str(provider)] = self.project_embedding(emb)
        
        self.provider_embeddings = provider_embeddings
        
        # Cache
        with open(cache_file, 'wb') as f:
            pickle.dump(provider_embeddings, f)
        
        print(f"  ✓ Extracted {len(provider_embeddings)} provider embeddings")
        return provider_embeddings
    
    def extract_hospital_embeddings(self, nyu_edu_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Extract embeddings for hospitals"""
        print("\nExtracting hospital embeddings...")
        
        cache_file = self.cache_dir / 'hospital_embeddings.pkl'
        
        if config.USE_CACHED_EMBEDDINGS and cache_file.exists():
            print("  Loading cached embeddings...")
            with open(cache_file, 'rb') as f:
                self.hospital_embeddings = pickle.load(f)
            print(f"  ✓ Loaded {len(self.hospital_embeddings)} cached hospital embeddings")
            return self.hospital_embeddings
        
        hospital_embeddings = {}
        
        # Add UNK token
        hospital_embeddings[config.UNK_TOKEN] = self.project_embedding(
            torch.randn(config.LLM_EMBEDDING_DIM)
        )
        
        unique_hospitals = nyu_edu_df['billing_provider_name'].dropna().unique()
        print(f"  Processing {len(unique_hospitals)} hospitals...")
        
        for hospital in tqdm(unique_hospitals, desc="Hospitals"):
            text = f"Hospital: {hospital}"
            emb = self.get_cls_embedding(text)
            hospital_embeddings[str(hospital)] = self.project_embedding(emb)
        
        self.hospital_embeddings = hospital_embeddings
        
        # Cache
        with open(cache_file, 'wb') as f:
            pickle.dump(hospital_embeddings, f)
        
        print(f"  ✓ Extracted {len(hospital_embeddings)} hospital embeddings")
        return hospital_embeddings
    
    def create_patient_features(self, demographics_df: pd.DataFrame, 
                                sdoh_df: pd.DataFrame,
                                reference_date=None) -> Dict[int, torch.Tensor]:
        """
        Create features for patient nodes
        Combines demographics and SDOH features
        """
        print("\nCreating patient features...")
        
        patient_features = {}
        
        for _, patient_row in demographics_df.iterrows():
            patient_id = patient_row['sys_mbr_sk']
            empi = patient_row['empi']
            
            # Demographics features
            # Age at reference date
            if reference_date and pd.notna(patient_row.get('dob')):
                age = (reference_date - patient_row['dob']).days / 365.25
            else:
                age = 50.0  # Default age if missing
            
            # Gender encoding
            gender = 1.0 if patient_row['mbr_gender_cd'] == 'M' else 0.0
            
            # Get SDOH features
            sdoh_row = sdoh_df[sdoh_df['lumeris_empi'] == empi]
            
            if len(sdoh_row) > 0:
                sdoh_row = sdoh_row.iloc[0]
                # Extract numerical SDOH features (sample subset)
                sdoh_feats = []
                for col in ['sdh_agg_transportation', 'sdh_agg_affordability', 
                           'sdh_agg_health_beh', 'sdh_agg_housing_sup']:
                    val = sdoh_row.get(col, 0)
                    if pd.isna(val) or val == 'nan':
                        val = 0.0
                    sdoh_feats.append(float(val))
            else:
                sdoh_feats = [0.0, 0.0, 0.0, 0.0]
            
            # Combine features
            feat_vec = torch.tensor([age, gender] + sdoh_feats, dtype=torch.float32)
            
            # Pad or project to PROJECTED_DIM
            if len(feat_vec) < config.PROJECTED_DIM:
                padding = torch.zeros(config.PROJECTED_DIM - len(feat_vec))
                feat_vec = torch.cat([feat_vec, padding])
            else:
                feat_vec = feat_vec[:config.PROJECTED_DIM]
            
            patient_features[patient_id] = feat_vec
        
        print(f"  ✓ Created features for {len(patient_features)} patients")
        return patient_features


def extract_all_features(train_data: Dict, val_data: Dict, test_data: Dict):
    """
    Main feature extraction pipeline
    Extract features from all splits
    """
    print("="*80)
    print("FEATURE EXTRACTION PIPELINE")
    print("="*80)
    
    extractor = FeatureExtractor(use_llm=True)
    
    # Extract code embeddings (shared across splits)
    code_embeddings = extractor.extract_code_embeddings(train_data['procMapping'])
    
    # Extract provider embeddings (from train only, to prevent leakage)
    provider_embeddings = extractor.extract_provider_embeddings(train_data['procedures'])
    
    # Extract hospital embeddings (from train only)
    hospital_embeddings = extractor.extract_hospital_embeddings(train_data['nyu_edu'])
    
    # Create patient features for each split
    train_patient_feats = extractor.create_patient_features(
        train_data['demographics'], train_data['sdoh']
    )
    val_patient_feats = extractor.create_patient_features(
        val_data['demographics'], val_data['sdoh']
    )
    test_patient_feats = extractor.create_patient_features(
        test_data['demographics'], test_data['sdoh']
    )
    
    # Save all features
    output_dir = Path(config.OUTPUT_DIR)
    features = {
        'code_embeddings': code_embeddings,
        'provider_embeddings': provider_embeddings,
        'hospital_embeddings': hospital_embeddings,
        'train_patient_features': train_patient_feats,
        'val_patient_features': val_patient_feats,
        'test_patient_features': test_patient_feats,
        'projection_layer': extractor.projection.state_dict()
    }
    
    print("\nSaving features...")
    with open(output_dir / 'features.pkl', 'wb') as f:
        pickle.dump(features, f)
    
    print("  ✓ Features saved")
    print("\n" + "="*80)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*80)
    
    return features


if __name__ == "__main__":
    import pickle
    
    # Load preprocessed data
    with open(Path(config.OUTPUT_DIR) / 'train_data.pkl', 'rb') as f:
        train_data, _ = pickle.load(f)
    with open(Path(config.OUTPUT_DIR) / 'val_data.pkl', 'rb') as f:
        val_data, _ = pickle.load(f)
    with open(Path(config.OUTPUT_DIR) / 'test_data.pkl', 'rb') as f:
        test_data, _ = pickle.load(f)
    
    # Extract features
    features = extract_all_features(train_data, val_data, test_data)


