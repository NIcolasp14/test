#!/usr/bin/env python3
"""
Script to augment placeholder healthcare data while maintaining:
- ID correspondences across files
- Data patterns (empty cells, nan, ###, etc.)
- Realistic placeholder structure
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Configuration
NUM_MEMBERS = 150  # Number of unique member IDs to generate
INPUT_DIR = '/Users/nicolas/Desktop/lumeris/Files V2'
OUTPUT_DIR = '/Users/nicolas/Desktop/lumeris/Files V2/augmented'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_member_ids(n):
    """Generate pools of shared IDs"""
    sys_mbr_sk_pool = [random.randint(10000, 200000) for _ in range(n)]
    empi_pool = [random.randint(1000000, 11000000) for _ in range(n)]
    return sys_mbr_sk_pool, empi_pool


def random_date(start_year=2010, end_year=2025, as_hash=False):
    """Generate random date or ### placeholder"""
    if as_hash:
        return "#######"
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    date = start + timedelta(days=random_days)
    
    # Random format variations
    formats = ['%m/%d/%Y', '%m/%d/%y', '%-m/%-d/%Y', '%-m/%-d/%y']
    return date.strftime(random.choice(formats))


def maybe_empty(value, prob=0.2):
    """Return empty string with given probability"""
    return "" if random.random() < prob else value


def maybe_nan(value, prob=0.15):
    """Return 'nan' string with given probability"""
    return "nan" if random.random() < prob else value


def augment_demographics(sys_mbr_sk_pool, empi_pool, target_rows=50):
    """Augment demographics.csv"""
    print("Augmenting demographics.csv...")
    
    genders = ['M', 'F']
    client_ids = [2]  # Keep same as original
    
    rows = []
    for i, (sys_mbr_sk, empi) in enumerate(zip(sys_mbr_sk_pool[:target_rows], empi_pool[:target_rows])):
        # DOB with occasional ###
        dob = random_date(1920, 2010, as_hash=(random.random() < 0.15))
        
        # DOD is usually empty, occasionally has a date
        if random.random() < 0.1:
            dod = random_date(2010, 2025)
        else:
            dod = ""
        
        rows.append({
            'sys_mbr_sk': sys_mbr_sk,
            'mbr_gender_cd': random.choice(genders),
            'mbr_dob': dob,
            'mbr_dod': dod,
            'empi': empi,
            'client_id': random.choice(client_ids)
        })
    
    df = pd.DataFrame(rows)
    output_path = os.path.join(OUTPUT_DIR, 'demographics.csv')
    df.to_csv(output_path, index=True, index_label='')
    print(f"  Generated {len(df)} rows -> {output_path}")
    return df


def augment_diagnosis(sys_mbr_sk_pool, target_rows=100):
    """Augment diagnosis.csv"""
    print("Augmenting diagnosis.csv...")
    
    # Sample diagnosis patterns from original
    diagnoses = [
        ('54.1', 'Genital he Y', 'ID: HIV/ST', 'ID: Sexually Transmitted Diseases and Complications', 'ID: Sexually Transmitted Diseases and Complications'),
        ('1', 'Cholera due to vibrio cholerae', '', '', ''),
        ('401.9', 'Essential hypertension', 'CV: Hypertension', 'CV: Hypertension', 'CV: Hypertension'),
        ('250.00', 'Diabetes mellitus', 'EN: Diabetes', 'EN: Diabetes', 'EN: Diabetes'),
        ('272.4', 'Hyperlipidemia', '', '', ''),
        ('V76.12', 'Screening mammogram', '', '', ''),
        ('486', 'Pneumonia', 'RE: Pneumonia', 'RE: Pneumonia', 'RE: Pneumonia'),
    ]
    
    rows = []
    for i in range(target_rows):
        # Use sys_mbr_sk as clm_sys_mbr_sk (with correspondence)
        clm_sys_mbr_sk = random.choice(sys_mbr_sk_pool)
        
        # Date with occasional ###
        svc_dt = random_date(2010, 2024, as_hash=(random.random() < 0.25))
        
        diagnosis = random.choice(diagnoses)
        
        rows.append({
            'clm_sys_mbr_sk': clm_sys_mbr_sk,
            'type': 'diagnosis',
            'clm_claim_beg_svc_dt': svc_dt,
            'icd9_diagnosis_cd': diagnosis[0],
            'code_desc': diagnosis[1],
            'chronic_implctn_ind': diagnosis[2],
            'disease_cat': diagnosis[3],
            'clinical_grouping': diagnosis[4]
        })
    
    df = pd.DataFrame(rows)
    output_path = os.path.join(OUTPUT_DIR, 'diagnosis.csv')
    df.to_csv(output_path, index=True, index_label='')
    print(f"  Generated {len(df)} rows -> {output_path}")
    return df


def generate_procmapping_data():
    """Generate comprehensive procedure mapping data"""
    print("Generating procMapping.csv...")
    
    # CPT Codes - Office Visits and Common Procedures
    mappings = [
        # Office Visits
        ('9/23/2024', 'CPT', '99201', 'Office visit, new patient, level 1', 'Office Visits', 'Evaluation and Management', 'Office/Outpatient Services', '0', 'Standard care'),
        ('9/23/2024', 'CPT', '99202', 'Office visit, new patient, level 2', 'Office Visits', 'Evaluation and Management', 'Office/Outpatient Services', '0', 'Standard care'),
        ('9/23/2024', 'CPT', '99203', 'Office visit, new patient, level 3', 'Office Visits', 'Evaluation and Management', 'Office/Outpatient Services', '0', 'Standard care'),
        ('9/23/2024', 'CPT', '99204', 'Office visit, new patient, level 4', 'Office Visits', 'Evaluation and Management', 'Office/Outpatient Services', '0', 'Standard care'),
        ('9/23/2024', 'CPT', '99205', 'Office visit, new patient, level 5', 'Office Visits', 'Evaluation and Management', 'Office/Outpatient Services', '0', 'Standard care'),
        ('9/23/2024', 'CPT', '99211', 'Office visit, established patient, minimal', 'Office Visits', 'Evaluation and Management', 'Office/Outpatient Services', '0', 'Standard care'),
        ('9/23/2024', 'CPT', '99212', 'Office visit, established patient, level 2', 'Office Visits', 'Evaluation and Management', 'Office/Outpatient Services', '0', 'Standard care'),
        ('9/23/2024', 'CPT', '99213', 'Office visit, established patient, level 3', 'Office Visits', 'Evaluation and Management', 'Office/Outpatient Services', '0', 'Standard care'),
        ('9/23/2024', 'CPT', '99214', 'Office visit, established patient, level 4', 'Office Visits', 'Evaluation and Management', 'Office/Outpatient Services', '0', 'Standard care'),
        ('9/23/2024', 'CPT', '99215', 'Office visit, established patient, level 5', 'Office Visits', 'Evaluation and Management', 'Office/Outpatient Services', '0', 'Standard care'),
        
        # Preventive Visits
        ('9/23/2024', 'CPT', '99381', 'Preventive visit, new patient, infant', 'Prevention and Screening', 'Preventive Medicine', 'Preventive Medicine Services', '1', 'Preventive care'),
        ('9/23/2024', 'CPT', '99382', 'Preventive visit, new patient, age 1-4', 'Prevention and Screening', 'Preventive Medicine', 'Preventive Medicine Services', '1', 'Preventive care'),
        ('9/23/2024', 'CPT', '99383', 'Preventive visit, new patient, age 5-11', 'Prevention and Screening', 'Preventive Medicine', 'Preventive Medicine Services', '1', 'Preventive care'),
        ('9/23/2024', 'CPT', '99384', 'Preventive visit, new patient, age 12-17', 'Prevention and Screening', 'Preventive Medicine', 'Preventive Medicine Services', '1', 'Preventive care'),
        ('9/23/2024', 'CPT', '99385', 'Preventive visit, new patient, age 18-39', 'Prevention and Screening', 'Preventive Medicine', 'Preventive Medicine Services', '1', 'Preventive care'),
        ('9/23/2024', 'CPT', '99386', 'Preventive visit, new patient, age 40-64', 'Prevention and Screening', 'Preventive Medicine', 'Preventive Medicine Services', '1', 'Preventive care'),
        ('9/23/2024', 'CPT', '99387', 'Preventive visit, new patient, age 65+', 'Prevention and Screening', 'Preventive Medicine', 'Preventive Medicine Services', '1', 'Preventive care'),
        ('9/23/2024', 'CPT', '99391', 'Preventive visit, established patient, infant', 'Prevention and Screening', 'Preventive Medicine', 'Preventive Medicine Services', '1', 'Preventive care'),
        ('9/23/2024', 'CPT', '99392', 'Preventive visit, established patient, age 1-4', 'Prevention and Screening', 'Preventive Medicine', 'Preventive Medicine Services', '1', 'Preventive care'),
        ('9/23/2024', 'CPT', '99393', 'Preventive visit, established patient, age 5-11', 'Prevention and Screening', 'Preventive Medicine', 'Preventive Medicine Services', '1', 'Preventive care'),
        ('9/23/2024', 'CPT', '99394', 'Preventive visit, established patient, age 12-17', 'Prevention and Screening', 'Preventive Medicine', 'Preventive Medicine Services', '1', 'Preventive care'),
        ('9/23/2024', 'CPT', '99395', 'Preventive visit, established patient, age 18-39', 'Prevention and Screening', 'Preventive Medicine', 'Preventive Medicine Services', '1', 'Preventive care'),
        ('9/23/2024', 'CPT', '99396', 'Preventive visit, established patient, age 40-64', 'Prevention and Screening', 'Preventive Medicine', 'Preventive Medicine Services', '1', 'Preventive care'),
        ('9/23/2024', 'CPT', '99397', 'Preventive visit, established patient, age 65+', 'Prevention and Screening', 'Preventive Medicine', 'Preventive Medicine Services', '1', 'Preventive care'),
        
        # Lab Tests
        ('9/23/2024', 'CPT', '80053', 'Comprehensive metabolic panel', 'Testing', 'Laboratory', 'Chemistry', '1', 'Standard testing'),
        ('9/23/2024', 'CPT', '80061', 'Lipid panel', 'Testing', 'Laboratory', 'Chemistry', '1', 'Screening'),
        ('9/23/2024', 'CPT', '82947', 'Glucose, quantitative, blood', 'Testing', 'Laboratory', 'Chemistry', '1', 'Standard testing'),
        ('9/23/2024', 'CPT', '83036', 'Hemoglobin A1C', 'Testing', 'Laboratory', 'Chemistry', '1', 'Diabetes monitoring'),
        ('9/23/2024', 'CPT', '85025', 'Complete blood count', 'Testing', 'Laboratory', 'Hematology', '1', 'Standard testing'),
        ('9/23/2024', 'CPT', '85027', 'Complete blood count, automated', 'Testing', 'Laboratory', 'Hematology', '1', 'Standard testing'),
        ('9/23/2024', 'CPT', '81001', 'Urinalysis, automated', 'Testing', 'Laboratory', 'Urinalysis', '1', 'Standard testing'),
        ('9/23/2024', 'CPT', '84443', 'Thyroid stimulating hormone', 'Testing', 'Laboratory', 'Chemistry', '1', 'Standard testing'),
        
        # Imaging
        ('9/23/2024', 'CPT', '71045', 'Chest X-ray, single view', 'Testing', 'Radiology', 'Diagnostic Radiology', '0', 'Standard imaging'),
        ('9/23/2024', 'CPT', '71046', 'Chest X-ray, 2 views', 'Testing', 'Radiology', 'Diagnostic Radiology', '0', 'Standard imaging'),
        ('9/23/2024', 'CPT', '77080', 'Bone density study, DXA', 'Prevention and Screening', 'Radiology', 'Bone Density', '1', 'Osteoporosis screening'),
        ('9/23/2024', 'CPT', '77081', 'Bone density study, DXA, axial', 'Prevention and Screening', 'Radiology', 'Bone Density', '1', 'Osteoporosis screening'),
        
        # Mammography
        ('9/23/2024', 'CPT', '77065', 'Diagnostic mammography, unilateral', 'Prevention and Screening', 'Radiology', 'Breast Cancer Screening', '1', 'Breast cancer screening'),
        ('9/23/2024', 'CPT', '77066', 'Diagnostic mammography, bilateral', 'Prevention and Screening', 'Radiology', 'Breast Cancer Screening', '1', 'Breast cancer screening'),
        ('9/23/2024', 'CPT', '77067', 'Screening mammography, bilateral', 'Prevention and Screening', 'Radiology', 'Breast Cancer Screening', '1', 'Breast cancer screening'),
        
        # Cardiovascular
        ('9/23/2024', 'CPT', '93000', 'Electrocardiogram, complete', 'Testing', 'Cardiovascular', 'Diagnostic Cardiology', '0', 'Standard testing'),
        ('9/23/2024', 'CPT', '93005', 'Electrocardiogram, tracing only', 'Testing', 'Cardiovascular', 'Diagnostic Cardiology', '0', 'Standard testing'),
        ('9/23/2024', 'CPT', '93015', 'Cardiovascular stress test', 'Testing', 'Cardiovascular', 'Diagnostic Cardiology', '0', 'Cardiac evaluation'),
        ('9/23/2024', 'CPT', '93306', 'Echocardiography, complete', 'Testing', 'Cardiovascular', 'Diagnostic Cardiology', '0', 'Cardiac imaging'),
        
        # GI Procedures
        ('9/23/2024', 'CPT', '43239', 'Esophagogastroduodenoscopy, flexible', 'Testing', 'Gastroenterology', 'Endoscopy', '0', 'GI evaluation'),
        ('9/23/2024', 'CPT', '45378', 'Colonoscopy, flexible', 'Prevention and Screening', 'Gastroenterology', 'Colorectal Cancer Screening', '1', 'Colorectal cancer screening'),
        ('9/23/2024', 'CPT', '45380', 'Colonoscopy with biopsy', 'Prevention and Screening', 'Gastroenterology', 'Colorectal Cancer Screening', '1', 'Colorectal cancer screening'),
        ('9/23/2024', 'CPT', '45385', 'Colonoscopy with polyp removal', 'Prevention and Screening', 'Gastroenterology', 'Colorectal Cancer Screening', '1', 'Colorectal cancer screening'),
        
        # Immunizations - CVX codes
        ('9/23/2024', 'CVX', '08', 'Hepatitis B vaccine, pediatric', 'Immunizations', 'Hepatitis B', 'Hepatitis B Immunization', '1', 'Routine immunization'),
        ('9/23/2024', 'CVX', '20', 'Diphtheria, tetanus toxoids, pertussis vaccine', 'Immunizations', 'DTaP', 'DTaP Immunization', '1', 'Routine immunization'),
        ('9/23/2024', 'CVX', '21', 'Varicella vaccine', 'Immunizations', 'Varicella', 'Varicella Immunization', '1', 'Routine immunization'),
        ('9/23/2024', 'CVX', '33', 'Pneumococcal polysaccharide vaccine', 'Immunizations', 'Pneumococcal', 'Pneumococcal Immunization', '1', 'Routine immunization'),
        ('9/23/2024', 'CVX', '88', 'Influenza virus vaccine, unspecified', 'Immunizations', 'Influenza', 'Influenza Immunization', '1', 'Seasonal immunization'),
        ('9/23/2024', 'CVX', '94', 'Measles, mumps, rubella virus vaccine', 'Immunizations', 'MMR', 'MMR Immunization', '1', 'Routine immunization'),
        ('9/23/2024', 'CVX', '106', 'Diphtheria, tetanus, pertussis vaccine', 'Immunizations', 'DTaP', 'DTaP Immunization', '1', 'Routine immunization'),
        ('9/23/2024', 'CVX', '107', 'Diphtheria, tetanus toxoids vaccine', 'Immunizations', 'DTaP', 'DTaP Immunization', '1', 'Routine immunization'),
        ('9/23/2024', 'CVX', '113', 'Td (adult) vaccine', 'Immunizations', 'Td', 'Td Immunization', '1', 'Routine immunization'),
        ('9/23/2024', 'CVX', '115', 'Tdap vaccine', 'Immunizations', 'Tdap', 'Tdap Immunization', '1', 'Routine immunization'),
        ('9/23/2024', 'CVX', '140', 'Influenza A, seasonal, injectable', 'Immunizations', 'Influenza', 'Influenza Immunization', '1', 'Seasonal immunization'),
        ('9/23/2024', 'CVX', '141', 'Influenza A, seasonal, intranasal', 'Immunizations', 'Influenza', 'Influenza Immunization', '1', 'Seasonal immunization'),
        ('9/23/2024', 'CVX', '150', 'Influenza, injectable, preservative free', 'Immunizations', 'Influenza', 'Influenza Immunization', '1', 'Seasonal immunization'),
        ('9/23/2024', 'CVX', '152', 'Pneumococcal conjugate vaccine', 'Immunizations', 'Pneumococcal', 'Pneumococcal Immunization', '1', 'Routine immunization'),
        ('9/23/2024', 'CVX', '161', 'Influenza, injectable, preservative free', 'Immunizations', 'Influenza', 'Influenza Immunization', '1', 'Seasonal immunization'),
        ('9/23/2024', 'CVX', '171', 'Influenza, injectable, MDCK', 'Immunizations', 'Influenza', 'Influenza Immunization', '1', 'Seasonal immunization'),
        ('9/23/2024', 'CVX', '185', 'Seasonal influenza, recombinant', 'Immunizations', 'Influenza', 'Influenza Immunization', '1', 'Seasonal immunization'),
        ('9/23/2024', 'CVX', '197', 'Influenza, high dose seasonal', 'Immunizations', 'Influenza', 'Influenza Immunization', '1', 'Seasonal immunization'),
        ('9/23/2024', 'CVX', '200', 'Influenza, seasonal, Southern Hemisphere', 'Immunizations', 'Influenza', 'Influenza Immunization', '1', 'Seasonal immunization'),
        ('9/23/2024', 'CVX', '208', 'COVID-19, mRNA, unspecified', 'Immunizations', 'COVID-19', 'COVID-19 Immunization', '1', 'Pandemic immunization'),
        ('9/23/2024', 'CVX', '210', 'COVID-19 vaccine, vector-non-replicating', 'Immunizations', 'COVID-19', 'COVID-19 Immunization', '1', 'Pandemic immunization'),
        ('9/23/2024', 'CVX', '211', 'COVID-19, subunit', 'Immunizations', 'COVID-19', 'COVID-19 Immunization', '1', 'Pandemic immunization'),
        ('9/23/2024', 'CVX', '212', 'COVID-19 vaccine, vector-non-replicating', 'Immunizations', 'COVID-19', 'COVID-19 Immunization', '1', 'Pandemic immunization'),
        ('9/23/2024', 'CVX', '213', 'SARS-COV-2 (COVID-19) vaccine, UNSPECIFIED', 'Immunizations', 'COVID-19', 'COVID-19 Immunization', '1', 'Pandemic immunization'),
        ('9/23/2024', 'CVX', '300', 'COVID-19, mRNA, bivalent', 'Immunizations', 'COVID-19', 'COVID-19 Immunization', '1', 'Pandemic immunization'),
        ('9/23/2024', 'CVX', '301', 'COVID-19, mRNA, bivalent, preservative free', 'Immunizations', 'COVID-19', 'COVID-19 Immunization', '1', 'Pandemic immunization'),
        ('9/23/2024', 'CVX', '316', 'Meningococcal pentavalent vaccine', 'Immunizations', 'Meningococcal', 'Meningococcal Immunization', '1', 'NCQA added to value set'),
        
        # Additional Common Procedures
        ('9/23/2024', 'CPT', '90471', 'Immunization administration, first vaccine', 'Immunizations', 'Administration', 'Immunization Administration', '1', 'Vaccine admin'),
        ('9/23/2024', 'CPT', '90472', 'Immunization administration, each additional', 'Immunizations', 'Administration', 'Immunization Administration', '1', 'Vaccine admin'),
        ('9/23/2024', 'CPT', '90632', 'Hepatitis A vaccine, adult', 'Immunizations', 'Hepatitis A', 'Hepatitis A Immunization', '1', 'Routine immunization'),
        ('9/23/2024', 'CPT', '90633', 'Hepatitis A vaccine, pediatric', 'Immunizations', 'Hepatitis A', 'Hepatitis A Immunization', '1', 'Routine immunization'),
        ('9/23/2024', 'CPT', '90707', 'MMR vaccine', 'Immunizations', 'MMR', 'MMR Immunization', '1', 'Routine immunization'),
        ('9/23/2024', 'CPT', '90710', 'MMRV vaccine', 'Immunizations', 'MMRV', 'MMRV Immunization', '1', 'Routine immunization'),
        ('9/23/2024', 'CPT', '90714', 'Td vaccine', 'Immunizations', 'Td', 'Td Immunization', '1', 'Routine immunization'),
        ('9/23/2024', 'CPT', '90715', 'Tdap vaccine', 'Immunizations', 'Tdap', 'Tdap Immunization', '1', 'Routine immunization'),
        ('9/23/2024', 'CPT', '90732', 'Pneumococcal vaccine', 'Immunizations', 'Pneumococcal', 'Pneumococcal Immunization', '1', 'Routine immunization'),
        ('9/23/2024', 'CPT', '90746', 'Hepatitis B vaccine, adult', 'Immunizations', 'Hepatitis B', 'Hepatitis B Immunization', '1', 'Routine immunization'),
        ('9/23/2024', 'CPT', '91013', 'Esophageal motility study', 'Testing', 'Gastroenterology', 'GI Function Tests', '0', 'GI evaluation'),
        
        # Screenings
        ('9/23/2024', 'CPT', '81479', 'Molecular pathology procedure, unlisted', 'Testing', 'Pathology', 'Molecular Pathology', '0', 'Advanced testing'),
        ('9/23/2024', 'CPT', '82270', 'Occult blood, fecal', 'Prevention and Screening', 'Laboratory', 'Colorectal Cancer Screening', '1', 'Colorectal cancer screening'),
        ('9/23/2024', 'CPT', '82274', 'Fecal immunochemical test', 'Prevention and Screening', 'Laboratory', 'Colorectal Cancer Screening', '1', 'Colorectal cancer screening'),
        ('9/23/2024', 'CPT', '87624', 'HPV DNA test, high-risk types', 'Prevention and Screening', 'Laboratory', 'Cervical Cancer Screening', '1', 'Cervical cancer screening'),
        ('9/23/2024', 'CPT', '88141', 'Pap smear, cervical', 'Prevention and Screening', 'Pathology', 'Cervical Cancer Screening', '1', 'Cervical cancer screening'),
        ('9/23/2024', 'CPT', '88142', 'Pap smear, cervical, manual screen', 'Prevention and Screening', 'Pathology', 'Cervical Cancer Screening', '1', 'Cervical cancer screening'),
        ('9/23/2024', 'CPT', '88175', 'Pap smear, cervical, automated screen', 'Prevention and Screening', 'Pathology', 'Cervical Cancer Screening', '1', 'Cervical cancer screening'),
        
        # Mental Health
        ('9/23/2024', 'CPT', '90791', 'Psychiatric diagnostic evaluation', 'Behavioral Health', 'Mental Health', 'Mental Health Services', '0', 'Mental health evaluation'),
        ('9/23/2024', 'CPT', '90832', 'Psychotherapy, 30 minutes', 'Behavioral Health', 'Mental Health', 'Mental Health Services', '0', 'Mental health treatment'),
        ('9/23/2024', 'CPT', '90834', 'Psychotherapy, 45 minutes', 'Behavioral Health', 'Mental Health', 'Mental Health Services', '0', 'Mental health treatment'),
        ('9/23/2024', 'CPT', '90837', 'Psychotherapy, 60 minutes', 'Behavioral Health', 'Mental Health', 'Mental Health Services', '0', 'Mental health treatment'),
        
        # Physical Therapy
        ('9/23/2024', 'CPT', '97110', 'Therapeutic exercises', 'Rehabilitation', 'Physical Therapy', 'Therapy Services', '0', 'Physical therapy'),
        ('9/23/2024', 'CPT', '97112', 'Neuromuscular reeducation', 'Rehabilitation', 'Physical Therapy', 'Therapy Services', '0', 'Physical therapy'),
        ('9/23/2024', 'CPT', '97140', 'Manual therapy', 'Rehabilitation', 'Physical Therapy', 'Therapy Services', '0', 'Physical therapy'),
        ('9/23/2024', 'CPT', '97530', 'Therapeutic activities', 'Rehabilitation', 'Physical Therapy', 'Therapy Services', '0', 'Physical therapy'),
    ]
    
    rows = []
    for i, mapping in enumerate(mappings):
        rows.append({
            'Clinical Date Change': mapping[0],
            'CODE_TP_NM': mapping[1],
            'Code': mapping[2],
            'CODE_DESC': mapping[3],
            'LumerisEngage Profile Section': mapping[4],
            'LumerisEngage Value Set': mapping[5],
            'HEDIS Value Set Name': mapping[6],
            'Include in Prevention and Screening': mapping[7],
            'CHANGE NOTES': mapping[8]
        })
    
    df = pd.DataFrame(rows)
    output_path = os.path.join(OUTPUT_DIR, 'procMapping.csv')
    df.to_csv(output_path, sep=';', index=True, index_label='')
    print(f"  Generated {len(df)} procedure mappings -> {output_path}")
    return df


def augment_procedures(sys_mbr_sk_pool, proc_mapping_df, target_rows=100):
    """Augment procedures.csv using codes from procMapping"""
    print("Augmenting procedures.csv...")
    
    # Extract CPT codes from procMapping
    cpt_codes = proc_mapping_df[proc_mapping_df['CODE_TP_NM'] == 'CPT'][['Code', 'CODE_DESC', 'LumerisEngage Profile Section']].values.tolist()
    
    providers = [
        'VAID, BRIJ RAJ',
        'ST. ANTHONY\'S MEDICAL CENTER',
        'SMITH, JOHN',
        'QUEST DIAGNOSTICS',
        'BJC HEALTHCARE',
        'MERCY CLINIC',
        'SSM HEALTH',
        'BARNES JEWISH HOSPITAL',
        'WASH U PHYSICIANS',
        'ST. LUKE\'S HOSPITAL',
    ]
    
    rows = []
    for i in range(target_rows):
        sys_mbr_sk = random.choice(sys_mbr_sk_pool)
        claim_sk = random.randint(10000000, 20000000)
        svc_dt = random_date(2012, 2024)
        
        # Select a CPT code from mapping
        code_data = random.choice(cpt_codes)
        cpt_code = code_data[0]
        code_desc = code_data[1][:20] if len(code_data[1]) > 20 else code_data[1]  # Truncate like original
        profile_section = code_data[2] if random.random() > 0.6 else ''  # Sometimes empty
        
        # Randomly assign provider
        provider = random.choice(providers)
        
        # Hospital admission ID - mostly empty
        hosp_adm_id = str(random.randint(100000, 999999)) if random.random() < 0.3 else ''
        
        rows.append({
            'sys_mbr_sk': sys_mbr_sk,
            'type': 'procedure',
            'claim_sk': claim_sk,
            'svc_from_dt': svc_dt,
            'cpt_cd': cpt_code,
            'code_desc': code_desc,
            'prov_npi_full_nm': provider,
            'hosp_adm_id': hosp_adm_id,
            'LumerisEngage Profile Section': profile_section
        })
    
    df = pd.DataFrame(rows)
    output_path = os.path.join(OUTPUT_DIR, 'procedures.csv')
    df.to_csv(output_path, index=True, index_label='')
    print(f"  Generated {len(df)} rows -> {output_path}")
    return df


def augment_nyu_edu(sys_mbr_sk_pool, target_rows=60):
    """Augment nyu_edu.csv"""
    print("Augmenting nyu_edu.csv...")
    
    providers = [
        ('1960000000', 'SSM Health St Marys Hospital St Louis'),
        ('1630000000', 'HSHS St Elizabeths Hospital'),
        ('1600000000', 'SSM Health DePaul Hospital'),
        ('1740000000', 'Barnes Jewish Hospital'),
        ('1850000000', 'Mercy Hospital St Louis'),
    ]
    
    diagnoses = [
        'Tachycardia, unspecified',
        'COVID-19',
        'Chest pain',
        'Abdominal pain',
        'Shortness of breath',
        'Syncope',
    ]
    
    days_of_week = [1, 2, 3, 4, 5, 6, 7]
    
    rows = []
    for i in range(target_rows):
        sys_mbr_sk = random.choice(sys_mbr_sk_pool)
        provider = random.choice(providers)
        hosp_adm_id = random.randint(100000, 999999)
        
        # Generate month_date and hosp_adm_dt
        month_date = random_date(2023, 2025)
        hosp_adm_dt = random_date(2023, 2025)
        
        # Generate counts and amounts with many empties
        ed_count = random.choice([1, '', ''])
        ed_to_ip = random.choice([1, '', '', ''])
        paid_amount = random.choice([
            round(random.uniform(1000, 15000), 2),
            round(random.uniform(1000, 15000), 2),
            ''
        ])
        
        row = {
            'sys_mbr_sk': sys_mbr_sk,
            'month_date': month_date,
            'hosp_adm_id': hosp_adm_id,
            'billing_provider_npi_id': provider[0],
            'billing_provider_name': provider[1],
            'presenting_diagnosis': random.choice(diagnoses),
            'day_of_week': random.choice(days_of_week),
            'hosp_adm_dt': hosp_adm_dt,
            'enc_cs_accountable_vst_count': random.choice([1, 2, '']),
            'ed_count': ed_count,
            'ed_to_ip_transfer_count': ed_to_ip,
            'ed_to_ip_transfer_count_notmh': ed_to_ip if ed_to_ip else '',
            'ed_to_ip_transfer_count_mh': '',
            'outpat_ed_count': '',
            'ed_to_obs_count': random.choice(['', '', 1]),
            'lumeris_ed_avoidable_count': random.choice(['', '', 1]),
            'cs_ed_avoidable_count': '',
            'lumeris_ed_unavoidable_count': '',
            'cs_ed_unavoidable_count': '',
            'lumeris_ed_avoidable_count_home': '',
            'cs_ed_avoidable_count_home': '',
            'lumeris_ed_unavoidable_count_home': '',
            'cs_ed_unavoidable_count_home': '',
            'lumeris_ed_avoidable_count_obs': '',
            'cs_ed_avoidable_count_obs': '',
            'lumeris_ed_unavoidable_count_obs': '',
            'cs_ed_unavoidable_count_obs': '',
            'ed_amount': paid_amount,
            'ed_to_ip_transfer_amount': paid_amount if ed_to_ip else '',
            'ed_to_ip_transfer_amount_notmh': paid_amount if ed_to_ip else '',
            'ed_to_ip_transfer_amount_mh': '',
            'outpat_ed_amount': '',
            'ed_to_obs_amount': random.choice(['', '', round(random.uniform(1000, 15000), 2)]),
            'lumeris_ed_avoidable_amount': '',
            'cs_ed_avoidable_amount': '',
            'lumeris_ed_unavoidable_amount': '',
            'cs_ed_unavoidable_amount': '',
            'lumeris_ed_avoidable_amount_home': '',
            'cs_ed_avoidable_amount_home': '',
            'lumeris_ed_unavoidable_amount_home': '',
            'cs_ed_unavoidable_amount_home': '',
            'lumeris_ed_avoidable_amount_obs': '',
            'cs_ed_avoidable_amount_obs': '',
            'lumeris_ed_unavoidable_amount_obs': '',
            'cs_ed_unavoidable_amount_obs': '',
            'paid': paid_amount,
            'member_month_count': 1
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    output_path = os.path.join(OUTPUT_DIR, 'nyu_edu.csv')
    df.to_csv(output_path, index=True, index_label='')
    print(f"  Generated {len(df)} rows -> {output_path}")
    return df


def augment_sdoh(empi_pool, target_rows=50):
    """Augment sdoh.csv with many columns"""
    print("Augmenting sdoh.csv...")
    
    # Read original to get column names
    original = pd.read_csv(os.path.join(INPUT_DIR, 'sdoh.csv'), sep=';', index_col=0)
    columns = original.columns.tolist()
    
    # Dictionary of value ranges for different types of columns
    rank_values = list(range(1, 21))
    rank_100_values = list(range(1, 101))
    binary_values = [0, 1, '']
    small_int_values = list(range(0, 10))
    float_values = [round(random.uniform(-100, 100), 3) for _ in range(20)]
    
    rows = []
    for i, empi in enumerate(empi_pool[:target_rows]):
        row = {'lumeris_empi': empi}
        
        for col in columns[1:]:  # Skip lumeris_empi as we already set it
            col_lower = col.lower()
            
            # Determine value type based on column name patterns
            if 'rank_base_20' in col_lower:
                value = maybe_nan(random.choice(rank_values))
            elif 'rank_base_100' in col_lower:
                value = maybe_nan(random.choice(rank_100_values))
            elif 'rank_base_10' in col_lower or 'rank_base_5' in col_lower:
                value = maybe_nan(random.choice(small_int_values))
            elif '_onezero' in col_lower or 'indicator' in col_lower:
                value = maybe_nan(random.choice(binary_values))
            elif 'latitude' in col_lower:
                value = round(random.uniform(37.0, 42.0), 6)
            elif 'longitude' in col_lower:
                value = round(random.uniform(-95.0, -88.0), 6)
            elif 'sdh_agg' in col_lower:
                value = maybe_nan(random.choice([-10, -8, -5, -3, 0, 2, 4, 5, 8, 10]))
            elif 'income' in col_lower:
                value = maybe_nan(random.choice(list(range(1, 10))))
            elif 'household' in col_lower or 'size' in col_lower:
                value = maybe_nan(random.choice([1, 2, 3, 4, 5]))
            elif 'children' in col_lower:
                value = maybe_nan(random.choice([0, 1, 2, 3, '', '']))
            else:
                # Default: mix of small integers and nans
                value = maybe_nan(random.choice(small_int_values + ['', '']))
            
            row[col] = value
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    output_path = os.path.join(OUTPUT_DIR, 'sdoh.csv')
    df.to_csv(output_path, sep=';', index=True, index_label='')
    print(f"  Generated {len(df)} rows -> {output_path}")
    return df


def main():
    print("="*60)
    print("Healthcare Data Augmentation Script")
    print("="*60)
    print(f"Generating data for {NUM_MEMBERS} members...")
    print()
    
    # Generate shared ID pools
    sys_mbr_sk_pool, empi_pool = generate_member_ids(NUM_MEMBERS)
    
    # Generate procedure mapping first (needed for procedures.csv)
    proc_mapping_df = generate_procmapping_data()
    
    # Augment each file with at least 100 rows
    augment_demographics(sys_mbr_sk_pool, empi_pool, target_rows=max(100, NUM_MEMBERS))
    augment_diagnosis(sys_mbr_sk_pool, target_rows=max(100, int(NUM_MEMBERS * 0.8)))
    augment_procedures(sys_mbr_sk_pool, proc_mapping_df, target_rows=max(100, int(NUM_MEMBERS * 0.8)))
    augment_nyu_edu(sys_mbr_sk_pool, target_rows=max(100, int(NUM_MEMBERS * 0.7)))
    augment_sdoh(empi_pool, target_rows=max(100, NUM_MEMBERS))
    
    print()
    print("="*60)
    print(f"✓ All files augmented successfully!")
    print(f"✓ Output directory: {OUTPUT_DIR}")
    print("="*60)
    print()
    print("Files generated (all with 100+ rows):")
    print(f"  - procMapping.csv: {len(proc_mapping_df)} procedure/vaccine codes")
    print(f"  - demographics.csv: {max(100, NUM_MEMBERS)} members")
    print(f"  - diagnosis.csv: {max(100, int(NUM_MEMBERS * 0.8))} diagnoses")
    print(f"  - procedures.csv: {max(100, int(NUM_MEMBERS * 0.8))} procedures")
    print(f"  - nyu_edu.csv: {max(100, int(NUM_MEMBERS * 0.7))} ED visits")
    print(f"  - sdoh.csv: {max(100, NUM_MEMBERS)} SDOH records")
    print()
    print("ID Correspondences maintained:")
    print(f"  - sys_mbr_sk pool: {len(sys_mbr_sk_pool)} unique IDs")
    print(f"  - empi pool: {len(empi_pool)} unique IDs")
    print(f"  - All procedure codes in procedures.csv exist in procMapping.csv")
    print()
    print("Pattern preservation:")
    print("  ✓ Empty cells (varying by column)")
    print("  ✓ 'nan' values (especially in sdoh.csv)")
    print("  ✓ '###' instead of dates (~15-25% of dates)")
    print("  ✓ Various date formats")
    print("  ✓ Realistic value distributions from procMapping reference")


if __name__ == "__main__":
    main()

