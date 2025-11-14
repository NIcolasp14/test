"""
Pipeline Verification Script
Checks that all modules are structured correctly without running them
"""

import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if Path(filepath).exists():
        print(f"  ✓ {description}: {filepath}")
        return True
    else:
        print(f"  ✗ {description}: {filepath} NOT FOUND")
        return False

def check_data_files():
    """Check input data files"""
    print("\n1. Checking Input Data Files:")
    print("="*60)
    
    data_dir = Path("Files V2/augmented")
    files = [
        ('demographics.csv', 'Demographics data'),
        ('diagnosis.csv', 'Diagnosis data'),
        ('procedures.csv', 'Procedures data'),
        ('nyu_edu.csv', 'ED utilization data'),
        ('sdoh.csv', 'SDOH data'),
        ('procMapping.csv', 'Procedure mapping'),
    ]
    
    all_exist = True
    for filename, desc in files:
        filepath = data_dir / filename
        if not check_file_exists(filepath, desc):
            all_exist = False
    
    return all_exist

def check_pipeline_modules():
    """Check pipeline module files"""
    print("\n2. Checking Pipeline Modules:")
    print("="*60)
    
    modules = [
        ('config.py', 'Configuration'),
        ('data_preprocessing.py', 'Data preprocessing'),
        ('feature_engineering.py', 'Feature engineering'),
        ('graph_construction.py', 'Graph construction'),
        ('models.py', 'Model definitions'),
        ('train.py', 'Training pipeline'),
        ('evaluate.py', 'Evaluation metrics'),
        ('main.py', 'Main execution'),
    ]
    
    all_exist = True
    for filename, desc in modules:
        if not check_file_exists(filename, desc):
            all_exist = False
    
    return all_exist

def check_module_structure():
    """Check basic module structure by importing (without dependencies)"""
    print("\n3. Checking Module Structure:")
    print("="*60)
    
    # Read and check basic structure of each module
    checks = []
    
    # Check config.py has key variables
    with open('config.py', 'r') as f:
        config_content = f.read()
        has_hyperparams = 'HIDDEN_DIM' in config_content and 'LEARNING_RATE' in config_content
        checks.append(('config.py has hyperparameters', has_hyperparams))
    
    # Check data_preprocessing.py has main functions
    with open('data_preprocessing.py', 'r') as f:
        preproc_content = f.read()
        has_functions = 'def preprocess_pipeline' in preproc_content and 'def time_based_split' in preproc_content
        checks.append(('data_preprocessing.py has required functions', has_functions))
    
    # Check models.py has model classes
    with open('models.py', 'r') as f:
        models_content = f.read()
        has_models = 'class SimpleTGN' in models_content and 'class TGAT' in models_content and 'class HGTModel' in models_content
        checks.append(('models.py has TGN, TGAT, HGT', has_models))
    
    # Check train.py has training loop
    with open('train.py', 'r') as f:
        train_content = f.read()
        has_training = 'def train_model' in train_content and 'def train_epoch' in train_content
        checks.append(('train.py has training functions', has_training))
    
    # Check evaluate.py has metrics
    with open('evaluate.py', 'r') as f:
        eval_content = f.read()
        has_metrics = 'def concordance_index' in eval_content and 'def compute_auroc' in eval_content
        checks.append(('evaluate.py has evaluation metrics', has_metrics))
    
    # Check main.py has pipeline
    with open('main.py', 'r') as f:
        main_content = f.read()
        has_pipeline = 'def main' in main_content and 'def train_all_models' in main_content
        checks.append(('main.py has pipeline orchestration', has_pipeline))
    
    all_pass = True
    for check_name, result in checks:
        if result:
            print(f"  ✓ {check_name}")
        else:
            print(f"  ✗ {check_name}")
            all_pass = False
    
    return all_pass

def check_documentation():
    """Check documentation files"""
    print("\n4. Checking Documentation:")
    print("="*60)
    
    docs = [
        ('requirements.txt', 'Requirements file'),
        ('README_PIPELINE.md', 'Pipeline README'),
        ('README_AUGMENTATION.md', 'Data README'),
    ]
    
    all_exist = True
    for filename, desc in docs:
        if not check_file_exists(filename, desc):
            all_exist = False
    
    return all_exist

def print_summary(data_ok, modules_ok, structure_ok, docs_ok):
    """Print summary"""
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    results = [
        ("Input Data", data_ok),
        ("Pipeline Modules", modules_ok),
        ("Module Structure", structure_ok),
        ("Documentation", docs_ok),
    ]
    
    all_ok = all(r[1] for r in results)
    
    for name, status in results:
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"  {name:20s}: {status_str}")
    
    print("="*60)
    
    if all_ok:
        print("\n🎉 All checks passed! Pipeline is ready.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run pipeline: python main.py")
        print("  3. Check results in outputs/, models/, and results/ directories")
    else:
        print("\n⚠️  Some checks failed. Please review the issues above.")
    
    return all_ok

def main():
    """Run all verifications"""
    print("\n" + "="*60)
    print("ED UTILIZATION PIPELINE VERIFICATION")
    print("="*60)
    
    data_ok = check_data_files()
    modules_ok = check_pipeline_modules()
    structure_ok = check_module_structure()
    docs_ok = check_documentation()
    
    all_ok = print_summary(data_ok, modules_ok, structure_ok, docs_ok)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())


