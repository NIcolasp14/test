"""
Helper script to clean cached outputs and force reprocessing
Run this before rerunning main.py to apply fixes
"""

import shutil
from pathlib import Path
import config

def clean_cache():
    """Remove cached preprocessed data, features, and graphs"""
    output_dir = Path(config.OUTPUT_DIR)
    
    if output_dir.exists():
        print(f"🗑️  Cleaning cache directory: {output_dir}")
        shutil.rmtree(output_dir)
        print("✓ Cache cleaned successfully.")
        print("\nYou can now run: python main.py")
    else:
        print(f"ℹ️  Cache directory not found: {output_dir}. Nothing to clean.")

if __name__ == "__main__":
    clean_cache()
