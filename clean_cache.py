"""
Clean cached data to force reprocessing
Run this when you've fixed data parsing issues
"""

import shutil
from pathlib import Path

def clean_cache():
    """Remove all cached preprocessing, features, and graphs"""
    output_dir = Path("outputs")
    
    if output_dir.exists():
        print("🧹 Cleaning cached data...")
        
        files_to_remove = [
            'train_data.pkl',
            'val_data.pkl', 
            'test_data.pkl',
            'features.pkl',
            'graphs.pkl'
        ]
        
        removed = 0
        for file in files_to_remove:
            file_path = output_dir / file
            if file_path.exists():
                file_path.unlink()
                print(f"  ✓ Removed {file}")
                removed += 1
        
        if removed > 0:
            print(f"\n✅ Cleaned {removed} cached files")
            print("   Run 'python main.py' to reprocess with fixed code")
        else:
            print("  ℹ️  No cached files found")
    else:
        print("ℹ️  Output directory doesn't exist yet")

if __name__ == "__main__":
    clean_cache()

