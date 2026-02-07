#!/usr/bin/env python
"""
Database initialization and migration script.
This script sets up the database schema and optionally loads data from CSV.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from db import init_db, SessionLocal, Sample, Album
from model.utils.loader import populate_albums_from_db


def main():
    print("=" * 60)
    print("PetreeBlitz Database Initialization")
    print("=" * 60)
    
    # Initialize database tables
    print("\n1. Creating database tables...")
    try:
        init_db()
        print("✓ Database tables created successfully!")
    except Exception as e:
        print(f"✗ Error creating tables: {e}")
        return False
    
    # Load data from CSV if it exists and database is empty
    csv_path = 'data/tree_patches_with_clusters.csv'
    
    db = SessionLocal()
    try:
        existing_albums = db.query(Album).count()
        
        if existing_albums == 0 and os.path.exists(csv_path):
            print(f"\n2. Loading data from {csv_path}...")
            try:
                populate_albums_from_db(csv_path, db)
                
                # Count loaded data
                samples_count = db.query(Sample).count()
                albums_count = db.query(Album).count()
                
                print(f"✓ Successfully loaded {samples_count} samples into {albums_count} albums!")
            except Exception as e:
                print(f"✗ Error loading data: {e}")
                return False
        elif existing_albums > 0:
            print(f"\n2. Skipping CSV load: Database already contains {existing_albums} albums")
        else:
            print(f"\n2. Skipping CSV load: {csv_path} not found")
            print("   (You can run this script again after placing the CSV file)")
    finally:
        db.close()
    
    print("\n" + "=" * 60)
    print("Database initialization complete!")
    print("=" * 60)
    print("\nYou can now start the API with:")
    print("  python -m uvicorn api.main:app --reload")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

