#!/usr/bin/env python
"""
Database utility script for common operations.
Usage: python scripts/db_utils.py [command]

Commands:
  init          - Initialize database (same as init_database.py)
  clear-all     - Delete all data and recreate tables
  stats         - Show database statistics
  backup        - Backup database to backup directory
  restore       - Restore database from backup
  reset         - Reset to initial state with CSV data
  list-albums   - List all albums
  list-samples  - List all samples with details
"""

import os
import sys
import shutil
import pickle
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db import (
    init_db, SessionLocal, engine, Base, 
    Album, Sample, Feedback, Embedding
)
from model.utils.loader import populate_albums_from_db


def init_database():
    """Initialize database with schema."""
    print("Initializing database...")
    init_db()
    
    csv_path = 'data/tree_patches_with_clusters.csv'
    if os.path.exists(csv_path):
        print(f"Loading data from {csv_path}...")
        db = SessionLocal()
        try:
            populate_albums_from_db(csv_path, db)
            samples_count = db.query(Sample).count()
            albums_count = db.query(Album).count()
            print(f"✓ Loaded {samples_count} samples into {albums_count} albums")
        finally:
            db.close()
    else:
        print(f"⚠ {csv_path} not found, database is empty")
    
    print("✓ Database initialized")


def clear_database():
    """Delete all data and reinitialize database."""
    print("⚠ WARNING: This will delete all data!")
    confirm = input("Type 'yes' to confirm: ")
    
    if confirm != "yes":
        print("Cancelled")
        return
    
    print("Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    print("Recreating tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Database cleared and reinitialized")


def show_stats():
    """Display database statistics."""
    db = SessionLocal()
    try:
        samples_count = db.query(Sample).count()
        albums_count = db.query(Album).count()
        feedback_count = db.query(Feedback).count()
        embeddings_count = db.query(Embedding).count()
        learned_embeddings = db.query(Embedding).filter(Embedding.is_learned == True).count()
        
        print("\n" + "=" * 50)
        print("Database Statistics")
        print("=" * 50)
        print(f"Albums:                  {albums_count:,}")
        print(f"Samples (total):         {samples_count:,}")
        print(f"Embeddings:              {embeddings_count:,}")
        print(f"  - Original:            {embeddings_count - learned_embeddings:,}")
        print(f"  - Learned from feedback: {learned_embeddings:,}")
        print(f"Feedback entries:        {feedback_count:,}")
        print("=" * 50)
        
    finally:
        db.close()


def backup_database():
    """Backup database file."""
    backup_dir = Path("backups")
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_file = Path("petree_blitz.db")
    
    if not db_file.exists():
        print("✗ Database file not found")
        return
    
    backup_file = backup_dir / f"petree_blitz_{timestamp}.db"
    shutil.copy2(db_file, backup_file)
    print(f"✓ Database backed up to {backup_file}")


def restore_database():
    """Restore database from most recent backup."""
    backup_dir = Path("backups")
    
    if not backup_dir.exists():
        print("✗ No backup directory found")
        return
    
    backups = sorted(backup_dir.glob("petree_blitz_*.db"))
    if not backups:
        print("✗ No backup files found")
        return
    
    latest_backup = backups[-1]
    print(f"Latest backup: {latest_backup}")
    
    confirm = input("Restore from this backup? (yes/no): ")
    if confirm != "yes":
        print("Cancelled")
        return
    
    db_file = Path("petree_blitz.db")
    shutil.copy2(latest_backup, db_file)
    print(f"✓ Database restored from {latest_backup}")


def reset_database():
    """Reset database to initial state."""
    print("Resetting database to initial state...")
    clear_database()
    print("Loading initial data from CSV...")
    init_database()
    print("✓ Database reset complete")


def list_albums():
    """List all albums."""
    db = SessionLocal()
    try:
        albums = db.query(Album).all()
        
        if not albums:
            print("No albums found")
            return
        
        print("\n" + "=" * 70)
        print(f"{'Album ID':<30} {'Album Name':<20} {'Samples':<10}")
        print("=" * 70)
        
        for album in albums:
            samples_count = db.query(Sample).filter(Sample.album_id == album.album_id).count()
            print(f"{album.album_id:<30} {album.name:<20} {samples_count:<10}")
        
        print("=" * 70)
    finally:
        db.close()


def list_samples():
    """List all samples with details."""
    db = SessionLocal()
    try:
        samples = db.query(Sample).order_by(Sample.timestamp.desc()).all()
        
        if not samples:
            print("No samples found")
            return
        
        print("\n" + "=" * 100)
        print(f"{'Sample ID':<38} {'Album':<20} {'Image':<8} {'Feedback':<10} {'Timestamp':<24}")
        print("=" * 100)
        
        for sample in samples[:100]:  # Show first 100
            has_image = "Yes" if sample.image_bytes else "No"
            has_feedback = "Yes" if sample.feedback else "No"
            timestamp = sample.timestamp.strftime("%Y-%m-%d %H:%M:%S") if sample.timestamp else "N/A"
            print(f"{sample.sample_id:<38} {sample.album_id or 'N/A':<20} {has_image:<8} {has_feedback:<10} {timestamp:<24}")
        
        if len(samples) > 100:
            print(f"... and {len(samples) - 100} more")
        
        print("=" * 100)
    finally:
        db.close()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1].lower()
    
    commands = {
        "init": init_database,
        "clear-all": clear_database,
        "stats": show_stats,
        "backup": backup_database,
        "restore": restore_database,
        "reset": reset_database,
        "list-albums": list_albums,
        "list-samples": list_samples,
    }
    
    if command not in commands:
        print(f"Unknown command: {command}")
        print(__doc__)
        return
    
    try:
        commands[command]()
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
