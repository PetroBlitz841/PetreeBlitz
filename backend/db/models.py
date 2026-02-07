"""Database models using SQLAlchemy ORM."""

from sqlalchemy import Column, String, Integer, DateTime, LargeBinary, Boolean, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

from .config import Base


# ===== Database Models =====

class Album(Base):
    """Album model for storing tree species/categories."""
    __tablename__ = "albums"

    album_id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    samples = relationship("Sample", back_populates="album")
    embeddings = relationship("Embedding", back_populates="album")


class Sample(Base):
    """Sample model for storing identified images."""
    __tablename__ = "samples"

    sample_id = Column(String, primary_key=True, index=True)
    album_id = Column(String, ForeignKey("albums.album_id"))
    image_bytes = Column(LargeBinary, nullable=True)  # Raw image data
    image_path = Column(String, nullable=True)  # Path to original image file
    predictions = Column(JSON, nullable=True)  # List of predictions: [{"label": "...", "confidence": 0.9}]
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    album = relationship("Album", back_populates="samples")
    feedback = relationship("Feedback", back_populates="sample", uselist=False, cascade="all, delete-orphan")


class Feedback(Base):
    """Feedback model for storing user corrections."""
    __tablename__ = "feedback"

    feedback_id = Column(String, primary_key=True, index=True)
    sample_id = Column(String, ForeignKey("samples.sample_id"), unique=True)
    was_correct = Column(Boolean, nullable=False)
    correct_label = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    sample = relationship("Sample", back_populates="feedback")


class Embedding(Base):
    """Embedding model for storing computed embeddings."""
    __tablename__ = "embeddings"

    embedding_id = Column(String, primary_key=True, index=True)
    album_id = Column(String, ForeignKey("albums.album_id"))
    original_sample_id = Column(String, nullable=True)  # Reference to original sample if learned from feedback
    embedding_vector = Column(LargeBinary, nullable=False)  # Serialized numpy array
    embedding_dim = Column(Integer, nullable=False)  # Dimension of embedding
    is_learned = Column(Boolean, default=False)  # Whether this is from learned feedback
    amplification_iteration = Column(Integer, default=0)  # For amplified learning
    patch_index = Column(Integer, nullable=True)  # Which patch this embedding is from (0-15)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    album = relationship("Album", back_populates="embeddings")
