<div align="center">
  <a href="https://github.com/PetroBlitz841/PetreeBlitz">
    <img src="frontend/public/pt-logo.svg" alt="PetreeBlitz Logo" width="300" height="100">
  </a>
  <p>A charcoal identification system used for Archaeobotany research.</p>
</div>

# PetreeBlitz

A web-based tree species identification system that uses deep learning (ResNet18) and wood anatomy features (IAWA) to classify hardwood species from microscopy cross-section images.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Database Schema](#database-schema)
- [Example Data](#example-data)

---

## Overview

PetreeBlitz identifies tree species from wood cross-section microscopy images. The system receives photos of the three sampling planes and returns predictions for species.

Users can provide feedback to correct identifications and feature detections, improving accuracy over time through **collaborative learning**.

## Features

- **Species Identification** — Upload a wood cross-section image and receive ranked predictions with confidence scores
- **IAWA Feature Analysis** — Simulating detection of extracted 27 wood anatomy features with per-species support breakdowns
- **Feedback & Learning** — Correct identifications and feature detections to iteratively improve the model
- **Taxonomic Browser** — Browse species albums in a grid or a taxonomic hierarchy (Order → Family → Genus → Species)
- **Album Details** — View all identified samples per species with taxonomy breadcrumbs and Wikipedia links
- **Multi-Plane Support** — Upload images from traverse, radial-longitudinal, and tangential-longitudinal planes
- **Region Filtering** — Filter species by TDWG geographic regions
- **TIFF Support** — Automatic TIFF-to-PNG conversion for browser display

## Tech Stack

| Layer        | Technology                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Frontend** | ![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB) ![TypeScript](https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white) ![MaterialUI](https://img.shields.io/badge/Material%20UI-%23FFFFFF?style=for-the-badge&logo=MUI&logoColor=#007FFF) ![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white) ![React Router](https://img.shields.io/badge/React_Router-CA4245?style=for-the-badge&logo=react-router&logoColor=white) |
| **Backend**  | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi) ![SQLAlchemy](https://img.shields.io/badge/sqlalchemy-%23D71F00.svg?style=for-the-badge&logo=sqlalchemy&logoColor=white)                                                                                                                                                                                                                                                                 |
| **Database** | ![SQLite](https://img.shields.io/badge/sqlite-%2307405e.svg?style=for-the-badge&logo=sqlite&logoColor=white)                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| **ML Model** | ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) torchvision (ResNet18, ImageNet pretrained)                                                                                                                                                                                                                                                                                                                                                                                                                              |

## Project Structure

```
PetreeBlitz/
├── package.json          # Root scripts (dev, init-db, run-model)
│
├── backend/
│   ├── api/              # FastAPI routes and CORS config
│   ├── db/               # SQLAlchemy models and DB config
│   ├── model/            # ML inference, preprocessing, clustering, learning
│   ├── scripts/          # DB initialisation and clustering pipeline
│   └── data/             # Source images, patches, and CSV data
│
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── layout/   # Header, Footer
│       │   ├── identify/ # Upload, results, features, feedback, settings
│       │   ├── albums/   # Album cards, image cards, detail dialogs
│       │   ├── taxonomy/ # Taxonomy tree, breadcrumb, rank chips
│       │   └── common/   # Shared components (WikiLink)
│       ├── pages/        # IdentifyPage, AlbumsPage, AlbumDetailsPage
│       ├── hooks/        # Custom React hooks
│       ├── services/     # API client
│       └── utils/        # Taxonomy data, styling helpers, region data
│
├── example/              # 44 reference species folders
└── generated/            # UMAP visualisation outputs
```

## Prerequisites

- **Python** 3.10+
- **Node.js** 18+ and **npm**

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/PetroBlitz841/PetreeBlitz.git
cd PetreeBlitz
```

### 2. Set up the backend

```bash
cd backend
python -m venv .venv

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
cd ..
```

### 3. Set up the frontend

```bash
cd frontend
npm install
cd ..
```

### 4. Install root dependencies

```bash
npm install
```

### 5. Run the ML pipeline (first time only)

Place source microscopy images (≥896×896) in `backend/data/Trees/`, then:

```bash
npm run run-model
```

This will preprocess images into patches, compute embeddings, run clustering, and save the results to `backend/data/tree_patches_with_clusters.csv`.

### 6. Initialise the database

```bash
npm run init-db
```

Creates the SQLite database and seeds it from the generated CSV.

## Usage

### Development mode

Start both backend and frontend concurrently:

```bash
npm run dev
```

Or run them separately:

```bash
npm run dev:backend    # FastAPI on http://localhost:8000
npm run dev:frontend   # Vite on http://localhost:5173
```

## Database Schema

View the database schema in [Database Schema.md](./Database%20Schema.md)

## Example Data

The `example/` directory contains reference specimens for **44 tropical hardwood species** spanning 12 orders and 27 genera, primarily from Amazonian forests. Species include genera such as _Apuleia_, _Aspidosperma_, _Byrsonima_, _Cecropia_, _Copaifera_, _Dipteryx_, _Hymenaea_, _Ocotea_, _Qualea_, _Vochysia_, and others.
The data was downloaded from the linked [Charcoal Dataset](https://zenodo.org/records/10214399) captured using SEM microscopy in TIFF format.

> Maruyama, T., Oliveira, L. S., Britto Jr, Nisgoski, S., Automatic Classification of Native Wood Charcoal, Ecological Informatics, 48:1-7, 2018.
