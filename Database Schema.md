# Database Schema

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│    Album     │     │    Sample    │     │    Feedback      │
├──────────────┤     ├──────────────┤     ├──────────────────┤
│ album_id PK  │◄────│ album_id FK  │     │ feedback_id PK   │
│ name         │     │ sample_id PK │◄────│ sample_id FK (1:1│)
│ created_at   │     │ image_bytes  │     │ was_correct      │
│ updated_at   │     │ image_path   │     │ correct_label    │
└──────────────┘     │ predictions  │     │ created_at       │
                     │ iawa_features│     └──────────────────┘
                     │ timestamp    │
                     └──────┬───────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                                       │
┌───────┴────────┐                  ┌───────────┴──────────┐
│   Embedding    │                  │  FeatureFeedback     │
├────────────────┤                  ├──────────────────────┤
│ embedding_id PK│                  │ id PK                │
│ album_id FK    │                  │ sample_id FK         │
│ original_sample│                  │ feature_id           │
│ embedding_vec  │                  │ is_present           │
│ embedding_dim  │                  │ importance_weight    │
│ is_learned     │                  │ created_at           │
│ weight         │                  └──────────────────────┘
│ patch_index    │
│ created_at     │
└────────────────┘
```
