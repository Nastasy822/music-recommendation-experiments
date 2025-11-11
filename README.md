# recsys-pet-project

This project is created to study, implement, and demonstrate different types of recommender systems.
It aims to show how recommendation algorithms can help users discover relevant content or products by analyzing their preferences and behavior.
The project covers the full workflow — from data preparation and model training to evaluation and deployment via an API.


# 🧩 Dataset Selection

## 1. Chosen Dataset

For this project, we use the **YAMBDA (Yandex Music Big Data Analytics)** dataset —  
a large-scale, real-world dataset of user interactions from the **Yandex.Music** streaming platform.  
It is publicly available through [🤗 Hugging Face Datasets](https://huggingface.co/datasets/yandex/yambda) and contains millions of anonymized user–track events.

---

## 2. Why This Dataset

The **YAMBDA** dataset was selected because it combines **realism, scale, and accessibility**, making it perfectly suited for research in recommendation systems and user modeling.

**Key reasons:**
- 📊 **Real user behavior:** based on genuine listening and “like” events from a real platform, not synthetic data.  
- 💾 **Large scale:** over **50 million** interactions, enabling testing of algorithms under realistic data loads.  
- 🕒 **Temporal data:** includes timestamps for each interaction, allowing time-based and session-aware modeling.  
- ⚙️ **Versatility:** suitable for both classical (ALS, collaborative filtering) and modern (deep learning) recommendation methods.  
- 🌍 **Open and reproducible:** available via Hugging Face for transparent, shareable experiments.

In short — YAMBDA offers both **data realism** and **industrial-scale complexity**, essential for evaluating recommender algorithms in practice.

---

## 3. Dataset Structure

The dataset is stored in **Parquet** format under `flat/50m/likes.parquet` and contains the following core fields:

| Column | Type | Description |
|---------|------|-------------|
| `user_id` | `int64` | Unique anonymized user identifier |
| `item_id` | `int64` | Unique track identifier |
| `timestamp` | `int64` | UNIX timestamp of the event |
| `event_type` | `string` | Type of interaction (e.g., `like`, `listen`, `skip`) |

Some variations may include additional metadata fields depending on the subset.

---


