## recsys-pet-project

This project is created to study, implement, and demonstrate different types of recommender systems.
It aims to show how recommendation algorithms can help users discover relevant content or products by analyzing their preferences and behavior.
The project covers the full workflow — from data preparation and model training to evaluation and deployment via an API.


### 📊 Dataset Selection

The dataset for this project was chosen to provide a realistic and diverse environment for building and evaluating recommender systems.  
It includes user–item interaction data (such as ratings, clicks, or reviews) and item metadata that can be used for both collaborative and content-based recommendation approaches.

When selecting a dataset, the main criteria were:

- **Realistic user behavior** — interactions that reflect genuine preferences  
- **Rich content features** — product descriptions, genres, categories, or textual reviews  
- **Sufficient scale** — enough users and items to train and validate models  
- **Public availability** — open access for reproducibility and learning  

Possible datasets that meet these requirements include:

- [**MovieLens**](https://grouplens.org/datasets/movielens/) — movie ratings and genres  
- [**Goodbooks-10k**](https://github.com/zygmuntz/goodbooks-10k) — book ratings with metadata  
- [**Amazon Product Reviews**](https://nijianmo.github.io/amazon/index.html) — product ratings and textual reviews  
- [**Yelp Dataset**](https://www.yelp.com/dataset) — business reviews with user feedback  

For demonstration purposes, a subset of the **Amazon Product Reviews** dataset can be used, as it provides both explicit ratings and textual information — allowing for a balanced comparison of different recommendation strategies.


