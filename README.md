# Data Science

Coursework for **Introduction to Data Science** at the University of Tehran, Faculty of Electrical and Computer Engineering (Spring 2025, Dr. Bahrak & Dr. Yaghoobzadeh).

## Assignments

| # | Folder | What I built | Tech |
|---|--------|--------------|------|
| CA2 *(team)* | [Kafka & Spark streaming](ca2-kafka-spark-streaming/) | A real-time pipeline for a simulated payment network ("Darooghe"): a transaction generator feeding **Kafka**, validating and routing consumers, **Spark Structured Streaming** for fraud detection and live commission analytics, Spark batch jobs, storage in **MongoDB**, and monitoring with **Prometheus** | Kafka (KRaft), PySpark, MongoDB, Prometheus, Docker Compose |
| CA3 | [Classification, regression & recommendation](ca3-classification-regression-recommendation/) | Three prediction challenges: cancer-survival **classification** (EDA, target encoding, TF-IDF + SVD features, LightGBM/XGBoost); **regression** with log-target modelling, feature selection and a stacked CatBoost ensemble; a movie **recommender** tuning SVD, SVD++ and KNN-baseline models with the `surprise` library | scikit-learn, LightGBM, XGBoost, CatBoost, surprise |
| CA4 | [Deep learning](ca4-deep-learning-mlp-cnn-rnn/) | An **MLP** (PyTorch) that predicts FIFA World Cup 2022 match results; a **CNN** flower classifier, from a VGG-style network built from scratch to a fine-tuned **ResNet50** with data augmentation; an **RNN** forecasting Bitcoin prices | PyTorch, TensorFlow/Keras |
| CA5 & 6 | [SSL, semantic search, LLMs & segmentation](ca5-6-ssl-semantic-search-llm-segmentation/) | Four tasks: **semi-supervised** video-game review-score prediction (Word2Vec / sentence-embedding features, pseudo-labelling and uncertainty-based active learning); **Persian semantic search** over NiniSite Q&A (hazm preprocessing, bge-m3 embeddings in **LanceDB**, full-text vs. vector search, cross-encoder reranking); **BERT on SWAG** multiple choice, where **LoRA** fine-tuning lifts accuracy from 47% zero-shot to 69% (compared against few-shot and chain-of-thought prompting); unsupervised **image segmentation** of football players with K-Means, DBSCAN and agglomerative clustering on colour, position and ResNet features, scored with Dice and IoU | Hugging Face (Transformers, PEFT), LanceDB, sentence-transformers, hazm, scikit-learn, OpenCV |

## Final project – Ride-hailing demand forecasting

[final-project-ride-hailing-demand/](final-project-ride-hailing-demand/): *A Machine Learning Approach to Forecasting Ride-Hailing Demand Using Geospatial and Weather Data.* The project analysed **14.3 million NYC Uber trips** (January–June 2015) joined with weather data, then built and validated a demand-forecasting model. It found that Manhattan dominates the market. The folder contains the final presentation.

## Team

CA2 was done with **Mohammad Taha Majlesi** and **Mohammad Hossein Mazhari**.

## Notes

- CA2 needs a running Kafka broker: `docker compose up -d` in the CA2 folder, then start `darooghe_pulse.py` (producer) and the consumers. `transactions.jsonl` is a generated sample and isn't committed.
- CA5 & 6: the saved run of the notebook is missing outputs for Task 1.3 (the kernel restarted) and Tasks 2.3–2.4 (the embedding step needs a GPU). The code for those parts is complete; re-run them on a GPU runtime to regenerate the results.
- The datasets provided by the course (CA3 train/test files, `matches.csv`, `BTC-USD.csv`) aren't included. The CA4 flower dataset is downloaded automatically with `kagglehub`.
