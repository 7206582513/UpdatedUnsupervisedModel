import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

import umap
import hdbscan

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

def build_autoencoder(input_dim, encoding_dim=5):
    input_layer = Input(shape=(input_dim,))
    x = Dense(32, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)
    encoded = Dense(encoding_dim, activation='relu')(x)

    x = Dense(16, activation='relu')(encoded)
    x = BatchNormalization()(x)
    decoded = Dense(input_dim, activation='linear')(x)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder, encoder

def evaluate_model(model_name, embedding, n_clusters=None):
    try:
        if model_name == "HDBSCAN":
            model = hdbscan.HDBSCAN(min_cluster_size=10)
            labels = model.fit_predict(embedding)
        elif model_name == "KMeans":
            model = KMeans(n_clusters=n_clusters, random_state=42)
            labels = model.fit_predict(embedding)
        elif model_name == "Agglomerative":
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(embedding)
        elif model_name == "DBSCAN":
            model = DBSCAN(eps=0.5, min_samples=5)
            labels = model.fit_predict(embedding)
        else:
            return -1, []

        if len(set(labels)) <= 1:
            return -1, labels

        mask = labels != -1
        if np.sum(mask) < 2:
            return -1, labels

        score = silhouette_score(embedding[mask], np.array(labels)[mask])
        return round(score, 4), labels
    except:
        return -1, []

def explain_model_choice(best_model, all_scores):
    reasons = {
        "HDBSCAN": "HDBSCAN performed best because it automatically determined the number of clusters, handled variable density, and removed noisy points effectively.",
        "KMeans": "KMeans worked best likely because the customer segments were spherical and well-separated, and the data was pre-scaled and PCA-encoded.",
        "Agglomerative": "Agglomerative Clustering performed best because of hierarchical relationships in the data that matched well with the underlying patterns.",
        "DBSCAN": "DBSCAN performed best likely due to well-defined density regions and clear spatial separations in 2D space."
    }
    if best_model in reasons:
        return reasons[best_model]
    return "The selected model gave the highest silhouette score compared to other algorithms."

def process_and_cluster(filepath):
    df = pd.read_csv(filepath)
    df['balance_utilization'] = df['current_balance'] / (df['credit_limit'] + 1e-5)

    # Outlier Removal
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # PCA
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    # Autoencoder
    autoencoder, encoder = build_autoencoder(X_pca.shape[1])
    X_train, X_val = train_test_split(X_pca, test_size=0.2, random_state=42)
    autoencoder.fit(X_train, X_train, validation_data=(X_val, X_val),
                    epochs=100, batch_size=32,
                    callbacks=[EarlyStopping(patience=10)],
                    verbose=0)
    encoded_data = encoder.predict(X_pca)

    # UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = reducer.fit_transform(encoded_data)

    models = ["HDBSCAN", "KMeans", "Agglomerative", "DBSCAN"]
    scores = {}
    labels_dict = {}

    for model in models:
        score, labels = evaluate_model(model, embedding, n_clusters=4)
        scores[model] = score
        labels_dict[model] = labels

    best_model = max(scores, key=scores.get)
    best_labels = labels_dict[best_model]
    best_score = scores[best_model]
    explanation = explain_model_choice(best_model, scores)

    # Plot best model
    os.makedirs('static', exist_ok=True)
    plot_path = 'static/cluster_plot.png'
    plt.figure(figsize=(8, 6))
    mask = np.array(best_labels) != -1
    sns.scatterplot(x=embedding[mask][:, 0], y=embedding[mask][:, 1],
                    hue=np.array(best_labels)[mask], palette='viridis', s=100)
    plt.title(f"{best_model} Clusters (Silhouette Score: {best_score})")
    plt.xlabel("UMAP Feature 1")
    plt.ylabel("UMAP Feature 2")
    plt.legend(title="Cluster")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    df_result = df.copy()
    df_result['Cluster'] = best_labels

    return df_result, best_model, best_score, scores, explanation, plot_path
