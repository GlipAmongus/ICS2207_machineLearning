import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris, load_wine
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

def load_dataset(name):
    if name == "iris":
        iris = fetch_ucirepo(id=53)
        data = iris
        feature_names = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    elif name == "wine":
        wine = fetch_ucirepo(id=109)
        data = wine
        feature_names = [
            "alcohol", "malic acid", "ash", "alcalinity of ash", "magnesium",
            "total phenols", "flavanoids", "nonflavanoid phenols", "proanthocyanins",
            "color intensity", "hue", "OD280/OD315 of diluted wines", "proline"
        ]
    else:
        raise ValueError("Unsupported dataset. Choose 'iris' or 'wine'.")
    
    df = pd.DataFrame(data.data, columns=feature_names)
    df['target'] = data.target
    return df, feature_names

def plot_3d(df, features, target_name, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        df[features[0]], df[features[1]], df[features[2]],
        c=df['target'], cmap='viridis', label=df[target_name]
    )
    
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
    plt.title(title)
    plt.legend(*scatter.legend_elements(), title=target_name)
    plt.show()

# Component 2: k-Means Clustering

def kmeans_clustering(df, features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[features])
    return kmeans

def plot_clusters_3d(df, features, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        df[features[0]], df[features[1]], df[features[2]],
        c=df['cluster'], cmap='viridis'
    )
    
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
    plt.title(title)
    plt.show()

# Component 3: k-NN Classification

def knn_classification(df, features, target_name):
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Elbow method
    k_values = range(1, 15)
    scores = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, X_train, y_train, cv=5).mean()
        scores.append(score)
    
    optimal_k = k_values[np.argmax(scores)]
    print(f"Optimal k: {optimal_k}")

    # Train and test the model
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {accuracy:.2f}")

# Main program
if __name__ == "__main__":
    dataset_name = input("Enter dataset name ('iris' or 'wine'): ").lower()
    df, feature_names = load_dataset(dataset_name)
    
    print("Features available:")
    for i, feature in enumerate(feature_names):
        print(f"{i + 1}. {feature}")
    
    selected_features = []
    for i in range(3):
        feature_index = int(input(f"Select feature {i + 1} (1-{len(feature_names)}): ")) - 1
        selected_features.append(feature_names[feature_index])

    # Component 1
    plot_3d(df, selected_features, target_name='target', title="3D Scatter Plot of Selected Features")

    # Component 2
    kmeans = kmeans_clustering(df, selected_features, n_clusters=len(np.unique(df['target'])))
    plot_clusters_3d(df, selected_features, title="k-Means Clustering")

    # Component 3
    knn_classification(df, selected_features, target_name='target')
