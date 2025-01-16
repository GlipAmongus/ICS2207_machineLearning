import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix, classification_report

# Load the datasets
def load_data(dataset_name="iris"):
    if dataset_name == "iris":
        # Load Iris dataset
        data = datasets.load_iris()
    elif dataset_name == "wine":
        # Load Wine dataset
        data = datasets.load_wine()
    return data

# Visualize the data with three features in 3D
def plot_3d(data, selected_features, title="3D Scatter Plot"):
    X = data.data[:, selected_features]  # Selecting the 3 features for the plot
    y = data.target
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis')
    ax.set_xlabel(data.feature_names[selected_features[0]])
    ax.set_ylabel(data.feature_names[selected_features[1]])
    ax.set_zlabel(data.feature_names[selected_features[2]])
    ax.set_title(title)
    plt.show()

# k-Means Clustering Algorithm
def k_means_clustering(data, selected_features, n_clusters=3):
    X = data.data[:, selected_features]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
            
    # Plotting the clustering result
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_kmeans, cmap='brg')        
    ax.set_xlabel(data.feature_names[selected_features[0]])
    ax.set_ylabel(data.feature_names[selected_features[1]])
    ax.set_zlabel(data.feature_names[selected_features[2]])
    ax.set_title("k-Means Clustering")
    plt.show()

    # Evaluate the clustering performance (Silhouette Score)
    silhouette_avg = silhouette_score(X, y_kmeans)
    print(f"Silhouette Score: {silhouette_avg}")

# k-Nearest Neighbors (k-NN) Algorithm
def k_nn_classification(data, selected_features, test_size=0.3, k_value=5):
    X = data.data[:, selected_features]
    y = data.target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialize and train the k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of k-NN with k={k_value}: {accuracy * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return knn

# Find optimal k using the elbow method (for k-Means clustering)
def elbow_method(data, selected_features, max_k=10):
    X = data.data[:, selected_features]
    inertia = []
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    
    # Plotting the elbow graph
    plt.plot(range(1, max_k + 1), inertia, marker='o')
    plt.title("Elbow Method For Optimal k")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.show()

# Main function to run all components
def main():
    # ------------ IRIS ------------
    # Load the dataset
    dataset_name = "iris"
    data = load_data(dataset_name)
    
    # Visualize the data (pick three features)
    print(dataset_name)
    print("Available features:")
    for i, feature in enumerate(data.feature_names):
        print(f"{i}: {feature}")
    
    # User input for selecting 3 features
    selected_features = []
    while len(selected_features) < 3:
        try:
            feature_index = int(input(f"Select feature {len(selected_features) + 1} (0-{len(data.feature_names) - 1}): "))
            if feature_index < 0 or feature_index >= len(data.feature_names):
                print("Invalid index. Please select a valid feature index.")
            elif feature_index in selected_features:
                print("Feature already selected. Please choose a different one.")
            else:
                selected_features.append(feature_index)
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    plot_3d(data, selected_features, title=f"{dataset_name.capitalize()} - 3D Scatter")
    
    # k-Means Clustering
    k_means_clustering(data, selected_features, n_clusters=3)
    
    # k-NN Classification
    k_nn_classification(data, selected_features, test_size=0.3, k_value=5)
    
    # Use the elbow method for k-NN
    elbow_method(data, selected_features)
    # ------------------------------
    
    
    # ------------ WINE ------------
    # Load the dataset
    dataset_name = "wine"
    data = load_data(dataset_name)
    
    # Visualize the data (pick three features)
    print(dataset_name)
    print("Available features:")
    for i, feature in enumerate(data.feature_names):
        print(f"{i}: {feature}")
    
    # User input for selecting 3 features
    selected_features = []
    while len(selected_features) < 3:
        try:
            feature_index = int(input(f"Select feature {len(selected_features) + 1} (0-{len(data.feature_names) - 1}): "))
            if feature_index < 0 or feature_index >= len(data.feature_names):
                print("Invalid index. Please select a valid feature index.")
            elif feature_index in selected_features:
                print("Feature already selected. Please choose a different one.")
            else:
                selected_features.append(feature_index)
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    plot_3d(data, selected_features, title=f"{dataset_name.capitalize()} - 3D Scatter")
    
    # k-Means Clustering
    k_means_clustering(data, selected_features, n_clusters=3)
    
    # k-NN Classification
    k_nn_classification(data, selected_features, test_size=0.3, k_value=5)
    
    # Use the elbow method for k-NN
    elbow_method(data, selected_features)

if __name__ == "__main__":
    main()