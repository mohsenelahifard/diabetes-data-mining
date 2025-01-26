import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
import seaborn as sns
import math
import copy
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
df = pd.read_csv("modified_diabetes_prediction_dataset.csv")
print(f"The count of null cells in each column is:\n{df.isnull().sum()}")
print(f"Records with null values:\n{df[df.isnull().any(axis=1)]}")
print("Distinct Values for each column with count of each one are: ")
for column in df.columns:
    print(df[column].value_counts().sort_index())
df.replace({"gender": {"unknown": np.nan, "Other": np.nan}}, inplace=True)
df.replace({"smoking_history": {"yes": "current"}}, inplace=True)
df.replace({"blood_glucose_level": {9999: np.nan}}, inplace=True)
df["age"] = df["age"].where(df["age"] >= 0, np.nan)
df["bmi"].plot(kind="hist", bins=30, edgecolor="black")
plt.xlabel("BMI")
plt.title("Distribution of BMI")
plt.show()
skewness = skew(df["bmi"])
print(f"Skewness of BMI: {skewness}")
sns.boxplot(x="bmi", data=df)
plt.show()
df["bmi"] = df["bmi"].where((df["bmi"] >= 10) & (df["bmi"] <= 60), np.nan)
print("Duplicate records are: ")
duplicates = df[df.duplicated(keep=False)]
print(duplicates)
df = df.drop_duplicates()
for column in ["age", "bmi"]:
    value_counts = df[column].value_counts(normalize=True)
    df.loc[df[column].isna(), column] = np.random.choice(
        value_counts.index.tolist(), size=df[column].isna().sum(), p=value_counts.values
    )
df = df.dropna()
df = df.reset_index(drop=True)
df_zero = df[df["diabetes"] == 0]
df_one = df[df["diabetes"] == 1]
sample_size = int(len(df_one) * 0.7)
zero_sample = df_zero.sample(n=sample_size, random_state=42)
one_sample = df_one.sample(n=sample_size, random_state=42)
balanced_train = pd.concat([zero_sample, one_sample])
balanced_train = balanced_train.sample(frac=1, random_state=42).reset_index(drop=True)
test_data = df.drop(balanced_train.index).reset_index(drop=True)
train_data = balanced_train
attribute = train_data.columns.to_list()[:-1]
for col in attribute:
    if col == "age":
        bins = [0, 24, 54, float("inf")]
        labels = ["Youth", "Middle", "Old"]
        train_data[col] = pd.cut(
            train_data[col], bins=bins, labels=labels, include_lowest=True
        )
        test_data[col] = pd.cut(
            test_data[col], bins=bins, labels=labels, include_lowest=True
        )
    elif col in ["bmi", "HbA1c_level", "blood_glucose_level"]:
        if col == "bmi":
            bins = [0, 18.5, 24.9, float("inf")]
        elif col == "HbA1c_level":
            bins = [0, 5.7, 6.5, float("inf")]
        elif col == "blood_glucose_level":
            bins = [0, 140, 200, float("inf")]
        train_data[col] = pd.cut(
            train_data[col], bins=bins, labels=["Low", "Normal", "High"]
        )
        test_data[col] = pd.cut(
            test_data[col], bins=bins, labels=["Low", "Normal", "High"]
        )
    elif col in ["gender", "hypertension", "heart_disease"]:
        pass
train_data["diabetes"] = train_data["diabetes"].map({0: "No", 1: "Yes"})
test_data["diabetes"] = test_data["diabetes"].map({0: "No", 1: "Yes"})
X_train = train_data.iloc[:, :].values
X_test = test_data.iloc[:, :].values
print("Decision-Tree classification results:")


class Node(object):
    def __init__(self):
        self.value = None
        self.decision = None
        self.childs = []


def findEntropy(data, rows):
    yes, no, ans = 0, 0, -1
    idx = len(data[0]) - 1
    for i in rows:
        if data[i][idx] == "Yes":
            yes += 1
        else:
            no += 1
    if yes + no != 0:
        x, y = yes / (yes + no), no / (yes + no)
    else:
        x, y = 0, 0
    if x == 1:
        ans = 1
    elif y == 1:
        ans = 0
    entropy = -1 * (x * math.log2(x) + y * math.log2(y)) if x != 0 and y != 0 else 0
    return entropy, ans


def findMaxGain(data, rows, columns):
    maxGain, retidx = 0, -1
    entropy, ans = findEntropy(data, rows)
    if entropy == 0:
        return maxGain, retidx, ans
    for j in columns:
        mydict = {}
        for i in rows:
            key = data[i][j]
            mydict[key] = mydict.get(key, 0) + 1
        gain = entropy
        for key in mydict:
            yes = no = 0
            for k in rows:
                if data[k][j] == key:
                    if data[k][-1] == "Yes":
                        yes += 1
                    else:
                        no += 1
            if yes + no != 0:
                x, y = yes / (yes + no), no / (yes + no)
            else:
                x, y = 0, 0
            if x != 0 and y != 0:
                gain += (mydict[key] * (x * math.log2(x) + y * math.log2(y))) / len(
                    rows
                )
        if gain > maxGain:
            maxGain = gain
            retidx = j
    return maxGain, retidx, ans


def buildTree(data, rows, columns):
    maxGain, idx, ans = findMaxGain(data, rows, columns)
    root = Node()
    root.childs = []
    if maxGain == 0:
        root.value = "Yes" if ans == 1 else "No"
        return root
    root.value = attribute[idx]
    mydict = {}
    for i in rows:
        key = data[i][idx]
        mydict[key] = mydict.get(key, 0) + 1
    newcolumns = copy.deepcopy(columns)
    newcolumns.remove(idx)
    for key in mydict:
        newrows = [i for i in rows if data[i][idx] == key]
        temp = buildTree(data, newrows, newcolumns)
        temp.decision = key
        root.childs.append(temp)
    return root


def predict(root, data_row):
    if not root.childs:
        return root.value
    for child in root.childs:
        if data_row[attribute.index(root.value)] == child.decision:
            return predict(child, data_row)
    return "No"


def evaluate_tree(root, X):
    y_true = [row[-1] for row in X]
    y_pred = [predict(root, row) for row in X]
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="Yes")
    rec = recall_score(y_true, y_pred, pos_label="Yes")
    f1 = f1_score(y_true, y_pred, pos_label="Yes")
    cm = confusion_matrix(y_true, y_pred, labels=["Yes", "No"])
    return acc, prec, rec, f1, cm


rows_train = [i for i in range(len(X_train))]
rows_test = [i for i in range(len(X_test))]
columns = [i for i in range(len(attribute))]
root = buildTree(X_train, rows_train, columns)
acc_train, prec_train, rec_train, f1_train, cm_train = evaluate_tree(root, X_train)
acc_test, prec_test, rec_test, f1_test, cm_test = evaluate_tree(root, X_test)
print("Train Metrics:")
print(
    f"Accuracy: {acc_train:.2f}, Precision: {prec_train:.2f}, Recall: {rec_train:.2f}, F1-Score: {f1_train:.2f}"
)
print("Confusion Matrix (Train):\n", cm_train)
print("\nTest Metrics:")
print(
    f"Accuracy: {acc_test:.2f}, Precision: {prec_test:.2f}, Recall: {rec_test:.2f}, F1-Score: {f1_test:.2f}"
)
print("Confusion Matrix (Test):\n", cm_test)


def collect_tree_data(
    root, pos={}, x=0, y=0, level=1, parent=None, edges=[], labels=[]
):
    if root is None:
        return pos, edges, labels
    pos[(x, y)] = f"{root.value if root.value else root.decision}"
    if parent is not None:
        edges.append((parent, (x, y)))
        labels.append(root.decision)
    n = len(root.childs)
    for i, child in enumerate(root.childs):
        child_x = x + (i - n / 2) * (3 / level)
        child_y = y - 1.5
        pos, edges, labels = collect_tree_data(
            child, pos, child_x, child_y, level + 1, (x, y), edges, labels
        )
    return pos, edges, labels


def draw_tree(pos, edges, labels):
    fig, ax = plt.subplots(figsize=(20, 15))
    for idx, ((x1, y1), (x2, y2)) in enumerate(edges):
        ax.plot([x1, x2], [y1, y2], "k-", lw=1)
        label_x, label_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(label_x, label_y, labels[idx], color="red", fontsize=8, ha="center")
    for (x, y), label in pos.items():
        circle_radius = max(0.2, 0.05 * len(label))
        ax.add_patch(plt.Circle((x, y), circle_radius, color="blue"))
        ax.text(x, y, label, color="white", ha="center", va="center", fontsize=9)
    ax.axis("off")
    plt.show()


pos, edges, labels = collect_tree_data(root)
draw_tree(pos, edges, labels)
df_sample = df.sample(n=5000, random_state=42)
df_sample = df_sample.reset_index(drop=True)
df_zero = df_sample[df_sample["diabetes"] == 0]
df_one = df_sample[df_sample["diabetes"] == 1]
sample_size = int(len(df_one) * 0.7)
zero_sample = df_zero.sample(n=sample_size, random_state=42)
one_sample = df_one.sample(n=sample_size, random_state=42)
balanced_train = pd.concat([zero_sample, one_sample])
balanced_train = balanced_train.sample(frac=1, random_state=42).reset_index(drop=True)
test_data = df_sample.drop(balanced_train.index).reset_index(drop=True)


def preprocess_dataset(data):
    data = data.copy()
    data["gender"] = data["gender"].map({"Male": 1, "Female": 0})
    smoking_map = {
        "No Info": 0,
        "never": 0.2,
        "not current": 0.4,
        "former": 0.6,
        "ever": 0.8,
        "current": 1,
    }
    data["smoking_history"] = data["smoking_history"].map(smoking_map)
    numeric_columns = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    for col in numeric_columns:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    return data


train_data = preprocess_dataset(balanced_train)
test_data = preprocess_dataset(test_data)
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values
print("KNN classification results:")


def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2) ** 2))


def predict_knn(X_train, y_train, row, k):
    distances = []
    for i, train_row in enumerate(X_train):
        dist = euclidean_distance(train_row, row)
        distances.append((dist, y_train[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]
    classes = [neighbor[1] for neighbor in k_nearest]
    prediction = max(set(classes), key=classes.count)
    return prediction


def evaluate_knn(X_train, y_train, X_test, y_test, k):
    y_pred = [predict_knn(X_train, y_train, row, k) for row in X_test]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
    return acc, prec, rec, f1, cm


k = 2
acc_test, prec_test, rec_test, f1_test, cm_test = evaluate_knn(
    X_train, y_train, X_test, y_test, k
)
print("\nTest Metrics:")
print(
    f"Accuracy: {acc_test:.2f}, Precision: {prec_test:.2f}, Recall: {rec_test:.2f}, F1-Score: {f1_test:.2f}"
)
print("Confusion Matrix (Test):\n", cm_test)
class KMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        np.random.seed(42)
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]
        for _ in range(self.max_iter):
            self.labels = self._assign_clusters(X)
            previous_centroids = self.centroids.copy()
            self.centroids = self._update_centroids(X)
            if self._is_converged(previous_centroids, self.centroids):
                break

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X):
        return np.array(
            [X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)]
        )

    def _is_converged(self, old_centroids, new_centroids):
        return np.all(np.linalg.norm(old_centroids - new_centroids, axis=1) < self.tol)

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
df_zero = df[df["diabetes"] == 0]
df_one = df[df["diabetes"] == 1]
sample_size = int(len(df_one) * 0.7)
zero_sample = df_zero.sample(n=sample_size, random_state=42)
one_sample = df_one.sample(n=sample_size, random_state=42)
sampled_df = pd.concat([zero_sample, one_sample])
data = preprocess_dataset(sampled_df)
data.drop(columns=['diabetes'], inplace=True)
inertia = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data.values)
    inertia_k = 0
    for cluster, centroid in enumerate(kmeans.centroids):
        cluster_points = data.values[kmeans.labels == cluster]
        inertia_k += np.sum(np.linalg.norm(cluster_points - centroid, axis=1)**2)
    inertia.append(inertia_k)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()
print("K-means Clustering results:")

kmeans_manual = KMeans(n_clusters=2)
kmeans_manual.fit(data.values)
sampled_df['Cluster'] = kmeans_manual.predict(data.values)
plt.figure(figsize=(8, 5))
plt.scatter(data['hypertension'], data['blood_glucose_level'], c=sampled_df['Cluster'], cmap='viridis', s=50)
plt.title('Clusters Visualization')
plt.xlabel('hypertension')
plt.ylabel('blood_glucose_level')
plt.colorbar(label='Cluster')
plt.show()
cluster_summary = sampled_df.groupby(['Cluster', 'heart_disease']).size().unstack(fill_value=0)
print(sampled_df.groupby(['Cluster', 'heart_disease']).size().unstack(fill_value=0))
print(sampled_df.groupby(['Cluster', 'diabetes']).size().unstack(fill_value=0))
print("Hierarchical Clustering results:")


def calculate_distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def hierarchical_clustering(X):
    n_samples = X.shape[0]
    clusters = {i: [i] for i in range(n_samples)}
    distances = {
        (i, j): calculate_distance(X[i], X[j])
        for i in range(n_samples)
        for j in range(i + 1, n_samples)
    }
    merge_steps = []
    while len(clusters) > 1:
        (cluster_i, cluster_j), min_distance = min(
            distances.items(), key=lambda x: x[1]
        )
        merge_steps.append((cluster_i, cluster_j, min_distance))
        new_cluster = clusters[cluster_i] + clusters[cluster_j]
        new_cluster_id = max(clusters.keys()) + 1
        clusters[new_cluster_id] = new_cluster
        del clusters[cluster_i]
        del clusters[cluster_j]
        distances = {
            (i, j): dist
            for (i, j), dist in distances.items()
            if i not in [cluster_i, cluster_j] and j not in [cluster_i, cluster_j]
        }
        for cluster in clusters.keys():
            if cluster != new_cluster_id:
                dist = np.mean(
                    [
                        calculate_distance(X[p1], X[p2])
                        for p1 in clusters[cluster]
                        for p2 in new_cluster
                    ]
                )
                distances[
                    min(cluster, new_cluster_id), max(cluster, new_cluster_id)
                ] = dist
    return merge_steps


def plot_dendrogram(merge_steps, n_samples):
    plt.figure(figsize=(12, 8))
    cluster_heights = {i: 0 for i in range(n_samples)}
    current_cluster_id = n_samples
    for cluster_i, cluster_j, distance in merge_steps:
        x1 = cluster_i if cluster_i < n_samples else current_cluster_id
        x2 = cluster_j if cluster_j < n_samples else current_cluster_id + 1
        y1 = cluster_heights[cluster_i]
        y2 = cluster_heights[cluster_j]
        plt.plot([x1, x1, x2, x2], [y1, distance, distance, y2], c="b")
        cluster_heights[current_cluster_id] = distance
        current_cluster_id += 1
    plt.title("Dendrogram (Hierarchical Clustering)")
    plt.xlabel("Samples or Clusters")
    plt.ylabel("Distance")
    plt.show()
    

data = preprocess_dataset(df)
data.drop(columns=["diabetes"], inplace=True)
sampled_data = data.sample(n=100, random_state=42).values
merge_steps = hierarchical_clustering(sampled_data)
plot_dendrogram(merge_steps, sampled_data.shape[0])
sampled_data = data.sample(n=100, random_state=42).values
linked = linkage(sampled_data, method='ward')
plt.figure(figsize=(12, 8))
dendrogram(linked)
plt.axhline(y=1.0, color='r', linestyle='--')
plt.title('Dendrogram with Threshold')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()
threshold = 1.0
clusters = fcluster(linked, t=threshold, criterion='distance')
clustered_data = data.sample(n=100, random_state=42).copy()
clustered_data['Cluster'] = clusters
for cluster_num in np.unique(clusters):
    print(f"Cluster {cluster_num}:")
    print(clustered_data[clustered_data['Cluster'] == cluster_num])