import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

# Load data
movies = pd.read_csv(r"C:\Users\sayan\Downloads\ml-latest\ml-latest\movies.csv")
ratings = pd.read_csv(r"C:\Users\sayan\Downloads\ml-latest\ml-latest\ratings.csv")

# Filter active users and popular movies
popular_movies = ratings['movieId'].value_counts()
popular_movies = popular_movies[popular_movies >= 300].index
active_users = ratings['userId'].value_counts()
active_users = active_users[active_users >= 100].index

filtered_ratings = ratings[
    ratings['movieId'].isin(popular_movies) & ratings['userId'].isin(active_users)
]

# Train-test split
train_ratings, test_ratings = train_test_split(filtered_ratings, test_size=0.2, random_state=42)

# Pivot tables
train_movie_ratings = pd.merge(train_ratings, movies, on='movieId')
user_movie_matrix = train_movie_ratings.pivot_table(index='userId', columns='title', values='rating').fillna(0)

movie_user_matrix = user_movie_matrix.T
sparse_movie_matrix = csr_matrix(movie_user_matrix.values)

# KNN model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(sparse_movie_matrix)

movie_index_list = movie_user_matrix.index.tolist()

def get_movie_recommendations(title):
    if title not in movie_user_matrix.index:
        return []
    idx = movie_index_list.index(title)
    distances, indices = model_knn.kneighbors(sparse_movie_matrix[idx], n_neighbors=6)
    recommendations = []
    for i in range(1, len(indices.flatten())):
        recommended_title = movie_user_matrix.index[indices.flatten()[i]]
        similarity = 1 - distances.flatten()[i]
        recommendations.append((recommended_title, similarity))
    return recommendations

def precision_recall_at_k(model, test_data, k=5, sample_size=100):
    user_movie_pivot_test = pd.merge(test_data, movies, on='movieId').pivot_table(
        index='userId', columns='title', values='rating').fillna(0)
    users = user_movie_pivot_test.index.tolist()
    if len(users) > sample_size:
        users = np.random.choice(users, sample_size, replace=False)
    hits, total_pred, total_actual = 0, 0, 0
    for user_id in users:
        rated_movies = user_movie_pivot_test.loc[user_id]
        liked_movies = rated_movies[rated_movies >= 4.0].index.tolist()
        if not liked_movies:
            continue
        movie = np.random.choice(liked_movies)
        recs = get_movie_recommendations(movie)[:k]
        rec_titles = [r[0] for r in recs]
        actual_liked = set(liked_movies)
        recommended_set = set(rec_titles)
        true_positives = actual_liked.intersection(recommended_set)
        hits += len(true_positives)
        total_pred += len(recommended_set)
        total_actual += len(actual_liked)
    precision = hits / total_pred if total_pred != 0 else 0
    recall = hits / total_actual if total_actual != 0 else 0
    return precision, recall

# Run evaluation
precision, recall = precision_recall_at_k(model_knn, test_ratings, k=5, sample_size=100)
print(f"📊 Evaluation Results:")
print(f"   Precision@5: {precision:.4f}")
print(f"   Recall@5   : {recall:.4f}")

# --- Graphical Analysis ---

# 1. Similarity Distribution Plot
random_movie = np.random.choice(movie_user_matrix.index.tolist())
distances, _ = model_knn.kneighbors(sparse_movie_matrix[movie_index_list.index(random_movie)], n_neighbors=20)
similarities = 1 - distances.flatten()[1:]

plt.figure(figsize=(8, 4))
sns.histplot(similarities, bins=10, kde=True, color='skyblue')
plt.title(f'Similarity Scores for Recommendations of "{random_movie}"')
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 2. User-Movie Interaction Heatmap
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 10))

# Select subset of the matrix
subset_matrix = user_movie_matrix.iloc[:50, :50]

# Convert user IDs into strings like "User 123"
subset_matrix.index = [f"User {i}" for i in subset_matrix.index]

# Create the heatmap with custom y-tick labels
sns.heatmap(subset_matrix, cmap="YlGnBu", cbar=True, yticklabels=True)

plt.title("User-Movie Rating Heatmap (Subset)")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# 3. Precision and Recall at K Curve
k_values = [3, 5, 10]
precisions = []
recalls = []
for k in k_values:
    p, r = precision_recall_at_k(model_knn, test_ratings, k=k, sample_size=100)
    precisions.append(p)
    recalls.append(r)

plt.figure(figsize=(8, 4))
plt.plot(k_values, precisions, marker='o', label='Precision@K')
plt.plot(k_values, recalls, marker='x', label='Recall@K')
plt.title("Precision and Recall at Different K Values")
plt.xlabel("K")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#  GUI
root = tk.Tk()
root.title("🎬 Movie Recommendation System")
root.geometry("600x550")
root.configure(bg="#2C3E50")

tk.Label(root, text="🎥 ML-Based Movie Recommendation System", font=("Helvetica", 16, "bold"), bg="#2C3E50", fg="white").pack(pady=10)
tk.Label(root, text="Select a movie:", font=("Arial", 12), bg="#2C3E50", fg="white").pack()

movie_titles = sorted(movie_user_matrix.index.tolist())
movie_var = tk.StringVar()
movie_combobox = ttk.Combobox(root, textvariable=movie_var, values=movie_titles, width=50, font=("Arial", 11))
movie_combobox.pack(pady=5)

def show_recommendations():
    movie_title = movie_var.get().strip()
    if not movie_title:
        messagebox.showwarning("Warning", "Please select a movie from the dropdown.")
        return
    recommendations = get_movie_recommendations(movie_title)
    if recommendations:
        output = f"🎬 Recommendations for '{movie_title}':\n\n"
        for movie, score in recommendations:
            output += f"⭐ {movie} (Similarity: {score:.2f})\n"
    else:
        output = "❌ Movie not found or no similar recommendations."
    result_text.config(state=tk.NORMAL)
    result_text.delete("1.0", tk.END)
    result_text.insert(tk.END, output)
    result_text.config(state=tk.DISABLED)

tk.Button(root, text="🔍 Get Recommendations", command=show_recommendations,
          font=("Arial", 12, "bold"), bg="#E74C3C", fg="white", padx=10, pady=5).pack(pady=10)

frame = tk.Frame(root, bg="#2C3E50")
frame.pack(pady=10, padx=10, fill="both", expand=True)

result_text = scrolledtext.ScrolledText(frame, width=65, height=14, font=("Arial", 12),
                                        wrap=tk.WORD, state=tk.DISABLED, bg="#ECF0F1", fg="#2C3E50")
result_text.pack(padx=10, pady=5, fill="both", expand=True)

root.mainloop()
