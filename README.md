# Movie-Recommendation-System
This is a Python-based movie recommendation system that uses the K-Nearest Neighbors (KNN) algorithm with cosine similarity to recommend movies based on user ratings. It includes a user-friendly Tkinter GUI, graphical analysis, and performance evaluation metrics such as Precision@K and Recall@K.

📁 Dataset
The system uses the MovieLens dataset.

Required files:

movies.csv – Contains movie titles and IDs

ratings.csv – Contains user ratings for each movie

📌 Features
✅ Core Functionalities:
Movie recommendations based on collaborative filtering

User-based rating matrix construction

Interactive GUI for selecting movies and displaying recommendations

📊 Evaluation:
Precision@K and Recall@K metrics

Precision and Recall plotted across different values of K (3, 5, 10)

📈 Visualizations:
Similarity Distribution Plot – Shows how similar recommended movies are to a selected movie.

User-Movie Heatmap – Shows rating distribution between a subset of users and movies.

Precision-Recall Curve – Evaluates recommendation quality at different levels of K.

🧠 ML Model
Model Used: K-Nearest Neighbors (sklearn.neighbors.NearestNeighbors)

Similarity Metric: Cosine Similarity

Training Data: Ratings matrix pivoted by userId and movie title

Filtering: Only includes:

Users who have rated at least 100 movies

Movies with at least 300 ratings

🖼 GUI Interface
Built with Tkinter, the GUI provides:

A dropdown list to select a movie

A button to fetch top 5 similar movie recommendations

A scrollable text area to view the results

▶️ How to Run
Ensure required libraries are installed:


pip install pandas numpy matplotlib seaborn scikit-learn
Place your dataset (movies.csv, ratings.csv) in the correct path (update paths in the script if necessary).

Run the Python script:


python movie_recommendation_system.py
The GUI window will open where you can select a movie and see recommendations.

📂 Project Structure

movie_recommendation_system.py    # Main Python script
movies.csv                        # Movie metadata (MovieLens)
ratings.csv                       # User ratings (MovieLens)
README.md                         # Project documentation
🛠 Customization
You can adjust the recommendation parameters:

Minimum ratings per movie (>= 300)

Minimum ratings per user (>= 100)

Number of neighbors (n_neighbors=6)

Evaluation sample size (sample_size=100)

📌 Dependencies
Python 3.x

pandas

numpy

matplotlib

seaborn

scikit-learn

tkinter (included in standard Python)

📬 Output Example

📊 Evaluation Results:
   Precision@5: 0.3120
   Recall@5   : 0.2014

🎬 Recommendations for 'The Matrix (1999)':

⭐ The Matrix Reloaded (2003) (Similarity: 0.86)
⭐ Inception (2010) (Similarity: 0.81)
⭐ The Matrix Revolutions (2003) (Similarity: 0.78)
⭐ Equilibrium (2002) (Similarity: 0.76)
⭐ Minority Report (2002) (Similarity: 0.74)














