import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import csv

# ==============================================================================
# 1. DATA LOADING AND CLEANING
# ==============================================================================

def clean_raw_csv_file(raw_file_path, clean_file_path):
    """
    Reads a raw CSV, cleans fields by removing newlines, and saves a new
    CSV with all fields properly quoted to prevent parsing errors.
    """
    try:
        with open(raw_file_path, 'r', encoding='utf-8', errors='replace') as infile, \
             open(clean_file_path, 'w', encoding='utf-8', newline='') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)

            print(f"Cleaning {raw_file_path}...")
            header = next(reader)
            writer.writerow(header)
            
            for row in reader:
                if len(row) == len(header):
                    cleaned_row = [field.replace('\n', ' ').replace('\r', '') if field else '' for field in row]
                    writer.writerow(cleaned_row)
        print(f"Cleaned data saved to {clean_file_path}")
        return True
    except FileNotFoundError:
        print(f"Error: The file {raw_file_path} was not found.")
        return False
    except StopIteration:
        print(f"Error: The file {raw_file_path} appears to be empty.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during CSV cleaning: {e}")
        return False

def clean_column_names(df):
    """
    Standardizes all column names in a DataFrame to prevent KeyErrors.
    - Strips leading/trailing whitespace
    - Converts to lowercase
    - Replaces spaces with underscores
    """
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def load_data(file_path, required_columns):
    """
    Loads a CSV file, cleans its column names, and validates required columns.
    """
    try:
        # Try loading with standard UTF-8 first, fallback to latin1
        try:
            df = pd.read_csv(file_path, engine='python')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, engine='python', encoding='latin1')
        
        # Proactively clean column names to prevent KeyErrors
        df = clean_column_names(df)
        
        # Check if all required columns are present
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: The file {file_path} is missing required columns: {missing_cols}")
            print(f"Available columns: {df.columns.tolist()}")
            return None
            
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading {file_path}: {e}")
        return None

# ==============================================================================
# 2. PREPROCESSING AND FEATURE ENGINEERING
# ==============================================================================

def preprocess_users(df):
    """Preprocesses the users dataframe."""
    df['interested_in'] = df['interested_in'].fillna('').astype(str)
    df['interested_in_list'] = df['interested_in'].apply(lambda x: [topic.strip() for topic in x.split(',') if topic.strip()])
    return df

def preprocess_posts(df):
    """Preprocesses the posts dataframe, handling types and outliers."""
    # Ensure text columns are strings
    df['topics'] = df['topics'].fillna('').astype(str)
    df['like_user_ids'] = df['like_user_ids'].fillna('').astype(str)
    df['content'] = df['content'].fillna('').astype(str)
    
    # Coerce numeric types, filling errors with NaN
    df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce')
    df['likes'] = pd.to_numeric(df['likes'], errors='coerce')
    df['shares'] = pd.to_numeric(df['shares'], errors='coerce')
    
    # Handle dates
    df['created_at'] = pd.to_datetime(df['created_at'], format='mixed', errors='coerce')
    df.dropna(subset=['created_at', 'user_id'], inplace=True) # Drop rows with invalid date or user_id

    # **FIX**: Explicitly convert to float BEFORE capping to avoid dtype warnings.
    df['likes'] = df['likes'].fillna(0).astype(float)
    df['shares'] = df['shares'].fillna(0).astype(float)

    # Cap outliers using Z-score method
    print("\nChecking for outliers in 'likes' and 'shares'...")
    for col in ['likes', 'shares']:
        mean, std = df[col].mean(), df[col].std()
        if std > 0:
            threshold = 3
            outlier_mask = (df[col] - mean).abs() > threshold * std
            if outlier_mask.any():
                print(f"Found and capped {outlier_mask.sum()} outliers in the '{col}' column.")
                cap_value = mean + threshold * std
                df.loc[outlier_mask, col] = cap_value
    
    df['like_user_ids_list'] = df['like_user_ids'].apply(
        lambda x: [int(uid) for uid in x.split(',') if uid.isdigit()]
    )
    return df

def apply_topic_inference(df):
    """Applies keyword-based topic inference to the posts dataframe."""
    topic_keywords = {
        'Investing 101': ['ipo', 'sip', 'allotment', 'f&o', 'nifty 50', 'sensex', 'etf', 'mutual fund', 'demat', 'stock', 'shares', 'equity'],
        'Market Trends': ['market', 'gdp', 'inflation', 'economy', 'fii', 'dii', 'bull', 'bear', 'valuation'],
        'Personal Finance': ['tax', 'gst', 'loan', 'emi', 'insurance', 'credit card', 'sip', 'save', 'income'],
        'News': ['news', 'modi', 'trump', 'rbi', 'sebi', 'government', 'policy', 'tariff'],
        'Memes': ['lol', 'lmao', 'meme', 'funny', '🤣', '😂'],
    }
    
    def infer_topics(content):
        inferred = set()
        content_lower = str(content).lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                inferred.add(topic)
        return list(inferred)

    existing_topics = df['topics'].apply(lambda x: [t.strip() for t in x.split(',') if t.strip()])
    inferred_topics = df['content'].apply(infer_topics)
    
    df['all_topics'] = [list(dict.fromkeys(e + i)) for e, i in zip(existing_topics, inferred_topics)]
    return df

def calculate_scores(df):
    """Calculates popularity and recency scores for each post."""
    # Popularity Score
    max_likes = df['likes'].max()
    max_shares = df['shares'].max()
    df['popularity_score'] = (df['likes'] / (max_likes or 1)) + (df['shares'] / (max_shares or 1))
    
    # Recency Score
    now_utc = datetime.now(timezone.utc)
    df['created_at_utc'] = df['created_at'].dt.tz_localize('UTC', ambiguous='infer')
    df['days_since_creation'] = (now_utc - df['created_at_utc']).dt.days
    decay_rate = 0.1
    df['recency_score'] = np.exp(-decay_rate * df['days_since_creation'])
    
    return df

# ==============================================================================
# 3. RECOMMENDATION AND EVALUATION
# ==============================================================================

def get_recommendations(user_id, users_df, posts_df, vectorizer, post_vectors, exclude_post_ids, top_n=10):
    """Generates top N post recommendations for a given user."""
    try:
        user_interests = users_df.loc[users_df['user_id'] == user_id, 'interested_in_list'].iloc[0]
    except IndexError:
        return [] # User not found

    # Filter out posts by the user and posts already seen
    candidate_posts = posts_df[
        (posts_df['user_id'] != user_id) & (~posts_df['post_id'].isin(exclude_post_ids))
    ].copy()

    if candidate_posts.empty:
        return []

    # Get indices of candidate posts to select correct vectors
    post_indices = candidate_posts.index.tolist()
    candidate_vectors = post_vectors[post_indices]
    
    # Calculate Semantic Score
    if user_interests:
        interests_str = ' '.join(user_interests)
        user_vector = vectorizer.transform([interests_str])
        sim_scores = cosine_similarity(user_vector, candidate_vectors).flatten()
        candidate_posts['semantic_score'] = sim_scores
    else:
        candidate_posts['semantic_score'] = 0

    # Calculate Final Score
    candidate_posts['final_score'] = (
        0.4 * candidate_posts['semantic_score'] +
        0.4 * candidate_posts['popularity_score'] +
        0.2 * candidate_posts['recency_score']
    )
    
    # Filter for relevance and return top N
    keyword_mask = candidate_posts['all_topics'].apply(lambda topics: any(item in user_interests for item in topics))
    relevant_posts = candidate_posts[(candidate_posts['semantic_score'] > 0.05) | keyword_mask]
    
    if relevant_posts.empty:
        # Fallback to the highest scored posts if no direct interest match
        relevant_posts = candidate_posts

    return relevant_posts.sort_values(by='final_score', ascending=False).head(top_n)['post_id'].tolist()

def evaluate_model(users_df, posts_df, vectorizer, post_vectors):
    """Evaluates the model using Precision@10 and Recall@10."""
    print("\n--- Evaluating Model Performance ---")
    precisions, recalls = [], []
    
    # Create a map of {user_id: {liked_post_ids}}
    all_liked_posts_map = {}
    for _, row in posts_df.iterrows():
        # Ensure post_id exists before iterating
        if 'post_id' in row and pd.notna(row['post_id']):
            for user_id in row['like_user_ids_list']:
                all_liked_posts_map.setdefault(user_id, set()).add(row['post_id'])

    # Use only users who have liked at least 2 posts for meaningful evaluation
    evaluation_users = [uid for uid, pids in all_liked_posts_map.items() if len(pids) > 1]
    if not evaluation_users:
        print("Not enough user interaction data to perform evaluation.")
        return

    for user_id in evaluation_users:
        liked_posts = list(all_liked_posts_map[user_id])
        random.shuffle(liked_posts)
        
        # Split liked posts into a training set (to be excluded) and a test set (ground truth)
        split_point = int(len(liked_posts) * 0.8)
        train_likes = set(liked_posts[:split_point])
        test_likes = set(liked_posts[split_point:])

        if not test_likes:
            continue

        recommendations = get_recommendations(user_id, users_df, posts_df, vectorizer, post_vectors, exclude_post_ids=train_likes)
        hits = len(set(recommendations) & test_likes)
        
        precisions.append(hits / len(recommendations) if recommendations else 0)
        recalls.append(hits / len(test_likes) if test_likes else 0)

    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    
    print(f"Evaluated on {len(evaluation_users)} users.")
    print(f"Average Precision@10: {avg_precision:.2%}")
    print(f"Average Recall@10:  {avg_recall:.2%}")
    print("------------------------------------")

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

def main():
    """Main function to run the entire recommendation pipeline."""
    # --- 1. Clean Raw CSVs ---
    if not clean_raw_csv_file('users.csv', 'users_clean.csv') or \
       not clean_raw_csv_file('posts.csv', 'posts_clean.csv'):
        return # Exit if cleaning fails

    # --- 2. Load and Validate Data ---
    users_df = load_data('users_clean.csv', required_columns=['user_id', 'interested_in'])
    posts_df = load_data('posts_clean.csv', required_columns=['post_id', 'user_id', 'content', 'created_at'])
    if users_df is None or posts_df is None:
        return # Exit if loading fails

    # --- 3. Preprocess Data ---
    print("\nPreprocessing data...")
    users_df = preprocess_users(users_df)
    posts_df = preprocess_posts(posts_df)
    
    # CRITICAL: Reset index after any filtering/dropping to ensure alignment with TF-IDF matrix
    posts_df.reset_index(drop=True, inplace=True)
    
    # --- 4. Feature Engineering ---
    print("\nEngineering features (topics, scores, embeddings)...")
    posts_df = apply_topic_inference(posts_df)
    posts_df = calculate_scores(posts_df)
    
    vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
    post_vectors = vectorizer.fit_transform(posts_df['content'])
    
    # Sanity check alignment
    if posts_df.shape[0] != post_vectors.shape[0]:
        print("\nCRITICAL ERROR: Mismatch between DataFrame rows and vector matrix. Exiting.")
        return

    # --- 5. Evaluate Model ---
    evaluate_model(users_df, posts_df, vectorizer, post_vectors)
    
    # --- 6. Generate and Save Final Recommendations ---
    print("\nGenerating final recommendations...")
    target_user_ids = users_df['user_id'].unique()[:50]
    
    # Build a map of all posts a user has liked to exclude them from recommendations
    all_liked_posts_map = {}
    for _, row in posts_df.iterrows():
        if 'post_id' in row and pd.notna(row['post_id']):
            for user_id in row['like_user_ids_list']:
                 all_liked_posts_map.setdefault(user_id, set()).add(row['post_id'])

    final_recs = []
    for user_id in target_user_ids:
        exclude_ids = all_liked_posts_map.get(user_id, set())
        recs = get_recommendations(user_id, users_df, posts_df, vectorizer, post_vectors, exclude_post_ids=exclude_ids)
        if recs:
            final_recs.append({
                'user_id': user_id,
                'recommended_post_ids': ','.join(map(str, recs))
            })

    # Save to CSV
    if final_recs:
        recommendations_df = pd.DataFrame(final_recs)
        output_filename = 'recommendations_final_semantic.csv'
        recommendations_df.to_csv(output_filename, index=False)
        print(f"\nSuccessfully generated recommendations for {len(recommendations_df)} users.")
        print(f"Output saved to '{output_filename}'")
        print("\nFirst 5 recommendations:")
        print(recommendations_df.head())
    else:
        print("\nNo recommendations were generated.")

if __name__ == "__main__":
    main()