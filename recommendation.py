import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

def data_quality_check(df, df_name):
    """
    Performs and prints a basic data quality check on a dataframe.
    - Checks for missing values.
    - Provides descriptive statistics for numeric columns.
    """
    print(f"\n--- Data Quality Check for {df_name} ---")
    print(f"Shape: {df.shape}")
    
    print("\nMissing Values per Column:")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        print("No missing values found.")
    else:
        print(missing_values[missing_values > 0])
        
    print("\nDescriptive Statistics for Numeric Columns:")
    print(df.describe())
    print(f"--- End of Data Quality Check for {df_name} ---")


def preprocess_users(df):
    """
    Preprocesses the users dataframe.
    - Fills NaN values in 'interested_in' with empty strings.
    - Splits the 'interested_in' string into a list of topics.
    """
    df['interested_in'] = df['interested_in'].fillna('').astype(str)
    df['interested_in_list'] = df['interested_in'].apply(lambda x: [topic.strip() for topic in x.split(',') if topic.strip()])
    return df

def preprocess_posts(df):
    """
    Preprocesses the posts dataframe.
    - Fills NaN values in key columns.
    - Converts date columns to datetime objects.
    - Handles outliers in 'likes' and 'shares' using Z-score capping.
    - Parses 'like_user_ids' into a list of integers.
    """
    df['topics'] = df['topics'].fillna('').astype(str)
    df['like_user_ids'] = df['like_user_ids'].fillna('').astype(str)
    df['content'] = df['content'].fillna('').astype(str)
    df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce') # Ensure user_id is numeric

    # Convert created_at to datetime objects, handling potential errors
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    # Drop rows where date conversion failed
    df.dropna(subset=['created_at'], inplace=True)
    
    # Convert likes and shares to numeric, coercing errors
    df['likes'] = pd.to_numeric(df['likes'], errors='coerce').fillna(0)
    df['shares'] = pd.to_numeric(df['shares'], errors='coerce').fillna(0)

    # --- Outlier Detection and Capping using Z-Score ---
    # This prevents extremely viral posts from dominating all recommendations.
    print("\nChecking for outliers in 'likes' and 'shares'...")
    for col in ['likes', 'shares']:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].std() > 0:
            mean = df[col].mean()
            std = df[col].std()
            z_scores = (df[col] - mean) / std
            
            # Define outlier threshold (e.g., z-score > 3)
            threshold = 3
            outlier_mask = np.abs(z_scores) > threshold
            
            if outlier_mask.any():
                print(f"Found and capped {outlier_mask.sum()} outliers in the '{col}' column.")
                # Calculate the cap value (mean + 3 * std)
                cap_value = mean + threshold * std
                df.loc[outlier_mask, col] = cap_value
    
    df['like_user_ids_list'] = df['like_user_ids'].apply(
        lambda x: [int(uid) for uid in x.split(',') if uid.isdigit()] if x else []
    )
    
    return df

def infer_topics(content, topic_keywords):
    """
    Infers topics for a post based on its content by searching for keywords.
    """
    inferred_topics = set()
    content_lower = content.lower()
    for topic, keywords in topic_keywords.items():
        if any(keyword in content_lower for keyword in keywords):
            inferred_topics.add(topic)
    return list(inferred_topics)

def apply_topic_inference(df):
    """
    Applies topic inference to the posts dataframe, filling in missing topics.
    """
    topic_keywords = {
        'Investing 101': ['ipo', 'sip', 'allotment', 'f&o', 'nifty 50', 'sensex', 'etf', 'mutual fund', 'demat', 'stock', 'shares', 'equity'],
        'Market Trends': ['market', 'gdp', 'inflation', 'economy', 'fii', 'dii', 'bull', 'bear', 'valuation'],
        'Personal Finance': ['tax', 'gst', 'loan', 'emi', 'insurance', 'credit card', 'sip', 'save', 'income'],
        'News': ['news', 'modi', 'trump', 'rbi', 'sebi', 'government', 'policy', 'tariff'],
        'Memes': ['lol', 'lmao', 'meme', 'funny', '🤣', '😂'],
        'Money': ['money', 'wealth', 'rupees', 'crore', 'lakh', 'cashback', 'income'],
        'Learn': ['learn', 'lesson', 'guide', 'explainer'],
        'Alternate Investments': ['crypto', 'gold', 'silver', 'real estate', 'nft'],
        'Shopping': ['amazon', 'flipkart', 'shopping', 'order']
    }

    def get_post_topics(row):
        existing_topics = [t.strip() for t in row['topics'].split(',') if t.strip()]
        inferred = infer_topics(row['content'], topic_keywords)
        # Combine and remove duplicates, maintaining order
        all_topics = list(dict.fromkeys(existing_topics + inferred))
        return all_topics

    df['all_topics'] = df.apply(get_post_topics, axis=1)
    return df

def calculate_popularity_recency_scores(df):
    """
    Calculates popularity and recency scores for each post.
    """
    # Popularity Score (normalized between 0 and 1 for each component)
    max_likes = df['likes'].max()
    max_shares = df['shares'].max()
    
    # Avoid division by zero if max is 0
    df['popularity_score'] = (df['likes'] / (max_likes if max_likes > 0 else 1)) + \
                             (df['shares'] / (max_shares if max_shares > 0 else 1))
    
    # Recency Score (Exponential Decay)
    now = datetime.now(timezone.utc)
    # Ensure datetime column is timezone-aware for correct subtraction
    df['created_at_utc'] = df['created_at'].dt.tz_localize('UTC')
    df['days_since_creation'] = (now - df['created_at_utc']).dt.days
    decay_rate = 0.1 # This value can be tuned
    df['recency_score'] = np.exp(-decay_rate * df['days_since_creation'])
    
    return df

def get_recommendations(user_id, users_df, posts_df, vectorizer, post_vectors, exclude_post_ids=set(), top_n=10):
    """
    Generates top N post recommendations for a given user using a hybrid scoring model.
    'exclude_post_ids' allows us to hide certain posts during evaluation.
    """
    try:
        user_row = users_df[users_df['user_id'] == user_id]
        if user_row.empty:
            # print(f"User {user_id} not found.") # Can be noisy, disable for production
            return []
        user_interests = user_row['interested_in_list'].iloc[0]

    except IndexError:
        print(f"Could not process user {user_id}.")
        return []

    # Filter out posts created by the user or in the exclusion list
    candidate_posts = posts_df[(posts_df['user_id'] != user_id) & (~posts_df['post_id'].isin(exclude_post_ids))].copy()
    
    # --- Semantic Similarity Calculation ---
    if user_interests:
        interests_string = ' '.join(user_interests)
        user_vector = vectorizer.transform([interests_string])
        
        post_indices = candidate_posts.index
        candidate_vectors = post_vectors[post_indices.tolist()] # Use .tolist() for robust indexing
        
        sim_scores = cosine_similarity(user_vector, candidate_vectors).flatten()
        candidate_posts['semantic_score'] = sim_scores
        
        keyword_mask = candidate_posts['all_topics'].apply(lambda topics: any(item in user_interests for item in topics))
        relevant_posts = candidate_posts[(candidate_posts['semantic_score'] > 0.05) | keyword_mask]
    else:
        candidate_posts['semantic_score'] = 0
        relevant_posts = candidate_posts

    if relevant_posts.empty:
        relevant_posts = candidate_posts.copy()
        if 'semantic_score' not in relevant_posts.columns:
            relevant_posts['semantic_score'] = 0
            
    # --- Final Hybrid Scoring ---
    relevant_posts['final_score'] = (0.4 * relevant_posts['semantic_score']) + \
                                    (0.4 * relevant_posts['popularity_score']) + \
                                    (0.2 * relevant_posts['recency_score'])

    top_recommendations = relevant_posts.sort_values(by='final_score', ascending=False).head(top_n)
    return top_recommendations['post_id'].tolist()

def evaluate_model(users_df, posts_df, vectorizer, post_vectors):
    """
    Evaluates the recommendation model using Precision@10 and Recall@10.
    """
    print("\n--- Evaluating Model Performance ---")
    precisions = []
    recalls = []
    
    # Find all posts liked by any user to create a global set of liked posts
    all_liked_posts_map = {}
    for _, row in posts_df.iterrows():
        for user_id in row['like_user_ids_list']:
            if user_id not in all_liked_posts_map:
                all_liked_posts_map[user_id] = set()
            all_liked_posts_map[user_id].add(row['post_id'])

    # Evaluate only for users with enough likes to create a test set
    evaluation_users = [uid for uid, pids in all_liked_posts_map.items() if len(pids) > 1]
    
    if not evaluation_users:
        print("Not enough user interaction data to perform evaluation.")
        return

    for user_id in evaluation_users:
        liked_posts = list(all_liked_posts_map[user_id])
        random.shuffle(liked_posts)
        
        # Split liked posts into a training set (to hide) and a test set (to predict)
        split_point = int(len(liked_posts) * 0.8)
        train_likes = set(liked_posts[:split_point])
        test_likes = set(liked_posts[split_point:])

        if not test_likes:
            continue

        # Get recommendations, excluding the 'train' likes from consideration
        recommendations = get_recommendations(user_id, users_df, posts_df, vectorizer, post_vectors, exclude_post_ids=train_likes)
        
        # Calculate hits: intersection of recommendations and the test set
        hits = len(set(recommendations) & test_likes)
        
        # Precision@10
        precision = hits / len(recommendations) if recommendations else 0
        precisions.append(precision)
        
        # Recall@10
        recall = hits / len(test_likes) if test_likes else 0
        recalls.append(recall)

    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    
    print(f"Evaluated on {len(evaluation_users)} users.")
    print(f"Average Precision@10: {avg_precision:.2%}")
    print(f"Average Recall@10:   {avg_recall:.2%}")
    print("------------------------------------")


if __name__ == "__main__":
    # --- 1. Load Data ---
    try:
        users_df = pd.read_csv('users.csv', encoding='latin1')
        posts_df = pd.read_csv('posts.csv', encoding='latin1')
    except FileNotFoundError:
        print("Make sure 'users.csv' and 'posts.csv' are in the same directory.")
        print("And that you have installed scikit-learn: pip install scikit-learn pandas numpy")
        exit()
    except Exception as e:
        print(f"An error occurred: {e}")
        exit()

    # --- 2. Initial Data Quality Check ---
    data_quality_check(users_df, "users.csv")
    data_quality_check(posts_df, "posts.csv")

    # --- 3. Preprocess Data ---
    print("\nPreprocessing data and handling outliers...")
    users_df = preprocess_users(users_df)
    posts_df = preprocess_posts(posts_df)
    
    # CRITICAL FIX: Reset index after cleaning to ensure alignment with matrix rows.
    posts_df.reset_index(drop=True, inplace=True)
    
    # --- 4. Feature Engineering ---
    print("\nEngineering features (topics, scores, and embeddings)...")
    posts_df = apply_topic_inference(posts_df)
    posts_df = calculate_popularity_recency_scores(posts_df)
    
    vectorizer = TfidfVectorizer(stop_words='english', min_df=2)
    post_vectors = vectorizer.fit_transform(posts_df['content'])
    
    # --- 5. Evaluate Model Performance ---
    evaluate_model(users_df, posts_df, vectorizer, post_vectors)
    
    # --- 6. Generate Final Recommendations for Output ---
    print("\nGenerating final recommendations for the top 50 users...")
    target_user_ids = users_df['user_id'].unique()[:50]
    
    all_recommendations = []
    # For the final output, we want to exclude all posts a user has ever liked.
    all_liked_posts_map = {}
    for _, row in posts_df.iterrows():
        for user_id in row['like_user_ids_list']:
            if user_id not in all_liked_posts_map:
                all_liked_posts_map[user_id] = set()
            all_liked_posts_map[user_id].add(row['post_id'])

    for user_id in target_user_ids:
        exclude_ids = all_liked_posts_map.get(user_id, set())
        recs = get_recommendations(user_id, users_df, posts_df, vectorizer, post_vectors, exclude_post_ids=exclude_ids)
        if recs:
            all_recommendations.append({
                'user_id': user_id,
                'recommended_post_ids': ','.join(map(str, recs))
            })

    # --- 7. Save Final Output to CSV ---
    recommendations_df = pd.DataFrame(all_recommendations)
    output_filename = 'recommendations_final_semantic.csv'
    recommendations_df.to_csv(output_filename, index=False)
    
    print(f"\nSuccessfully generated recommendations for {len(recommendations_df)} users.")
    print(f"Output saved to '{output_filename}'")
    print("\nFirst 5 recommendations:")
    print(recommendations_df.head())

