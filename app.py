import pandas as pd
from flask import Flask, jsonify, render_template

app = Flask(__name__)

# --- Load Data on Server Start ---
rec_df = pd.read_csv('recommendations_final_semantic.csv')
recommendations_data = {
    str(row['user_id']): row['recommended_post_ids'].split(',') 
    for _, row in rec_df.iterrows()
}

posts_df = pd.read_csv('posts_clean.csv')
posts_df = posts_df[['post_id', 'content']].dropna()
posts_data = {str(row['post_id']): row['content'] for _, row in posts_df.iterrows()}

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    return jsonify({'recommendations': recommendations_data, 'posts': posts_data})

# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True)
