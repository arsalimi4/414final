import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load the CSV file
file_path = 'amazon.csv'  # Update with the correct path to your file
df = pd.read_csv(file_path)

# Combine the text features into one for vectorization
df['combined_text'] = df['about_product'] + " " + df['review_title'] + " " + df['review_content']

# Vectorize the combined text using CountVectorizer
vectorizer = CountVectorizer()
text_features = vectorizer.fit_transform(df['combined_text'])

# Create a DataFrame from the text features
df_text_features = pd.DataFrame(text_features.toarray(), columns=vectorizer.get_feature_names_out())

# Combine the text features with the numerical features (excluding 'product_id', 'product_name', 'price', and 'discount_price')
df_combined = pd.concat([df[['discount_percentage', 'rating', 'rating_count']], df_text_features], axis=1)

# Standardize the numerical features
scaler = StandardScaler()
numerical_features_scaled = scaler.fit_transform(df_combined[['discount_percentage', 'rating', 'rating_count']])
df_combined[['discount_percentage', 'rating', 'rating_count']] = numerical_features_scaled

# Compute Cosine similarity for all products
cosine_similarities = cosine_similarity(df_combined)

# Choose a product to find similarities with (e.g., the first product)
chosen_product_index = 0
similarity_scores = cosine_similarities[chosen_product_index]

# Sort the similarity scores in descending order
sorted_indices = similarity_scores.argsort()[::-1]

# Get the top 10 most similar products (excluding the chosen product itself)
top_10_indices = [i for i in sorted_indices if i != chosen_product_index][:10]
top_10_products = df.loc[top_10_indices, ['product_id', 'product_name', 'discount_percentage', 'rating', 'rating_count', 'about_product', 'review_title', 'review_content']]

print(top_10_products)