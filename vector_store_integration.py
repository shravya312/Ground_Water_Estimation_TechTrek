
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the cleaned groundwater data
try:
    df = pd.read_csv("cleaned_groundwater_data.csv")
except FileNotFoundError:
    print("Error: cleaned_groundwater_data.csv not found. Please run excel_parser.py first.")
    exit()

# Combine relevant columns into a single text string for embedding
# This will be used to represent each row in the vector store
df['combined_text'] = df['STATE'].fillna('') + " " + \
                         df['DISTRICT'].fillna('') + " " + \
                         df['ASSESSMENT UNIT'].fillna('')

# Generate embeddings for all combined texts
corpus_embeddings = model.encode(df['combined_text'].tolist(), show_progress_bar=True)

def search_groundwater_data(query, top_k=5):
    # Generate embedding for the query
    query_embedding = model.encode(query)

    # Calculate cosine similarity between query embedding and corpus embeddings
    similarities = cosine_similarity([query_embedding], corpus_embeddings)[0]

    # Get the top_k most similar entries
    top_k_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for i in top_k_indices:
        results.append({
            "score": similarities[i],
            "data": df.iloc[i].to_dict()
        })
    return results

if __name__ == "__main__":
    print("Vector store created and embeddings generated.")

    sample_query1 = "Groundwater availability in Andaman and Nicobar Islands"
    print(f"\nSearching for: {sample_query1}")
    search_results = search_groundwater_data(sample_query1)
    for res in search_results:
        print(f"Score: {res['score']:.4f}, State: {res['data']['STATE']}, District: {res['data']['DISTRICT']}, Unit: {res['data']['ASSESSMENT UNIT']}")

    sample_query2 = "Rainfall data for Arunachal Pradesh"
    print(f"\nSearching for: {sample_query2}")
    search_results = search_groundwater_data(sample_query2)
    for res in search_results:
        print(f"Score: {res['score']:.4f}, State: {res['data']['STATE']}, District: {res['data']['DISTRICT']}, Unit: {res['data']['ASSESSMENT UNIT']}")
