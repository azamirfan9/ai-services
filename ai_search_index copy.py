import mysql.connector
from sentence_transformers import SentenceTransformer
import json
import numpy as np

# 1. Load Model Globally
print("Loading AI Model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ai_services"
    )

def save_to_ai_index(entity_type, entity_id, text_to_embed):
    """Generates a vector and saves a single record to MySQL."""
    db = get_db_connection()
    cursor = db.cursor()

    # Generate Vector
    vector = model.encode(text_to_embed).tolist() 

    sql = """
        INSERT INTO ai_search_index (entity_type, entity_id, text_content, embedding_vector)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
        text_content = VALUES(text_content), 
        embedding_vector = VALUES(embedding_vector)
    """
    values = (entity_type, entity_id, text_to_embed, json.dumps(vector))

    cursor.execute(sql, values)
    db.commit()
    cursor.close()
    db.close()
    print(f"✅ Saved {entity_type} ID {entity_id}")

def search_ai_index_fast(search_query, entity_type_filter=None, limit=5, min_threshold=0.4):
    """
    High-performance search using NumPy matrix math.
    """
    print(f"\n🔍 Searching for: '{search_query}'...")
    
    # 1. Vectorize the search query
    query_vec = model.encode(search_query)

    # 2. Fetch data from MySQL
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    
    if entity_type_filter:
        cursor.execute("SELECT * FROM ai_search_index WHERE entity_type = %s", (entity_type_filter,))
    else:
        cursor.execute("SELECT * FROM ai_search_index")
        
    rows = cursor.fetchall()
    cursor.close()
    db.close()

    if not rows:
        return []

    # 3. FAST MATRIX MATH (The "Optimization")
    # Convert all DB strings back into a single NumPy matrix
    all_vectors = np.array([json.loads(r['embedding_vector']) for r in rows])
    
    # Calculate Cosine Similarity for all rows at once
    # Formula: (A • B) / (||A|| * ||B||)
    dot_product = np.dot(all_vectors, query_vec)
    norms = np.linalg.norm(all_vectors, axis=1) * np.linalg.norm(query_vec)
    similarities = dot_product / norms

    # 4. Filter and Sort
    scored_results = []
    for i, score in enumerate(similarities):
        if score >= min_threshold:
            rows[i]['similarity_score'] = float(score)
            scored_results.append(rows[i])

    # Sort by highest score
    scored_results.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return scored_results[:limit]

# --- EXAMPLE USAGE ---

# 1. Insert/Update Alex Chen (or any other data)
# save_to_ai_index("user", 405, "Alex Chen. Senior Software Engineer in London. Expert in Python.")

# 2. Perform Fast Search
results = search_ai_index_fast("Is there the details of HP Pavilion and Apple MacBook?", limit=3)

print("\n--- Top Results ---")
for res in results:
    score = round(res['similarity_score'] * 100, 2)
    print(f"[{score}% Match] {res['text_content'][:70]}...")