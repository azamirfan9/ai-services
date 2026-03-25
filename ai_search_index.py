import mysql.connector
from sentence_transformers import SentenceTransformer
import json
import numpy as np

# 1. Load the free AI model globally (so it only downloads/loads once)  
print("Loading AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def save_to_ai_index(entity_type, entity_id, text_to_embed):
    """
    Converts text to a vector and saves it to the global MySQL search index.
    """
    # 2. Connect to your existing MySQL database
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ai_services"
    )
    cursor = db.cursor()

    # 3. Turn the text into a mathematical vector
    print(f"Generating vector embedding for {entity_type} ID: {entity_id}...")
    vector = model.encode(text_to_embed).tolist() 

    # 4. Insert the dynamic tags, Text, and Vector into the global table
    sql = """
        INSERT INTO ai_search_index (entity_type, entity_id, text_content, embedding_vector)
        VALUES (%s, %s, %s, %s)
    """
    values = (entity_type, entity_id, text_to_embed, json.dumps(vector))

    cursor.execute(sql, values)
    db.commit()

    print(f"Success! {entity_type.capitalize()} ID {entity_id} saved to the AI brain.\n")

    # Close the connection
    cursor.close()
    db.close()


# --- Testing the Global Brain ---

# Example A: Vectorizing a dead stock product
# save_to_ai_index(
#     entity_type="product",
#     entity_id=102,
#     text_to_embed="HP Pavilion 15.6 inch Laptop, Intel Core i5, 8GB RAM"
# )

# save_to_ai_index(
#     entity_type="product",
#     entity_id=103,
#     text_to_embed="Apple MacBook Air M2, 13-inch, 256GB SSD"
# )

# Example B: Vectorizing a User profile
# save_to_ai_index(
#     entity_type="user",
#     entity_id=405, # The user's ID in your 'Users' table
#     text_to_embed="Alex Chen. Senior Software Engineer based in London. Expert in Python, React, and cloud architecture. Passionate about building accessible web applications and hiking."
# )

# Example C: Vectorizing a Student record
# save_to_ai_index(
#     entity_type="student",
#     entity_id=90210, # The student's ID in your 'Students' table
#     text_to_embed="Sarah Jenkins. Junior majoring in Biology with a minor in Data Science. Interested in genetics, bioinformatics, and marine ecosystems. Looking for study partners for Organic Chemistry."
# )


# Example D: Vectorizing a Course syllabus
# save_to_ai_index(
#     entity_type="course",
#     entity_id=301, # The course ID in your 'Course' table
#     text_to_embed="CS301: Introduction to Artificial Intelligence. This course covers machine learning, neural networks, and natural language processing. Prerequisites: Python programming and basic linear algebra."
# )

# Example E: Vectorizing a Software Product
# save_to_ai_index(
#     entity_type="product",
#     entity_id=88, 
#     text_to_embed="Pro Photo Editor v2.0. Digital software for Mac and Windows. Features AI background removal, batch processing, and RAW image support. Lifetime license."
# )


def cosine_similarity(vec1, vec2):
    """Calculates the similarity between two vectors. Returns a score from -1 to 1."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def search_ai_index(search_query, entity_type_filter=None, limit=5, min_threshold=0.7):
    """
    Searches the database and only returns results that meet a minimum 
    similarity score (default 70%).
    """
    print(f"\nSearching for: '{search_query}'...")
    query_vector = model.encode(search_query).tolist()

    db = mysql.connector.connect(
        host="localhost", user="root", password="", database="ai_services"
    )
    cursor = db.cursor(dictionary=True)

    if entity_type_filter:
        sql = "SELECT * FROM ai_search_index WHERE entity_type = %s"
        cursor.execute(sql, (entity_type_filter,))
    else:
        sql = "SELECT * FROM ai_search_index"
        cursor.execute(sql)
        
    results = cursor.fetchall()
    cursor.close()
    db.close()

    scored_results = []
    for row in results:
        db_vector = json.loads(row['embedding_vector'])
        score = cosine_similarity(query_vector, db_vector)
        
        # --- THE FIX: Only keep matches that actually make sense ---
        if score >= min_threshold:
            row['similarity_score'] = score
            scored_results.append(row)

    scored_results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return scored_results[:limit]


# --- Testing the Search ---

# Example 1: Search specifically within "products"
search_results = search_ai_index(
    search_query="Looking for an HP Laptop", 
    entity_type_filter="product", # Here is your filter!
    limit=3
)

print("\n--- Top Search Results ---")
for match in search_results:
    # We round the score to 2 decimal places for readability
    score_percentage = round(match['similarity_score'] * 100, 2)
    print(f"[{score_percentage}% Match] Type: {match['entity_type']} | ID: {match['entity_id']}")
    print(f"Text: {match['text_content']}\n")