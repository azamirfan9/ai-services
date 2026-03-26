import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Configuration - REPLACE WITH YOUR NEON CONNECTION STRING
# It usually looks like: postgres://user:password@endpoint.aws.neon.tech/neondb?sslmode=require
DB_URL = "postgresql://neondb_owner:npg_uGYOW4y9JzAU@ep-soft-haze-amn07ui9-pooler.c-5.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

# 2. Initialize the AI Model
print("Loading AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_connection():
    """Connects to PostgreSQL and registers the vector type."""
    conn = psycopg.connect(DB_URL)
    # This is critical: it tells Python how to handle the 'vector' data type
    register_vector(conn)
    return conn

def save_to_pg_index(entity_type, entity_id, text_to_embed):
    try:
        vector = model.encode(text_to_embed)
        with get_connection() as conn:
            with conn.cursor() as cur:
                sql = """
                    INSERT INTO ai_search_index (entity_type, entity_id, text_content, embedding_vector)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (entity_type, entity_id) 
                    DO UPDATE SET 
                        text_content = EXCLUDED.text_content,
                        embedding_vector = EXCLUDED.embedding_vector;
                """
                cur.execute(sql, (entity_type, entity_id, text_to_embed, vector))
                conn.commit()
        print(f"✅ Success: Saved {entity_type} ID {entity_id}")
    except Exception as e:
        print(f"❌ Database Error: {e}")

# def search_ai_index(search_query, entity_type_filter=None, limit=3):
#     """Searches the database using the <=> (Cosine Distance) operator."""
#     query_vec = model.encode(search_query)
    
#     with get_connection() as conn:
#         with conn.cursor() as cur:
#             # PostgreSQL does the math! No need to fetch all rows into Python.
#             if entity_type_filter:
#                 sql = """
#                     SELECT text_content, 1 - (embedding_vector <=> %s) AS score
#                     FROM ai_search_index
#                     WHERE entity_type = %s
#                     ORDER BY embedding_vector <=> %s
#                     LIMIT %s;
#                 """
#                 cur.execute(sql, (query_vec, entity_type_filter, query_vec, limit))
#             else:
#                 sql = """
#                     SELECT text_content, 1 - (embedding_vector <=> %s) AS score
#                     FROM ai_search_index
#                     ORDER BY embedding_vector <=> %s
#                     LIMIT %s;
#                 """
#                 cur.execute(sql, (query_vec, query_vec, limit))
            
#             return cur.fetchall()

# def search_ai_index(search_query, limit=5, min_score=0.30): # Lowered to 0.30 to catch the HP
#     query_vec = model.encode(search_query)
    
#     with get_connection() as conn:
#         with conn.cursor() as cur:
#             cur.execute("""
#                 SELECT text_content, 1 - (embedding_vector <=> %s) AS score
#                 FROM ai_search_index
#                 ORDER BY embedding_vector <=> %s
#                 LIMIT %s;
#             """, (query_vec, query_vec, limit))
#             rows = cur.fetchall()

#     # Filter by the new lower threshold
#     return [row for row in rows if row[1] >= min_score]

def search_ai_index(search_query, entity_type_filter=None, limit=5, min_score=0.3):
    # 1. Get the "Important" words from the user's query
    # We ignore small words like 'is', 'the', 'show', 'me'
    stop_words = {'show', 'me', 'details', 'which', 'have', 'give', 'about', 'find', 'the', 'is', 'of'}
    query_words = [w.lower() for w in search_query.split() if w.lower() not in stop_words]
    
    query_vec = model.encode(search_query)
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Step 1: Broad search by Vector + Entity Type
            sql = "SELECT text_content, 1 - (embedding_vector <=> %s) AS score FROM ai_search_index"
            params = [query_vec]
            
            if entity_type_filter:
                sql += " WHERE entity_type = %s"
                params.append(entity_type_filter)
            
            sql += " ORDER BY embedding_vector <=> %s LIMIT 50" # Fetch a larger batch
            params.append(query_vec)
            
            cur.execute(sql, params)
            rows = cur.fetchall()

    # 2. DYNAMIC FILTERING
    # We look for rows that contain the specific keywords the user asked for
    final_results = []
    for row in rows:
        text_content = row[0].lower()
        score = row[1]
        
        if score < min_score:
            continue
            
        # Check: Does the database row contain the specific keywords from the query?
        # e.g. if query has 'Salim', does the row have 'Salim'?
        # e.g. if query has 'A', does the row have 'A'?
        match_count = 0
        for word in query_words:
            if word in text_content:
                match_count += 1
        
        # We prioritize rows that have MORE keyword matches
        # This keeps 'Grade A' rows and drops 'Grade B' if 'A' was in the query
        if match_count > 0:
            # We boost the score of rows that have exact keyword matches
            boosted_score = score + (match_count * 0.1)
            final_results.append((row[0], boosted_score))

    # Sort by the new boosted score and return top results
    final_results.sort(key=lambda x: x[1], reverse=True)
    return final_results[:limit]

def generate_human_response(query, db_results):
    if not db_results:
        return f"I couldn't find any laptops or Macbooks matching '{query}'."

    response = f"I found {len(db_results)} items that match your request:\n\n"
    
    for i, row in enumerate(db_results):
        text, score = row[0], round(row[1] * 100, 1)
        # Use a bullet point for every result found
        response += f"• [{score}% Match] {text}\n"
            
    return response

# Ensure these lines have ZERO spaces at the beginning
# save_to_pg_index("student", 101, "John Doe, roll no 111, grade value A")
# save_to_pg_index("student", 102, "Jack Micle, roll no 222, grade value B")
# save_to_pg_index("student", 103, "Salim, roll no 333, grade value A")
# save_to_pg_index("student", 104, "Akbar, roll no 44, grade value B")
# save_to_pg_index("student", 105, "Mukesh, roll no 555, grade value C")
#save_to_pg_index("product", 102, "Apple MacBook Air M3, 8GB RAM, 256GB SSD, Space Gray")

query = "Show me student details which have grade A"
raw_results = search_ai_index(query)

print("\n--- HUMAN RESPONSE ---")
print(generate_human_response(query, raw_results))