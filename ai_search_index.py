import mysql.connector
from sentence_transformers import SentenceTransformer
import json

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
save_to_ai_index(
    entity_type="product",
    entity_id=101,
    text_to_embed="Vintage 1990s graphic tee, dead stock, size large, mint condition."
)