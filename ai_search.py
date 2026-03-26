import psycopg2
from sentence_transformers import SentenceTransformer
import faiss
import torch
from transformers import pipeline

# --- STEP 1: THE BRAIN (Local Model) ---
# Using a small but powerful model that fits on most PCs
model_id = "microsoft/Phi-3-mini-4k-instruct"
pipe = pipeline("text-generation", model=model_id, device_map="auto", torch_dtype="auto")

# --- STEP 2: THE MEMORY (Vector Store) ---
embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(384) # 384 is the size of MiniLM embeddings

# Describe your tables here once. The Vector store will find them automatically.
metadata = [
    "Table 'students' has columns: id, name, total_marks, grade. Use this for student info.",
    "Table 'users' has columns: user_id, username, email, signup_date. Use this for user details.",
    "Table 'exams' has columns: exam_id, student_id, score. Use for test results."
]
embeddings = embedder.encode(metadata)
index.add(embeddings)

def get_relevant_context(user_query):
    # Vector search: Finds which tables are needed for the question
    query_vector = embedder.encode([user_query])
    D, I = index.search(query_vector, 1) 
    return metadata[I[0][0]]

# --- STEP 3: DATABASE EXECUTION ---
def run_sql(query):
    conn = psycopg2.connect(host="localhost", dbname="postgres", user="postgres", password="password")
    cur = conn.cursor()
    cur.execute(query)
    res = cur.fetchall()
    cur.close()
    conn.close()
    return res

# --- STEP 4: THE CHAT INTERFACE ---
def ai_chat():
    print("PostgreSQL AI Assistant Active.")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ['exit', 'quit']: break

        # 1. Get context from Vector Store
        context = get_relevant_context(user_input)
        
        # 2. Ask Model to write SQL based on Vector context
        prompt = f"<|system|>You are a SQL expert. Use this schema: {context}. Respond ONLY with the SQL query.<|end|><|user|>{user_input}<|end|><|assistant|>"
        output = pipe(prompt, max_new_tokens=50, clean_up_tokenization_spaces=True)
        sql_query = output[0]['generated_text'].split("<|assistant|>")[-1].strip()

        # 3. Execute and show results
        print(f"AI generated SQL: {sql_query}")
        try:
            results = run_sql(sql_query)
            print(f"Database Result: {results}")
        except Exception as e:
            print(f"Error executing SQL: {e}")

if __name__ == "__main__":
    ai_chat()