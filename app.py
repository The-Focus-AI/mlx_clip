import os
import psycopg2
import mlx_clip
from dotenv import load_dotenv
from flask import Flask, request, render_template_string, url_for, redirect

# Load environment variables
load_dotenv()
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

# Initialize the mlx_clip model
clip = mlx_clip.mlx_clip("mlx_model")

IMAGE_DIR = "assets/"

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Image Search</title>
    <style>
        body { font-family: sans-serif; margin: 40px; }
        .search-bar { margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
        .grid img { cursor: pointer; border: 2px solid #eee; border-radius: 8px; transition: border 0.2s; max-width: 100%; max-height: 300px; }
        .grid img:hover { border: 2px solid #007bff; }
    </style>
</head>
<body>
    <form class="search-bar" method="get" action="/">
        <input type="text" name="q" value="{{ query|default('') }}" placeholder="Search for images..." style="width: 300px; padding: 8px; font-size: 16px;">
        <button type="submit" style="padding: 8px 16px; font-size: 16px;">Search</button>
    </form>
    <div class="grid">
        {% for file_name in images %}
            <a href="{{ url_for('neighbors', file_name=file_name) }}"><img src="{{ url_for('static', filename='assets/' + file_name) }}" alt="{{ file_name }}"></a>
        {% endfor %}
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET'])
def index():
    query = request.args.get('q', '').strip()
    images = []
    if query:
        text_embedding = clip.text_encoder(query)
        embedding_list = text_embedding.tolist() if hasattr(text_embedding, 'tolist') else list(text_embedding)
        try:
            conn = psycopg2.connect(SUPABASE_DB_URL)
            cur = conn.cursor()
            sql = """
                SELECT file_name, embedding <#> %s::vector AS distance
                FROM image_embeddings
                ORDER BY distance ASC
                LIMIT 9;
            """
            cur.execute(sql, (embedding_list,))
            results = cur.fetchall()
            images = [row[0] for row in results]
            cur.close()
            conn.close()
        except Exception as e:
            images = []
    return render_template_string(HTML_TEMPLATE, images=images, query=query)

@app.route('/neighbors/<file_name>')
def neighbors(file_name):
    # Get the embedding for the selected image
    embedding = None
    try:
        conn = psycopg2.connect(SUPABASE_DB_URL)
        cur = conn.cursor()
        cur.execute("SELECT embedding FROM image_embeddings WHERE file_name = %s LIMIT 1;", (file_name,))
        row = cur.fetchone()
        if row:
            embedding = row[0]
        cur.close()
        conn.close()
    except Exception as e:
        embedding = None
    images = []
    if embedding:
        try:
            conn = psycopg2.connect(SUPABASE_DB_URL)
            cur = conn.cursor()
            sql = """
                SELECT file_name, embedding <#> %s::vector AS distance
                FROM image_embeddings
                WHERE file_name != %s
                ORDER BY distance ASC
                LIMIT 9;
            """
            cur.execute(sql, (embedding, file_name))
            results = cur.fetchall()
            images = [row[0] for row in results]
            cur.close()
            conn.close()
        except Exception as e:
            images = []
    return render_template_string(HTML_TEMPLATE, images=images, query=None)

if __name__ == '__main__':
    # Serve images from the assets directory as static files
    app.static_folder = '.'
    app.run(debug=True) 