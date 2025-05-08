from flask import Flask, request, jsonify, render_template
from model.inference import recommend_similar_songs
import os
from dotenv import load_dotenv
import requests

load_dotenv()

app = Flask(__name__, template_folder="templates")
app.secret_key = "replace_this_with_a_secret_key"

YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/search-youtube', methods=['POST'])
def search_youtube():
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400

        # Search YouTube
        response = requests.get(
            'https://www.googleapis.com/youtube/v3/search',
            params={
                'part': 'snippet',
                'q': query,
                'key': YOUTUBE_API_KEY,
                'type': 'video',
                'maxResults': 1
            }
        )
        
        data = response.json()
        
        if 'items' in data and len(data['items']) > 0:
            video_id = data['items'][0]['id']['videoId']
            return jsonify({'videoId': video_id})
        else:
            return jsonify({'error': 'No videos found'}), 404

    except Exception as e:
        print(f"🔥 Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        song_name = data.get('track_id')  # frontend uses 'track_id', we interpret as 'song name'
        if not song_name:
            return jsonify({'error': 'Missing song name'}), 400

        print("🎯 Getting recommendations for:", song_name)
        recommendations = recommend_similar_songs(song_name)
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        print(f"🔥 Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/me')
def me():
    # Simulated "logged-in" check for frontend visibility
    return jsonify({'status': 'logged in'})

if __name__ == '__main__':
    app.run(debug=True)
