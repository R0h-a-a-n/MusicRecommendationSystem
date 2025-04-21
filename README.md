# Music Recommendation System

A lyrics-based music recommendation system that uses natural language processing and machine learning to suggest similar songs based on lyrical content and style.

## Features

- Search by song name and artist
- Get personalized music recommendations
- Similarity scores based on lyrical content analysis
- Modern, responsive web interface

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env`:
```
GENIUS_ACCESS_TOKEN=your_genius_api_token
```

5. Download the model cache:
```bash
# Instructions for downloading model cache files
# These files are not included in the repository due to size
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://127.0.0.1:5000
```

3. Enter a song name and optionally an artist name to get recommendations

## Project Structure

- `app.py` - Flask application server
- `model/inference.py` - Recommendation engine and model inference
- `model/preprocess.py` - Data preprocessing and model training
- `templates/index.html` - Web interface
- `requirements.txt` - Python dependencies

## Technologies Used

- Python 3.8+
- Flask
- BERT Topic Modeling
- Sentence Transformers
- NLTK
- Genius API for lyrics
- scikit-learn
- NumPy
- Pandas
