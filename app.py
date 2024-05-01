# app.py

from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
app = Flask(__name__)

# Load the data and perform necessary preprocessing
df = pd.read_csv("Music_Dataset.csv")
# Perform data preprocessing steps here...
df.dropna(inplace=True)
df=df.drop_duplicates()
l = []
for i in df['User-Rating']:
    l.append(i[:3])
df['User-Rating'] = l
# Removing white spaces from 'Album/Movie' column
df['Album/Movie'] = df['Album/Movie'].str.replace(' ', '')

# Removing white spaces from 'Singer/Artists' column
df['Singer/Artists'] = df['Singer/Artists'].str.replace(' ', '')
df['Singer/Artists'] = df['Singer/Artists'].str.replace(',', ' ')
df['tags'] = df['Singer/Artists'] + ' ' + df['Genre'] + ' ' + df['Album/Movie'] + ' ' + df['User-Rating']
new_df = df[['Song-Name', 'tags']]

# Text Vectorization using CountVectorizer
cv = CountVectorizer(max_features=2000)
vectors = cv.fit_transform(new_df['tags']).toarray()

# Calculate Cosine Similarity Matrix
similarity = cosine_similarity(vectors)

# Get the list of music titles
music_titles = new_df['Song-Name'].tolist()  # Define and populate music_titles variable

def recommend(music):
    music_index = new_df[new_df['Song-Name'] == music].index[0]
    distances = similarity[music_index]
    music_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:10]
    recommendations = []
    for i in music_list:
        recommended_music_index = i[0]
        similarity_score = i[1]
        recommended_music_title = new_df.iloc[recommended_music_index]['Song-Name']
        recommendations.append((recommended_music_title, similarity_score))
    return recommendations

@app.route('/')
def home():
    return render_template('index.html', music_titles=music_titles)  # Pass music_titles to template

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    music_title = request.form['music_title']
    recommendations = list(recommend(music_title))  # Convert recommendations tuple to a list

    # Fetch additional details from JioSaavn API for each recommendation
    for i, recommendation in enumerate(recommendations):
        query = recommendation[0]  # Song title
        response = requests.get(f'https://saavn.dev/api/search?query={query}')
        data = response.json()

        # Print the response for debugging
        # print(data)

        # Check if the response contains the expected data structure
        if 'success' in data and data.get('success') == True:
            results = data.get('data').get('songs').get('results')
            if len(results) > 0:
                song_data = results[0]
                poster = song_data.get('image')[2]['url'] if 'image' in song_data else ''  # Song poster
                artists = [artist['name'] for artist in song_data['artists']['primary']] if 'artists' in song_data else []  # Artists
                artist = ', '.join(artists)
                listen_link = song_data.get('url', '')  # Listen link
                # Modify the recommendation tuple
                recommendations[i] = (recommendation[0], recommendation[1], poster, artist, listen_link)
            else:
             # Handle the case where no results are found
                recommendations[i] = (recommendation[0], recommendation[1], '', '', '')
        else:
            # Handle the case where the response does not contain the expected data
            recommendations[i] = (recommendation[0], recommendation[1], '', '', '')
            # print('false')

    
    return render_template('recommendations.html', recommendations=recommendations)





if __name__ == '__main__':
    app.run(debug=True)
