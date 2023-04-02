import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Set up Spotify API credentials
client_id = 'f326b699ccaa451c93b1a043e4de9b53'
client_secret = '6e581c740f62403a801fba1d5996beb3'

# Authenticate with the Spotify API
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


def get_audio_features(playlist_id):
    # Get the tracks in the playlist
    playlist_tracks = sp.playlist_tracks(playlist_id)
    track_ids = [track['track']['id'] for track in playlist_tracks['items']]

    # Fetch audio features for each track
    audio_features = sp.audio_features(track_ids)
    audio_features_df = pd.DataFrame(audio_features)
    audio_features_df.set_index('id', inplace=True)

    return audio_features_df

def recommend_songs(seed_playlist_id, num_recommendations=5):
    audio_features_df = get_audio_features(seed_playlist_id)

    # Select only numeric features
    numeric_features = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
        'duration_ms', 'time_signature'
    ]
    audio_features_df = audio_features_df[numeric_features]

    # Preprocess the audio features
    scaler = StandardScaler()
    audio_features_scaled = scaler.fit_transform(audio_features_df)
    audio_features_scaled_df = pd.DataFrame(audio_features_scaled, columns=numeric_features, index=audio_features_df.index)

    # Fit a k-nearest neighbors model
    knn = NearestNeighbors(n_neighbors=num_recommendations + 1)  # +1 to exclude the song itself from recommendations
    knn.fit(audio_features_scaled)

    # Recommend songs for each track in the seed playlist
    recommendations = {}
    for index, row in audio_features_scaled_df.iterrows():
        distances, indices = knn.kneighbors([row.values.reshape(1, -1).squeeze()])
        recommended_ids = [audio_features_scaled_df.iloc[idx].name for idx in indices.squeeze()[1:]]
        recommendations[index] = recommended_ids

    return recommendations

# Test the music recommendation system
seed_playlist_id = '76AakPVTu1i9r1arGD8Bck'
recommendations = recommend_songs(seed_playlist_id)

for track_id, recommended_ids in recommendations.items():
    track = sp.track(track_id)
    print(f"Song: {track['name']} by {track['artists'][0]['name']}")
    print("Recommended songs:")
    for rec_id in recommended_ids:
        rec_track = sp.track(rec_id)
        print(f"- {rec_track['name']} by {rec_track['artists'][0]['name']}")
    print()