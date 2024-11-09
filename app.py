import gradio as gr
import plotly.express as px
import geopandas as gpd
import tensorflow_hub as hub
import pandas as pd
import tensorflow as tf
import pickle
import numpy as np
import librosa
import logging
from pydub import AudioSegment

# Configure logging to capture information and error messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File paths for resources
SHAPEFILE_PATH = "map/ne_110m_admin_0_countries.shp"  # Shapefile for map visualization
BIRD_DATA_FILE = "Birds Voice.csv"                    # CSV file containing bird data
MODEL_PATH = 'yamnet_classifier_model.h5'             # Path to trained classifier model
LABEL_ENCODER_PATH = 'label_encoder.pkl'               # Path to label encoder for class labels
YAMNET_MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"  # URL for YAMNet model on TensorFlow Hub

# Load bird audio data
try:
    bird_data = pd.read_csv(BIRD_DATA_FILE)
    logging.info("Bird audio data loaded successfully.")
except Exception as e:
    logging.error("Error loading bird audio data: %s", e)
    raise

# Load the trained classifier model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("Trained model loaded successfully.")
except Exception as e:
    logging.error("Error loading the trained model: %s", e)
    raise

# Load the label encoder
try:
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    logging.info("Label encoder loaded successfully.")
except Exception as e:
    logging.error("Error loading label encoder: %s", e)
    raise

# Load YAMNet model from TensorFlow Hub
try:
    yamnet_model = hub.load(YAMNET_MODEL_HANDLE)
    logging.info("YAMNet model loaded successfully.")
except Exception as e:
    logging.error("Error loading YAMNet model: %s", e)
    raise

def load_audio_file(filepath):
    """
    Load an audio file and resample it to 16kHz.

    Parameters:
    - filepath (str): Path to the audio file

    Returns:
    - np.array: Audio data array
    """
    try:
        audio, sample_rate = librosa.load(filepath, sr=16000)
        return audio
    except Exception as e:
        logging.error("Error loading audio file: %s", e)
        raise

def yamnet_embeddings(audio_data):
    """
    Extract embeddings from audio data using the YAMNet model.

    Parameters:
    - audio_data (np.array): Audio data array

    Returns:
    - np.array: YAMNet embeddings
    """
    try:
        audio_data = np.squeeze(audio_data)  # Remove batch dimension if present
        scores, embeddings, spectrogram = yamnet_model(audio_data)
        return embeddings.numpy()
    except Exception as e:
        logging.error("Error extracting YAMNet embeddings: %s", e)
        raise

def prepare_test_data(audio):
    """
    Prepare the test data by converting an MP3 file to 16kHz, mono .wav format and extracting embeddings.

    Parameters:
    - audio (str): File path of the audio to process

    Returns:
    - str: Predicted bird species name
    """
    try:
        # Load and preprocess audio file
        sound = AudioSegment.from_mp3(audio)
        sound = sound.set_frame_rate(16000)
        sound = sound.set_channels(1)
        sound.export("wav_test/w_1", format="wav")
        
        # Load audio as numpy array for embedding extraction
        audio_data = load_audio_file("wav_test/w_1")
        audio_data = np.reshape(audio_data, (1, -1))  # Add batch dimension
        embeddings = yamnet_embeddings(audio_data)
        
        # Compute the mean embedding for classification
        mean_embedding = np.mean(embeddings, axis=0).reshape(1, -1)
        
        # Predict the class label
        prediction = model.predict(mean_embedding)
        predicted_class_idx = prediction.argmax(axis=1)
        
        # Return the human-readable bird species name
        return label_encoder.inverse_transform(predicted_class_idx)[0]
    except Exception as e:
        logging.error("Error preparing test data: %s", e)
        raise

def create_map(audio, country_spotted):
    """
    Generate a choropleth map showing the countries visited by a bird species identified from audio.

    Parameters:
    - audio (str): File path of the bird's audio
    - country_spotted (str): Country where the bird was spotted

    Returns:
    - tuple: Plotly figure, bird common name, scientific name, countries seen, HTML alert, analysis text
    """
    try:
        # Predict bird name and fetch related data
        bird_name = prepare_test_data(audio)
        scientific_name = bird_data[bird_data.common_name == bird_name].scientific_name.iloc[0]
        countries_seen = list(bird_data[bird_data.common_name == bird_name].Country.unique())
        
        # Load and filter world map shapefile
        world = gpd.read_file(SHAPEFILE_PATH)
        highlighted_countries = world[world['NAME'].isin(countries_seen)]
        
        # Create map with Plotly
        fig = px.choropleth(
            highlighted_countries,
            locations='NAME',
            locationmode='country names',
            color='NAME',
            title="Countries visited by bird in its usual fly path"
        )
        fig.update_layout(width=800, height=500, legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, font=dict(size=10)))
        
        # Analyze if bird is in its migratory path
        if country_spotted in countries_seen:
            analysis = "Bird spotted in its expected country."
            color = "lightgreen"
            alert_flag = "Spotted in its expected migratory route"
        else:
            analysis = "Bird spotted in a country outside its migratory path, possibly due to climate change or habitat loss."
            color = "#FF073A"
            alert_flag = "ALERT: Spotted in a new country"
        
        # Format alert for display
        alert_html = f"<span style='color: {color}; font-size: 20px'>{alert_flag}</span>"
        
        return fig, bird_name, scientific_name, countries_seen, alert_html, analysis
    except Exception as e:
        logging.error("Error creating map: %s", e)
        raise

# Set up Gradio interface
iface = gr.Interface(
    fn=create_map,
    inputs=[
        gr.Audio(type="filepath", label="Bird Sound Audio"),
        gr.Textbox(label="Enter Country of Spotting"),
    ],
    outputs=[
        gr.Plot(),
        gr.Textbox(label="Bird Name"),
        gr.Textbox(label="Scientific Name"),
        gr.Textbox(label="Countries of Migration"),
        gr.HTML(label="Alert"),
        gr.Textbox(label="Analysis")
    ],
    examples=[
        ["audio_inputs/Andean Tinamou2.mp3", "Japan"],
        ["audio_inputs/Andean Guan8.mp3", "Brazil"],
        ["audio_inputs/Common Ostrich2.mp3", "South Africa"]
    ],
    title="Migratory Bird Tracking via Audio Identification",
    description="Upload an audio sample of a bird's sound along with the country of spotting. Audio file should be between 5 to 20 seconds."
)

if __name__ == "__main__":
    logging.info("Launching Gradio interface.")
    iface.launch()
