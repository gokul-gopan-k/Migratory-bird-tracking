import gradio as gr
import plotly.express as px
import geopandas as gpd
import tensorflow_hub as hub
import pandas as pd
import tensorflow as tf
import pickle
import numpy as np
from pydub import AudioSegment
import librosa
# Path to the downloaded shapefile (replace with your local path)
shapefile_path = "map/ne_110m_admin_0_countries.shp"

# Sample countries to highlight
countries_to_highlight = ["United States of America", "Brazil", "India"]
data_csv =pd.read_csv("Birds Voice.csv")
model_3 = tf.keras.models.load_model('yamnet_classifier_model.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder_3 = pickle.load(f)

# Load YAMNet model
yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(yamnet_model_handle)


def load_audio_file(filepath):
    y, sr = librosa.load(filepath, sr=16000)  # Load with 16kHz sampling rate
    print(f"Audio data shape: {y.shape}")
    return y
def yamnet_embeddings(audio_data):
    audio_data = np.squeeze(audio_data) 
    scores, embeddings, spectrogram = yamnet_model(audio_data)
    return embeddings.numpy()  # Shape: (num_patches, 1024)

def test_inf(audio):
    sound = AudioSegment.from_mp3(audio)
    sound = sound.set_frame_rate(16000)  # Resample to 16kHz
    sound = sound.set_channels(1)        # Convert to mono
    sound.export("wav_test/w_1", format="wav")
    # Prepare the dataset by extracting embeddings
    all_embeddings = []
    audio_data = load_audio_file("wav_test/w_1")
    audio_data = np.reshape(audio_data, (1, -1))  # Reshape for batch dimension
    embeddings = yamnet_embeddings(audio_data)
    all_embeddings.append(np.mean(embeddings, axis=0))  # Use mean embedding
    X = np.array(all_embeddings) 
    predd = model_3.predict(X)
    predd_class_indices = predd.argmax(axis=1)
    # Now pass these indices to inverse_transform
    return label_encoder_3.inverse_transform(predd_class_indices)[0]


# Create world map using Plotly
def create_map(audio,country_spotted):

    c_name = data_csv[data_csv.common_name == test_inf(audio)].scientific_name.iloc[0]
    countries_seen = list(data_csv[data_csv.common_name == test_inf(audio)].Country.unique())
    
    # Load the world map from the shapefile
    world = gpd.read_file(shapefile_path)

    # Highlight the selected countries
    highlighted_countries = world[world['NAME'].isin(countries_seen)]
    
    # Plot the map with highlighted countries
    fig = px.choropleth(
        highlighted_countries,
        locations='NAME',
        locationmode='country names',
        color='NAME',
        title="Countries visited by bird in its usual fly path"
    )
    # Increase map size and reduce legend size
    fig.update_layout(
        width=800,  # Map width
        height=500,  # Map height
        legend=dict(
            orientation="h",   # Horizontal legend
            yanchor="top",     # Align legend at the top
            y=-0.2,            # Position legend below the map
            xanchor="center",  # Center align legend
            x=0.5,             # Center it horizontally
            font=dict(size=10)  # Legend font size
        )
       
        )
    if country_spotted in countries_seen:
        analysis = "Bird spotted in its expected country."
    else:
        analysis = "Bird spotted in a country outside its migratory path. There is likely change in migratory path due to reasons such as climate change or deforestation."
    return fig,test_inf(audio), c_name, countries_seen, analysis

# Create Gradio interface
iface = gr.Interface(
    fn=create_map,
    inputs=[ gr.Audio(type="filepath", label="Andean Tinamou Sound"),
             gr.Textbox(value="Japan", label="Enter Country of Spotting"),
            ],
    outputs=[gr.Plot(), gr.Textbox(value="Andean", label="Audio belongs to the bird"), gr.Textbox(value="Andean", label="Scientific name is"), 
             gr.Textbox(value="India, Australia", label="Countries of Migration are"),
             gr.Textbox(value="Bird sighted in an expected country", label="Analysis")
            ],
    examples = [["audio_inputs/Andean Tinamou2.mp3"],["audio_inputs/Andean Guan8.mp3"],[ "audio_inputs/Common Ostrich2.mp3"]],
    title="Migratory Birds tracking using audio signals",
    description="Upload audio of bird spotted along with country of spotting. Audio file should be around 5 to 20 seconds."
)

iface.launch()

