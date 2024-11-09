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


# Path to the downloaded shapefile 
shapefile_path = "map/ne_110m_admin_0_countries.shp"

# Bird audio data file
birdData_csv = pd.read_csv("Birds Voice.csv")

# Load trained model
model = tf.keras.models.load_model('yamnet_classifier_model.h5')

# Load label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load YAMNet model for inference
yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
yamnet_model = hub.load(yamnet_model_handle)


def load_audio_file(filepath):
    # Load with 16kHz sampling rate
    audio, sample_rate = librosa.load(filepath, sr=16000)  
    return audio

def yamnet_embeddings(audio_data):
    audio_data = np.squeeze(audio_data) 
    scores, embeddings, spectrogram = yamnet_model(audio_data)
    return embeddings.numpy()  

def prepare_testData(audio):
    sound = AudioSegment.from_mp3(audio)
    # Resample to 16kHz
    sound = sound.set_frame_rate(16000)  
    # Convert to mono
    sound = sound.set_channels(1)        
    sound.export("wav_test/w_1", format="wav")
    # Prepare the dataset by extracting embeddings
    all_embeddings = []
    audio_data = load_audio_file("wav_test/w_1")
     # Reshape for batch dimension
    audio_data = np.reshape(audio_data, (1, -1)) 
    embeddings = yamnet_embeddings(audio_data)
    # Use mean embedding
    all_embeddings.append(np.mean(embeddings, axis=0))  
    X_test = np.array(all_embeddings) 
    pred = model.predict(X_test)
    pred_class_indices = pred.argmax(axis=1)
    # Now pass these indices to inverse_transform
    return label_encoder.inverse_transform(pred_class_indices)[0]


# Create world map using Plotly
def create_map(audio,country_spotted):
    # Get scientific name
    sc_name = birdData_csv [birdData_csv .common_name == prepare_testData(audio)].scientific_name.iloc[0]
    # Get countries seen
    countries_seen = list(birdData_csv [birdData_csv .common_name == prepare_testData(audio)].Country.unique())
    
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
        # Map width
        width=800,  
        # Map height
        height=500,  
        legend=dict(
            # Horizontal legend
            orientation="h",   
            # Align legend at the top
            yanchor="top",    
            # Position legend below the map 
            y=-0.2,         
            # Center align legend   
            xanchor="center",  
            # Center it horizontally
            x=0.5,      
            # Legend font size       
            font=dict(size=10)  
        )      
        )
    # Generate analysis
    if country_spotted in countries_seen:
        analysis = "Bird spotted in its expected country."
        color = "lightgreen"
        alert_flag = " Spotted in its expected migratory route"
    else:
        analysis = "Bird spotted in a country outside its migratory path. There is likely change in migratory path due to reasons such as climate change or deforestation."
        color = "#FF073A"
        alert_flag = "ALERT: Spotted in a new country"
    alert_html = f"<span style='color: {color}; font-size: 20px'>{alert_flag}</span>"    
    return fig,prepare_testData(audio), sc_name, countries_seen, alert_html, analysis

# Create Gradio interface
iface = gr.Interface(
    fn=create_map,
    inputs=[ gr.Audio(type="filepath", label="Andean Tinamou Sound"),
             gr.Textbox(value="Japan", label="Enter Country of Spotting"),
            ],
    outputs=[gr.Plot(), gr.Textbox(value="Andean", label="Audio belongs to the bird"), gr.Textbox(value="Andean", label="Scientific name is"), 
             gr.Textbox(value="India, Australia", label="Countries of Migration are"),  gr.HTML(label="Alert"), 
             gr.Textbox(value="Bird sighted in an expected country", label="Analysis")
            ],
    examples = [["audio_inputs/Andean Tinamou2.mp3"],["audio_inputs/Andean Guan8.mp3"],[ "audio_inputs/Common Ostrich2.mp3"]],
    title="Migratory Birds tracking using audio signals",
    description="Upload audio of bird spotted along with country of spotting. Audio file should be around 5 to 20 seconds."
)

iface.launch()

