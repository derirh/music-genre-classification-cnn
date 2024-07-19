import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
import librosa
import tensorflow as tf
import pickle
from pydub import AudioSegment
from pydub.utils import which

AudioSegment.ffmpeg = which("ffmpeg")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload'

# Fungsi untuk ekstraksi fitur dari file audio menggunakan PyDub
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    try:
        print(f"Loading audio file from: {file_path}")
        signal, sample_rate = librosa.load(file_path, sr=None)
        print("Audio file loaded successfully")
        
        if chroma:
            stft = np.abs(librosa.stft(signal))
            resultt = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40).T, axis=0)
            resultt = np.hstack((resultt, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            resultt = np.hstack((resultt, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=signal, sr=sample_rate).T, axis=0)
            resultt = np.hstack((resultt, mel))
        return resultt
    
    except Exception as e:
        print(f"Error in extract_features: {e}")
        return None

# Memuat encoder dengan pickle
MODEL_PATH = 'model.h5'
ENCODER_PATH = 'encoder.pkl'

model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

def classify_genre(features):
    features = features.reshape(1, -1)  # Ubah bentuk sesuai dengan input model
    predictions = model.predict(features)
    predicted_label = encoder.inverse_transform([np.argmax(predictions)])[0]
    return f"Genre : {predicted_label}"

@app.route('/', methods=['GET', 'POST'])
def index():
    resultt = ""
    if request.method == 'POST':
        if 'music-file' not in request.files:
            resultt = "Tidak ada file yang diunggah"
        else:
            file = request.files['music-file']
            if file.filename == '':
                resultt = "File tidak dipilih"
            else:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Ensure the file is saved and accessible
                if not os.path.exists(file_path):
                    resultt = "File tidak dapat disimpan"
                else:
                    features = extract_features(file_path)
                    if features is None:
                        resultt = "Ekstraksi fitur gagal"
                    else:
                        resultt = classify_genre(features)
    return render_template('index.html', result=resultt)

if __name__ == '__main__':
    app.run(debug=True)

