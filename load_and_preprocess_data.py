import os
import matplotlib.pyplot as plt
import numpy as np
#for loading and visualizing audio files
import librosa
import librosa.display
import tensorflow as tf

audio_path = "./Generative Music/data/pianoTriadDataset/audio/"
audio_clips = os.listdir(audio_path)
#print("No. of .wav files in audio folder = ",len(audio_clips))

def load_and_preprocess_data(audio_path, audio_clips, target_shape=(128, 128)):
    data = []

    for i in range(len(audio_clips)):
        audio_data, sr = librosa.load(audio_path + audio_clips[i], sr=16000) 
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr)
        mel_spectrogram = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
        
        print('Data:', data[i])

    np.save("./Generative Music/data/preprocessed_data.npy", np.array(data))
    print("Preprocessed data saved successfully.")

    return np.array(data)

if __name__ == "__main__":
    load_and_preprocess_data(audio_path=audio_path, audio_clips=audio_clips)
  
        



    