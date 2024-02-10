import os
import matplotlib.pyplot as plt
import numpy as np
#for loading and visualizing audio files
import librosa
import librosa.display

audio_path = "./Generative Music/pianoTriadDataset/audio/"
audio_clips = os.listdir(audio_path)
#print("No. of .wav files in audio folder = ",len(audio_clips))

# Initialize variables to store the minimum and maximum values of Xdb
min_db = np.inf
max_db = -np.inf
# Format colorbar as desired
cmap = 'inferno'

# Iterate through all samples(4320) and get the min and max db for plotting, so every picture is plotted in the same range
for l in range(len(audio_clips)): 
    # load X with the respective path and the sampling rate of 16000hz
    x, sr = librosa.load(audio_path + audio_clips[l], sr=16000)
    # apply the Short-time Fourier transformation
    X = librosa.stft(x)
    # shift the amplitude to db
    Xdb = librosa.amplitude_to_db(abs(X))
    
    # Update min_db and max_db
    min_db = min(min_db, np.min(Xdb))
    max_db = max(max_db, np.max(Xdb))

# Iterate through the range of all samples again to plot the spectrograms
for l in range(10):
    # Again load X with the respective path and the sampling rate of 16000hz
    x, sr = librosa.load(audio_path + audio_clips[l], sr=16000)
    # Again apply the Short-time Fourier transformation
    X = librosa.stft(x)
    # shift the amplitude to db which is nearer to the way humans perceive sound (logarithmically)
    Xdb = librosa.amplitude_to_db(abs(X))
        
    fig = plt.figure(figsize=(14, 5))   
    # Use the same vmin and vmax for all plots
   
    # librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', vmin=min_db, vmax=max_db, cmap=cmap)
   
    # apply the log representation of the frequency
    # librosa.display.specshow(Xdb, sr=sr, y_axis="log", vmin=min_db, vmax=max_db, cmap=cmap)
    # plt.savefig('Generative Music/generatedSpectograms/log_representation/image_'+ audio_clips[l]+'.png'.format())

    librosa.display.specshow(Xdb, sr=sr, vmin=min_db, vmax=max_db, cmap=cmap)
    print('Shape:', Xdb.shape)
    plt.savefig('Generative Music/generatedSpectograms/image_'+ audio_clips[l]+'.png'.format())
    #plt.colorbar()  
    #plt.show()

    