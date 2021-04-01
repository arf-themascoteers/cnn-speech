import savee
import librosa
import numpy as np
import matplotlib.pyplot as plt

X, sample_rate = librosa.load(f"data/dev/KL_d01.wav"
                              , res_type='kaiser_fast'
                              , duration=2.5
                              , sr=44100
                              , offset=0.5
                              )
X,_ = librosa.effects.trim(X)

female = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
female = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
print(female)

mel_spect = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=2048, hop_length=1024)
mel_spect = librosa.amplitude_to_db(mel_spect, ref=np.max)

librosa.display.specshow(mel_spect, y_axis='mel', x_axis='time');
plt.title('Mel Spectrogram');
plt.colorbar(format='%+2.0f dB');
plt.show()