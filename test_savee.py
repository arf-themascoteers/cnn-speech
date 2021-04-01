import savee
import librosa
import numpy as np
import matplotlib.pyplot as plt
import preprocessor

X, sample_rate = librosa.load(f"data/dev/KL_d01.wav"
                              , res_type='kaiser_fast'
                              , duration=2.5
                              , sr=44100
                              , offset=0.5
                              )
# X,_ = librosa.effects.trim(X)
#
# female = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
# female = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
# print(female)

dev_x,dev_y = savee.get_mel_data("dev")
dev_x = preprocessor.preprocess_mel(dev_x)

librosa.display.specshow(dev_x[0], y_axis='mel', x_axis='time');
plt.title('Mel Spectrogram');
plt.colorbar(format='%+2.0f dB');
plt.show()