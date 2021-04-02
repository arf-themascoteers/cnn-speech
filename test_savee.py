import savee
import librosa
import matplotlib.pyplot as plt
import preprocessor



dev_x,dev_y = savee.get_mel_data("dev")
dev_x = preprocessor.preprocess_mel(dev_x)
print(dev_x[0].shape)
librosa.display.specshow(dev_x[0], y_axis='mel', x_axis='time');
plt.title('Mel Spectrogram');
plt.colorbar(format='%+2.0f dB');
plt.show()