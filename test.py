import savee
import preprocessor

savee.prepare_if_needed()
X,Y = savee.get_mfcc_data("dev")
X = preprocessor.preprocess_mfcc(X)
print(X)