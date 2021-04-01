import savee
import preprocessor

savee.prepare_if_needed()
X,Y = savee.get_data("dev")
X = preprocessor.preprocess(X)
print(X)