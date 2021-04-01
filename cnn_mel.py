import savee
import cnn
import preprocessor

savee.prepare_if_needed()
train_x,train_y = savee.get_mel_data("raw")
train_x = preprocessor.preprocess_mel(train_x)
test_x,test_y = savee.get_mel_data("test")
test_x = preprocessor.preprocess_mel(test_x)
nn = cnn.CNN(train_x, train_y, test_x, test_y)
nn.train()
accuracy = nn.test()
print("Test Accuracy")
print(accuracy)

