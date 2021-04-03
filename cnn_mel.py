import savee
import cnn
import preprocessor

savee.prepare_if_needed()

train_x,train_y = savee.get_mel_data("train")
test_x,test_y = savee.get_mel_data("test")

train_x = preprocessor.preprocess_mel(train_x)
test_x = preprocessor.preprocess_mel(test_x)

nn = cnn.CNN(train_x, train_y, test_x, test_y)
nn.train()
accuracy, loss = nn.test()
print("Test Accuracy")
print(accuracy)
print("Test Loss")
print(nn.loss)

