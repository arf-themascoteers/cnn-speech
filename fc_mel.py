import preprocessor
import savee
import fc

savee.prepare_if_needed()
train_x,train_y = savee.get_mel_data("train")
test_x,test_y = savee.get_mel_data("test")

train_x = preprocessor.preprocess_mel(train_x)
test_x = preprocessor.preprocess_mel(test_x)

train_x = [i.reshape(-1) for i in train_x]
test_x = [i.reshape(-1) for i in test_x]
nn = fc.FullyConnected(train_x, train_y, test_x, test_y)
nn.train()
accuracy, loss = nn.test()
print("Test Accuracy")
print(accuracy)
print("Test Loss")
print(nn.loss)

