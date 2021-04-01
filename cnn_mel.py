import savee
import cnn

savee.prepare_if_needed()
train_x,train_y = savee.get_mel_data("raw")
test_x,test_y = savee.get_mel_data("test")
nn = cnn.CNN(train_x, train_y, test_x, test_y)
nn.train()
accuracy = nn.test()
print("Test Accuracy")
print(accuracy)

