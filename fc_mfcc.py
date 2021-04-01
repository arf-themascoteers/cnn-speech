import savee
import preprocessor
import fc
import os

savee.prepare_if_needed()
train_x,train_y = savee.get_mfcc_data("train")
test_x,test_y = savee.get_mfcc_data("test")
train_x = preprocessor.preprocess_mfcc(train_x)
test_x = preprocessor.preprocess_mfcc(test_x)

print(train_x.shape)

nn = fc.FullyConnected(train_x, train_y, test_x, test_y)
nn.train()
accuracy = nn.test()
print("Test Accuracy")
print(accuracy)

