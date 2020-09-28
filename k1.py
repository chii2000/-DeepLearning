from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() #xが画像データ、yが正解ラベル0－9

#グラフを作成
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(9, 15))
#サブプロットを作成
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=0.05)
for i in range(9):
    ax = fig.add_subplot(1, 9, i + 1, xticks=[], yticks=[])
    ax.set_title(str(y_train[i]))
    ax.imshow(x_train[i], cmap='gray')

#画像前処理
from keras.utils import to_categorical
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

#one-hot表現に
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#モデル構築
from keras.models import Sequential
from keras.layers import Dense, Activation
model = Sequential()
model.add(Dense(units=256, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(units=100))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, batch_size=1000, epochs=10, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('test_loss:', score[0])
print('test_accuracy:', score[1])
















        