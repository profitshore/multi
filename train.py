from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

import numpy as np

# 카테고리 지정하기
categories = ["human","nonhuman"]
nb_classes = len(categories)

# 이미지 크기 지정하기
image_w = 224
image_h = 224

# 데이터 열기 --- (※1)
X_train, X_test, y_train, y_test = np.load("./dataset2/2obj.npy")

# 데이터 정규화하기
X_train = X_train.astype("float") / 256
X_test  = X_test.astype("float")  / 256
print('X_train shape:', X_train.shape)

# 모델 구축하기 --- (※2)
model = Sequential()

# first set of CONV => RELU => POOL
model.add(Convolution2D(20, 5, 5, border_mode="same",
input_shape=X_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# second set of CONV => RELU => POOL
model.add(Convolution2D(50, 5, 5, border_mode="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# set of FC => RELU layers
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

# softmax classifier
model.add(Dense(nb_classes))
model.add(Activation("softmax"))

model.compile(loss='binary_crossentropy',
    optimizer='sgd',
    metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, nb_epoch=1)

score = model.evaluate(X_test, y_test)
print('loss=', score[0])
print('accuracy=', score[1])

model.save('model2.h5')