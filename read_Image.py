from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten
import numpy as np
from PIL import Image
import cv2imageload

i = Image.open('/media/cecilia/marina-hd/Cecilia/Downloads/aid584101-v4-728px-Clap-With-One-Hand-Step-1-Version-4.jpg')
#print(i.size)
#i.show()
#imgSize = i.size
#rawData = i.tobytes()
#print(rawData)
image_black_white = i.convert('L')
image_black_white.show()
#i.save('result.png')



def neural_net_model():
    model = Sequential()
    model.add(Conv2D(12, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (4, 4), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model