import cv2
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

data_path = 'dataset'
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]
label_dict = dict(zip(categories, labels))
print(label_dict)

img_size = 100
data = []
target = []

for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (img_size, img_size))
            data.append(resized)
            target.append(label_dict[category])
        except Exception as e:
            print('Exception: ', e)

data = np.array(data) / 255
data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
target = np.array(target)

new_target = to_categorical(target)
print(new_target)

model = Sequential()

model.add(Conv2D(200, (3, 3), input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(100, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# train_data, test_data, train_target, test_target = train_test_split(data, new_target, test_size=0.1)
#
# history = model.fit(train_data, train_target, validation_split=0.2, epochs=20, batch_size=32)
#
# plt.plot(history.history['loss'], 'r', label='training loss')
# plt.plot(history.history['val_loss'], label='validation loss')
# plt.xlabel('# epochs')
# plt.ylabel('loss')
# plt.legend()
# plt.show()
#
# fig = plt.figure(figsize=(7, 4))
# plt.plot(history.history['accuracy'], 'r', label='training accuracy')
# plt.plot(history.history['val_accuracy'], label='validation accuracy')
# plt.xlabel('# epochs')
# plt.ylabel('loss')
# plt.legend()
# plt.show()
#
# print(model.evaluate(test_data, test_target))
#
# model.save('model.model')
model = load_model('model.model')

face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

source = cv2.VideoCapture(0)
labels_dict = {0: 'MASK', 1: 'NO MASK'}
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}
#
while True:

    ret, img = source.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_clsfr.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        resized = cv2.resize(face_img, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        result = model.predict(reshaped)
        print(result)
        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(img, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('LIVE', img)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cv2.destroyAllWindows()
source.release()




