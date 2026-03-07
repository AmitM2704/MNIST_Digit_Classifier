import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("mnist_model.h5")

img = cv2.imread("digit.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(28,28))
img = img/255.0
img = img.reshape(1,28,28)

prediction = model.predict(img)

digit = np.argmax(prediction)

print("Predicted digit:",digit)
