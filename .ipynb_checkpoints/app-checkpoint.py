import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("mnist_model.h5", compile=False)

cap = cv2.VideoCapture(0)

print("Press SPACE to capture digit")
print("Press Q to quit")

while True:

    ret, frame = cap.read()
    display = frame.copy()

    cv2.imshow("Webcam", display)

    key = cv2.waitKey(1)

    if key == 32:   # SPACE key

        img = frame.copy()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # Threshold to isolate digit
        _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:

            # Largest contour assumed to be digit
            c = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(c)

            digit = thresh[y:y+h, x:x+w]

            # Add padding
            digit = cv2.copyMakeBorder(digit,20,20,20,20,cv2.BORDER_CONSTANT,value=0)

            # Resize to MNIST format
            digit = cv2.resize(digit,(28,28))

            cv2.imshow("Processed Digit", digit)

            # Normalize
            digit = digit / 255.0

            # reshape for CNN
            digit = digit.reshape(1,28,28,1)

            prediction = model.predict(digit)

            digit_pred = np.argmax(prediction)

            print("Predicted Digit:", digit_pred)

        else:
            print("No digit detected")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()