import cv2
import numpy as np

AGE_PROTOTXT = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

age_net = cv2.dnn.readNetFromCaffe(AGE_PROTOTXT, AGE_MODEL)

image = cv2.imread("test.jpg")
(h, w) = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227),
                             (78.4263377603, 87.7689143744, 114.895847746),
                             swapRB=False)
age_net.setInput(blob)
preds = age_net.forward()
age = AGE_LIST[preds[0].argmax()]

text = f"Predicted Age: {age}"
cv2.putText(image, text, (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow("Age Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
Add detect.py for age detection
