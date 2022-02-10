import cv2

width = 800
height = 400

# load the image, resize it, and convert it to grayscale
image = cv2.imread("1.jpg")
image = cv2.resize(image, (width, height))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load the number plate detector
n_plate_detector = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
# detect the number plates in the grayscale image
detections = n_plate_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7)

# loop over the number plate bounding boxes
for (x, y, w, h) in detections:
    # draw a rectangle around the number plate
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(image, "Number plate detected", (x - 20, y - 10),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 2)

    # extract the number plate from the grayscale image
    number_plate = gray[y:y + h, x:x + w]
    cv2.imshow("Number plate", number_plate)

cv2.imshow("Number plate detection", image)
cv2.waitKey(0)
