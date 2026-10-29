import cv2

face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture('datasets/CERN_Higgs boson_EDIT.mp4')

while True:
    ok, img = cap.read()
    if not ok:
        print('Video Ending')
        cap.release()
        cv2.destroyAllWindows()
        break

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255))

    # Display
    cv2.imshow('img', img)

    # Press Key Q to quit and Close window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Release the VideoCapture object
        cap.release()
        # Close all window
        cv2.destroyAllWindows()
        break
