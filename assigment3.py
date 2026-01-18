import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.circle(
                roi_color,
                (ex + ew // 2, ey + eh // 2),
                ew // 2,
                (255, 0, 0),
                2
            )

        mouth_y = int(h * 0.6)
        mouth_h = int(h * 0.25)
        mouth_x = int(w * 0.2)
        mouth_w = int(w * 0.6)

        cv2.rectangle(
            roi_color,
            (mouth_x, mouth_y),
            (mouth_x + mouth_w, mouth_y + mouth_h),
            (0, 0, 255),
            2
        )

    cv2.imshow("Eyes and Mouth", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
