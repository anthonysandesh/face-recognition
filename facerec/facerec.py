import cv2
import dlib
from imutils import face_utils
from ultralytics import YOLO

def detect_real_faces(image, detector, predictor, yolo_model, confidence_threshold=0.4):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    real_faces = []
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Get the bounding box coordinates of the face
        (x, y, w, h) = face_utils.rect_to_bb(face)

        # Extract the face region from the image
        face_image = image[y:y+h, x:x+w]

        results = yolo_model.predict(face_image, conf=confidence_threshold)

        # If YOLO detects a face and there are facial landmarks present, consider it a real face
        if len(results) > 0 and shape is not None:
            real_faces.append(face)

    return real_faces

model = YOLO("best.pt")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Open the video capture device (0 for webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    real_faces = detect_real_faces(frame, detector, predictor, model)

    for face in real_faces:
        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Real Faces Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
