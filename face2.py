import cv2
from cvzone.FaceDetectionModule import FaceDetector

# Initialize the Face Detector
detector = FaceDetector()

# Load the input image
input_image_path = 'images/1000059966.jpg'  # Replace with your input image path
img = cv2.imread(input_image_path)
img=cv2.resize(img,(1080,720))
# Detect faces
img, bboxs = detector.findFaces(img, draw=False)
if not bboxs:
    print("No faces detected.")
else:
    print(f"Faces detected: {bboxs}")
# Draw rectangles around detected faces
if bboxs:
    for bbox in bboxs:
        x, y, w, h = bbox['bbox']
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        score = bbox["score"][0]
        cv2.putText(img, f'SCORE: {int(score * 100)}%', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Display the output image
cv2.imshow("Detected Faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
