import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)
print("Camera check ho raha hai... (Wait 5 seconds)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera nahi mila!")
        break
    
    cv2.imshow("TESTING CAMERA - Press Q to Close", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()