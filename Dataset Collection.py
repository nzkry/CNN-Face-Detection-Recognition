import cv2
import numpy as np
import torch
import os
from pathlib import Path
from torchvision import transforms
from mtcnn.mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1

# Initialize MTCNN and FaceNet
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

#Define MTCNN module
mtcnn = MTCNN()

#Define Inception Resnet V1 Module
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Function to classify face using FaceNet
def classify_face(face, min_size=160):  # Added minimum size for input validation
    """Preprocesses a face image and returns its embeddings using FaceNet.
    Performs basic size check to avoid potential errors.
    """

    if face.shape[0] < min_size or face.shape[1] < min_size:
        print(f"Error: Face image too small. Minimum size is {min_size}x{min_size}.")
        return None

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_tensor = transforms.ToTensor()(face).unsqueeze(0).to(device)

    # Get face embeddings
    with torch.no_grad():
        embeddings = resnet(face_tensor)
    return embeddings

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video capture resolution width
cam.set(4, 480)  # Set video capture resolution height

user_id = input('\n Enter user ID: ')
user_name = input('\n Enter user name: ')

print("\n [INFO] Initializing face capture. Look at the camera and wait ...")

count = 0

while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to capture image")
        break

    # Detect faces using MTCNN (check for updated method name if necessary)
    faces = mtcnn.detect_faces(img)

    if faces is not None:
        for result in faces:
            x, y, w, h = result['box']
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw rectangle around detected face
            face = img[y:y+h, x:x+w]

            # Minimum size check before processing
            if face.shape[0] < 160 or face.shape[1] < 160:
                print(f"Warning: Skipping small face (size: {face.shape[0]}x{face.shape[1]}).")
                continue

            # Create Folder Directory 
            folder_path = Path(f'C:/Users/nurzi/OneDrive/Documents/Python/CNN Face Detection & Recognition/Dataset/{user_name}')
            folder_path.mkdir(parents=True, exist_ok=True)

            #Save the detected face with user ID and name
            status = cv2.imwrite(f'C:/Users/nurzi/OneDrive/Documents/Python/CNN Face Detection & Recognition/Dataset/{user_name}/User_{user_id}_{user_name}_{count}.jpg', face)
            # Classify the face using FaceNet
            embeddings = classify_face(face)

            # Print face embeddings (for debugging or further processing)
            if embeddings is not None:
                print(embeddings)

            count += 1

    cv2.imshow('Dataset Collection', img)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
        break

print("\n [INFO] Exiting Program and cleanup stuff")
print("\n Image written to file-system : ", status)

cam.release()
cv2.destroyAllWindows()