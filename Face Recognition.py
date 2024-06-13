import cv2
import numpy as np
import os
import torch
from mtcnn.mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
from torchvision import datasets
from torch.utils.data import DataLoader
from scipy.spatial.distance import cosine
from sklearn.metrics import average_precision_score

import time

import paho.mqtt.client as mqttClient
import paho.mqtt.publish as publish
from paho import mqtt

broker = 'broker.hivemq.com'
port = 1883 
topic = "FaceDetection"
message = "Face Detected"

workers = 0 if os.name == 'nt' else 4

# Determine if an Nvidia GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# MQTT Establishment
def on_connect(client, data, flags, returnCode):
    if returnCode == 0:
        print("Connected...")
    else:
        print(f"Connection Error: {str(returnCode)}")

def on_message(client, data, message):
    messageText = message.payload.decode("utf-8")
    print(f"Message Topic: {message.topic};\t message text: {messageText}\n")

client = mqttClient.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(broker, port)

last_published_time = 0  # Variable to store the last time a message was published

# Initialize MTCNN for face detection
detector = MTCNN()

# Define collate_fn function
def collate_fn(batch):
    return tuple(zip(*batch))

# Load user embeddings from dataset
user_embeddings = {}
user_names = {} # Store user name
user_ids = {}  # Store user IDs
user_labels = []
dataset = datasets.ImageFolder('Dataset')
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

# Define a dataset and data loader
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=4)

# Initialize InceptionResnetV1 model for computing embeddings
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Function to calculate embeddings for a given face image
def get_embeddings(face):
    # Preprocess face image and calculate embeddings
    face = cv2.resize(face, (160, 160))  # Resize to match FaceNet input size
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert to RGB
    face = (face / 255.).astype(np.float32)  # Normalize pixel values
    face = torch.from_numpy(face.transpose(2, 0, 1)).unsqueeze(0).to(device)  # Convert to tensor
    with torch.no_grad():
        embeddings = resnet(face).cpu().numpy().flatten()  # Calculate embeddings
    return embeddings

# Initialize lists to store predicted and true labels
predicted_labels = []
true_labels = []
detected_users = set()  # Set to store unique detected users
tp, fp, fn = 0, 0, 0  # Initialize TP, FP, FN

# Load user embeddings from dataset
for filename, label in dataset.samples:
    if filename.endswith('.jpg'):
        parts = filename.split('_')
        user_id = int(parts[1])  # Extracting user_id from the filename
        user_name = parts[2]  # Extracting user_name from the filename
        user_img = cv2.imread(filename)  # Use the filename directly
        user_embeddings[user_name] = get_embeddings(user_img)  # Store embeddings
        user_names[user_name] = user_name
        user_ids[user_name] = user_id  # Store user_id with the user_name
        user_labels.append(user_name)  # Collect ground truth labels

# Initialize camera
cam = cv2.VideoCapture(0)
cam.set(3, 720)  # Set video capture resolution width
cam.set(4, 720)  # Set video capture resolution height

print("\n [INFO] Initializing face recognition. Look at the camera and wait ...")

# Initialize FPS variables
start_time = time.time()
frames_processed = 0

while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to capture image")
        break

    # Detect faces using MTCNN
    start_detection_time = time.time()
    faces = detector.detect_faces(img)

    detected_labels = []  # To track labels of detected faces in the current frame

    # Assuming `faces` contains bounding boxes of detected faces
    for result in faces:
        x, y, w, h = result['box']
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw rectangle around detected face
        face = img[y:y+h, x:x+w]

        # Calculate embeddings for the detected face
        query_embeddings = get_embeddings(face)

        # Perform face recognition
        best_match_name = None
        min_distance = float('inf')
        for user_name, user_embedding in user_embeddings.items():
            distance = cosine(query_embeddings, user_embedding)
            if distance < min_distance:
                min_distance = distance
                best_match_name = user_name

        # Output recognition result
        if min_distance < 0.25:  # Adjust threshold as needed

            end_detection_time = time.time()
            detection_time = end_detection_time - start_detection_time

            confidence = (1 - min_distance) * 100
            user_name = user_names.get(best_match_name, "Unknown")
            user_id = user_ids.get(best_match_name, "Unknown")

            print(f"User Name: {user_name}, User ID: {user_id}")
            print(f"Confidence: {confidence:.2f}")
            cv2.putText(img, f"Name: {user_name}", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"Confidence: {confidence:.2f}%", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"Detection Time: {detection_time:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

            # Check if 10 seconds have passed since the last message was published
            current_time = time.time()
            if current_time - last_published_time >= 10:
                print(message)
                client.publish(topic, message)
                last_published_time = current_time

            # Update TP, FP
            if user_name in user_labels:  # Check if recognized name is in the ground truth labels
                tp += 1
            else:
                fp += 1
            detected_labels.append(user_name)  # Track detected label

            # Store detected user name and ID in the set
            detected_users.add((user_name, user_id))
        else:
            cv2.putText(img, "Unknown", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            fn += 1
            detected_labels.append("Unknown")  # Track detected label

        # Store predicted and true labels
        predicted_labels.append(best_match_name)
        true_labels.append(user_labels[0])  # Assuming the first label corresponds to the true identity

    # Calculate FPS
    frames_processed += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:  # Update FPS every 1 second
        fps = frames_processed / elapsed_time
        frames_processed = 0
        start_time = time.time()

    # Draw FPS on the frame
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Face Recognition', img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Check if the "Esc" key is pressed
        break

# Release camera and close windows
cam.release()
cv2.destroyAllWindows()

# Calculate performance metrics
precision = tp / (tp + fp) if tp + fp > 0 else 0
recall = tp / (tp + fn) if tp + fn > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
average_precision = average_precision_score([1 if label == user_labels[0] else 0 for label in true_labels],
                                            [1 if label == user_labels[0] else 0 for label in predicted_labels])

# Print performance metrics
print("\nProgram Summary:")

# Print detected user names and IDs
print("\nDetected Users:")
for user_name, user_id in detected_users:
    print(f"User Name: {user_name}, User ID: {user_id}")
    
print("\nPerformance Metrics Summary:")
print('Running on device: {}'.format(device))
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Mean Average Precision (mAP): {average_precision:.4f}")



print("\n [INFO] Exiting Program and cleanup stuff")
