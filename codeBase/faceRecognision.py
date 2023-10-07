
import face_recognition
import cv2
import numpy as np
import os
import math

class faceRecognizer:
    def __init__(self) -> None:
        self.face_locations = []

        self.face_encodings = []
        self.face_names = []

        self.known_face_encodings = []
        self.known_face_names = []


    def faceConfidence(self, face_distance, face_match_threshold=0.6):
        rang = (1 - face_match_threshold)
        linear_val = (1 - face_distance) / (rang * 2)

        if face_distance > face_match_threshold:
            return str(round(linear_val * 100, 2)) + "%"
        else:
            value = (linear_val + ((1 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
            return str(round(value, 2)) + "%"
    
    def encode_faces(self):
        for image in os.listdir("CaptionGenerator_and_FaceRecognition/faces"):
            face_image = face_recognition.load_image_file(image)
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image.split(".")[0])
    
    def run_recognition(self, imgPath):
        frame = cv2.imread(imgPath)

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1] # BGR to RGB

        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

        self.face_names = []
        flag = False
        name = "Unknown"
        confidence = "0%"
        for face_encoding in self.face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = "0%"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = self.faceConfidence(face_distances[best_match_index])
                flag = True
                break
        
        if not flag:
            return "Unrecognized Face !", -1
        return name, confidence
    
    def run_recognition(self, imgPath):
        frame = cv2.imread(imgPath)

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1] # BGR to RGB

        self.face_locations = face_recognition.face_locations(rgb_small_frame)
        self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

        self.face_names = []
        flag = False
        name = "Unknown"
        confidence = "0%"
        for face_encoding in self.face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = "0%"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = self.faceConfidence(face_distances[best_match_index])
                flag = True
                break
        
        if not flag:
            return "Unrecognized Face !", -1
        return name, confidence


if __name__ == "__main__":
    recognizer = faceRecognizer()
    print(recognizer.run_recognition("CaptionGenerator_and_FaceRecognition/testImages/me.jpg"))