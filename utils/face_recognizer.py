import cv2
import numpy as np
from insightface.app import FaceAnalysis
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

class FaceRecognizer:

    def __init__(self,
                 detector_path=None,
                 recognizer_path=None,
                 det_threshold=0.5,
                 rec_threshold=0.6):

        self.det_threshold = det_threshold
        self.rec_threshold = rec_threshold

        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

        print("✓ InsightFace initialized")

    def detect_faces(self, image):

        faces = self.app.get(image)

        boxes = []

        for face in faces:
            bbox = face.bbox.astype(int)

            x1, y1, x2, y2 = bbox

            boxes.append((x1, y1, x2, y2))

        return boxes

    def get_all_face_embeddings(self, image):
        """Get all face boxes and embeddings from image in one pass"""
        faces = self.app.get(image)
        
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            embedding = face.embedding / np.linalg.norm(face.embedding)
            results.append({
                'box': (x1, y1, x2, y2),
                'embedding': embedding
            })
        
        return results

    def get_face_embedding(self, image, face_box=None):

        faces = self.app.get(image)

        if len(faces) == 0:
            return None

        # If face_box provided, match it to the correct detected face
        if face_box is not None:
            x1, y1, x2, y2 = face_box
            face_cx = (x1 + x2) / 2
            face_cy = (y1 + y2) / 2
            
            best_match = 0
            best_distance = float('inf')
            
            for idx, face in enumerate(faces):
                bbox = face.bbox.astype(int)
                det_x1, det_y1, det_x2, det_y2 = bbox
                det_cx = (det_x1 + det_x2) / 2
                det_cy = (det_y1 + det_y2) / 2
                
                distance = np.sqrt((face_cx - det_cx)**2 + (face_cy - det_cy)**2)
                if distance < best_distance:
                    best_distance = distance
                    best_match = idx
            
            embedding = faces[best_match].embedding
        else:
            # If no face_box provided, use first face
            embedding = faces[0].embedding

        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def compare_embeddings(self, embedding1, embedding2):

        similarity = np.dot(embedding1, embedding2)

        return float(similarity)

    def draw_faces(self, image, faces, labels=None):

        img_copy = image.copy()

        for i, (x1, y1, x2, y2) in enumerate(faces):

            color = (0, 255, 0)

            if labels and labels[i] == "Unknown":
                color = (0, 0, 255)

            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

            if labels and i < len(labels):

                label = labels[i]

                cv2.putText(
                    img_copy,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        return img_copy


class UserDatabase:    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.users = {}
        self.db_path.parent.mkdir(parents=True, exist_ok=True)        
        self.load()
    
    def load(self):
        if self.db_path.exists():
            try:
                with open(self.db_path, 'rb') as f:
                    self.users = pickle.load(f)
                print(f"✓ Loaded {len(self.users)} enrolled users")
            except Exception as e:
                print(f"⚠ Error loading database: {e}")
                self.users = {}
        else:
            print("ℹ No existing user database found")
    
    def save(self):
        try:
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.users, f)
            print(f"✓ Saved {len(self.users)} users to database")
        except Exception as e:
            print(f"✗ Error saving database: {e}")
    
    def add_user(self, user_id: str, name: str, embeddings: List[np.ndarray]):
        self.users[user_id] = {
            'name': name,
            'embeddings': embeddings
        }
        self.save()
        print(f"✓ User '{name}' (ID: {user_id}) added with {len(embeddings)} face samples")
    
    def remove_user(self, user_id: str):
        if user_id in self.users:
            name = self.users[user_id]['name']
            del self.users[user_id]
            self.save()
            print(f"✓ User '{name}' removed")
            return True
        return False
    
    def identify_user(self, embedding: np.ndarray, threshold: float = 0.6) -> Optional[Tuple[str, str, float]]:
        best_match = None
        best_similarity = threshold
        
        for user_id, user_data in self.users.items():
            for user_embedding in user_data['embeddings']:
                similarity = np.dot(embedding, user_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (user_id, user_data['name'], similarity)
        
        # Debug: Show similarity scores
        if not best_match:
            scores = []
            for user_id, user_data in self.users.items():
                max_sim = max([np.dot(embedding, e) for e in user_data['embeddings']])
                scores.append(f"{user_data['name']}: {max_sim:.4f}")
            if scores:
                print(f"  [DEBUG] No match (threshold={threshold:.2f}): {', '.join(scores)}")
        
        return best_match
    
    def get_all_users(self) -> dict:
        return {uid: data['name'] for uid, data in self.users.items()}
