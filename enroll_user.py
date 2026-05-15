import cv2
import numpy as np
import yaml
import argparse
from pathlib import Path
import time

from utils.face_recognizer import FaceRecognizer, UserDatabase


class UserEnrollment:    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.face_recognizer = FaceRecognizer(
            detector_path=self.config['models']['face_detector'],
            recognizer_path=self.config['models']['face_recognizer'],
            det_threshold=self.config['detection']['face_confidence_threshold']
        )
        
        self.user_db = UserDatabase(
            db_path=self.config['database']['face_encodings_path']
        )
        
        print("✓ Enrollment tool initialized")
    
    def enroll_from_camera(self, user_id: str, name: str, num_samples: int = 5):
        print(f"\nEnrolling user: {name} (ID: {user_id})")
        print(f"Capturing {num_samples} face samples...")
        print("Position your face in front of the camera and press SPACE to capture")
        
        cap = cv2.VideoCapture(0)
        embeddings = []
        captured = 0
        
        while captured < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            faces = self.face_recognizer.detect_faces(frame)
            
            display = frame.copy()
            for face_box in faces:
                x1, y1, x2, y2 = face_box
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.putText(display, f"Captured: {captured}/{num_samples}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, "Press SPACE to capture, Q to quit", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            if len(faces) > 0:
                cv2.putText(display, "Face detected - Ready to capture", (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display, "No face detected", (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow('User Enrollment', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and len(faces) > 0:
                face_box = faces[0]
                embedding = self.face_recognizer.get_face_embedding(frame, face_box)
                
                if embedding is not None:
                    embeddings.append(embedding)
                    captured += 1
                    print(f"✓ Captured sample {captured}/{num_samples}")
                    time.sleep(0.5)
                else:
                    print("✗ Failed to extract face embedding")
            
            elif key == ord('q'):
                print("Enrollment cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(embeddings) >= num_samples:
            self.user_db.add_user(user_id, name, embeddings)
            print(f"✓ User '{name}' enrolled successfully with {len(embeddings)} samples")
        else:
            print(f"✗ Enrollment failed - insufficient samples")
    
    def enroll_from_images(self, user_id: str, name: str, image_dir: str):
        print(f"\nEnrolling user: {name} (ID: {user_id})")
        print(f"Loading images from: {image_dir}")
        
        image_path = Path(image_dir)
        if not image_path.exists():
            print(f"✗ Directory not found: {image_dir}")
            return
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(image_path.glob(ext))
            image_files.extend(image_path.glob(ext.upper()))
        
        if len(image_files) == 0:
            print("✗ No images found in directory")
            return
        
        print(f"Found {len(image_files)} images")
        
        embeddings = []
        
        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None:
                print(f"✗ Failed to load: {img_file.name}")
                continue
            
            faces = self.face_recognizer.detect_faces(image)
            
            if len(faces) == 0:
                print(f"⚠ No face detected in: {img_file.name}")
                continue
            
            if len(faces) > 1:
                print(f"⚠ Multiple faces in: {img_file.name} - using first face")
            
            face_box = faces[0]
            embedding = self.face_recognizer.get_face_embedding(image, face_box)
            
            if embedding is not None:
                embeddings.append(embedding)
                print(f"✓ Processed: {img_file.name}")
            else:
                print(f"✗ Failed to extract embedding: {img_file.name}")
        
        if len(embeddings) > 0:
            self.user_db.add_user(user_id, name, embeddings)
            print(f"✓ User '{name}' enrolled successfully with {len(embeddings)} samples")
        else:
            print("✗ Enrollment failed - no valid face embeddings extracted")
    
    def list_users(self):
        users = self.user_db.get_all_users()
        
        if len(users) == 0:
            print("\nNo users enrolled")
        else:
            print(f"\nEnrolled Users ({len(users)}):")
            print("-" * 40)
            for user_id, name in users.items():
                print(f"  ID: {user_id:20} Name: {name}")
    
    def remove_user(self, user_id: str):
        if self.user_db.remove_user(user_id):
            print(f"✓ User removed: {user_id}")
        else:
            print(f"✗ User not found: {user_id}")


def main():
    parser = argparse.ArgumentParser(description="User Enrollment Tool")
    parser.add_argument('--mode', choices=['camera', 'images', 'list', 'remove'], 
                       required=True, help='Enrollment mode')
    parser.add_argument('--user-id', help='Unique user ID')
    parser.add_argument('--name', help='User display name')
    parser.add_argument('--image-dir', help='Directory containing user images')
    parser.add_argument('--samples', type=int, default=5, 
                       help='Number of samples to capture (camera mode)')
    
    args = parser.parse_args()
    
    enrollment = UserEnrollment()
    
    if args.mode == 'camera':
        if not args.user_id or not args.name:
            print("Error: --user-id and --name required for camera mode")
            return
        enrollment.enroll_from_camera(args.user_id, args.name, args.samples)
    
    elif args.mode == 'images':
        if not args.user_id or not args.name or not args.image_dir:
            print("Error: --user-id, --name, and --image-dir required for images mode")
            return
        enrollment.enroll_from_images(args.user_id, args.name, args.image_dir)
    
    elif args.mode == 'list':
        enrollment.list_users()
    
    elif args.mode == 'remove':
        if not args.user_id:
            print("Error: --user-id required for remove mode")
            return
        enrollment.remove_user(args.user_id)


if __name__ == "__main__":
    main()
