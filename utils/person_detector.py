import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple

class PersonDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5, img_size: int = 640):
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"✓ Person detector loaded: {model_path}")
        print(f"  Input: {self.input_name}, Outputs: {self.output_names}")
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w = image.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        pad_h = (self.img_size - new_h) // 2
        pad_w = (self.img_size - new_w) // 2
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        padded = padded.astype(np.float32) / 255.0
        padded = np.transpose(padded, (2, 0, 1))
        padded = np.expand_dims(padded, axis=0)
        
        return padded, scale, (pad_w, pad_h)
    
    def postprocess(self, outputs: np.ndarray, scale: float, pad: Tuple[int, int], 
                   orig_shape: Tuple[int, int]) -> List[dict]:        
        predictions = np.squeeze(outputs)
        if len(predictions.shape) == 2:
            if predictions.shape[0] < predictions.shape[1]:
                predictions = predictions.transpose()
        else:
            print("unexpected output shape:",predictions.shape)
            return []
        
        detections = []
        pad_w, pad_h = pad
        
        for pred in predictions:
            x_center, y_center, width, height = pred[:4]
            class_scores = pred[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            if class_id == 0 and confidence >= self.conf_threshold:
                x1 = (x_center - width / 2 - pad_w) / scale
                y1 = (y_center - height / 2 - pad_h) / scale
                x2 = (x_center + width / 2 - pad_w) / scale
                y2 = (y_center + height / 2 - pad_h) / scale
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(orig_shape[1], int(x2))
                y2 = min(orig_shape[0], int(y2))
                
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'confidence': float(confidence)
                })
        
        detections = self.non_max_suppression(detections, iou_threshold=0.45)
        
        return detections
    
    def non_max_suppression(self, detections: List[dict], iou_threshold: float = 0.45) -> List[dict]:
        if len(detections) == 0:
            return []
        
        boxes = np.array([d['box'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]
    
    def detect(self, image: np.ndarray) -> List[dict]:
        input_tensor, scale, pad = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        detections = self.postprocess(outputs[0], scale, pad, image.shape[:2])        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        img_copy = image.copy()        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            conf = det['confidence']            
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_copy, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
            cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return img_copy
