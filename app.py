import cv2
import mediapipe as mp
import numpy as np
import torch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
from segment_anything import sam_model_registry, SamPredictor

def track_hands(input_path, output_path, sam_checkpoint_path="sam_vit_b_01ec64.pth"):
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

    # Initialize SAM
    model_type = "vit_b"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Open video
    cap = cv2.VideoCapture(input_path)
    success, frame = cap.read()
    if not success:
        raise ValueError("Failed to read video")

    # Process first frame
    first_frame = frame.copy()
    hand_boxes = detect_hands(first_frame, hands)

    if not hand_boxes:
        print("No hands detected in the first frame.")
        cap.release()
        return

    # Generate initial masks with SAM
    predictor.set_image(first_frame)
    input_boxes = torch.tensor(hand_boxes, device=device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, first_frame.shape[:2])
    masks, _, _ = predictor.predict_torch(
        boxes=transformed_boxes,
        point_coords=None,
        point_labels=None,
        multimask_output=False
    )
    prev_masks = masks.cpu().numpy().squeeze(axis=1)

    # Prepare output video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Process and write first frame
    out.write(overlay_masks(first_frame, prev_masks))

    # Process subsequent frames
    while True:
        success, frame = cap.read()
        if not success:
            break

        current_boxes = get_boxes_from_masks(prev_masks, height, width)
        if not current_boxes:
            out.write(frame)
            prev_masks = []
            continue

        # Predict masks for current frame
        predictor.set_image(frame)
        input_boxes = torch.tensor(current_boxes, device=device)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, frame.shape[:2])
        masks, _, _ = predictor.predict_torch(
            boxes=transformed_boxes,
            point_coords=None,
            point_labels=None,
            multimask_output=False
        )
        prev_masks = masks.cpu().numpy().squeeze(axis=1)

        # Overlay and write frame
        out.write(overlay_masks(frame, prev_masks))

    # Release resources
    cap.release()
    out.release()
    hands.close()

def detect_hands(frame, hands):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    hand_boxes = []
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            x_coords = [lm.x * frame.shape[1] for lm in landmarks.landmark]
            y_coords = [lm.y * frame.shape[0] for lm in landmarks.landmark]
            x1, x2 = int(min(x_coords)), int(max(x_coords))
            y1, y2 = int(min(y_coords)), int(max(y_coords))
            hand_boxes.append([x1, y1, x2, y2])
    return hand_boxes

def get_boxes_from_masks(masks, img_height, img_width):
    boxes = []
    for mask in masks:
        y, x = np.where(mask)
        if len(x) == 0 or len(y) == 0:
            continue
        x1, x2 = max(0, np.min(x)), min(img_width, np.max(x))
        y1, y2 = max(0, np.min(y)), min(img_height, np.max(y))
        boxes.append([x1, y1, x2, y2])
    return boxes

def overlay_masks(frame, masks):
    overlay = frame.copy()
    for mask in masks:
        color = np.random.randint(0, 255, 3).tolist()
        overlay[mask] = color
    return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

if __name__ == "__main__":
    track_hands("test.mp4", "output.mp4")