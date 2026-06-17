import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# ---------------- LOAD MODELS ----------------
yolo_model = YOLO("yolov8n.pt")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# ---------------- LOAD ASSETS ----------------
necklace = cv2.imread("assets/necklace.png", cv2.IMREAD_UNCHANGED)
ring = cv2.imread("assets/ring.png", cv2.IMREAD_UNCHANGED)
earring = cv2.imread("assets/earring.png", cv2.IMREAD_UNCHANGED)

if necklace is None or ring is None or earring is None:
    print("❌ Check asset paths")
    exit()

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

# ---------------- WINDOW ----------------
cv2.namedWindow("AI Jewellery Try-On", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AI Jewellery Try-On", 900, 700)

# ---------------- OVERLAY FUNCTION ----------------
def overlay_image(frame, overlay, x, y):
    h, w, _ = overlay.shape

    for i in range(h):
        for j in range(w):
            if overlay[i][j][3] > 0:
                if 0 <= y+i < frame.shape[0] and 0 <= x+j < frame.shape[1]:
                    frame[y+i][x+j] = overlay[i][j][:3]

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ----------- YOLO: Detect Person -----------
    results = yolo_model(frame, verbose=False)

    person_boxes = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_boxes.append((x1, y1, x2, y2))

    neck_detected = False

    # ----------- PROCESS EACH PERSON -----------
    for (x1, y1, x2, y2) in person_boxes:

        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            continue

        rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)

        # ----------- FACE DETECTION -----------
        face_results = face_mesh.process(rgb_roi)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:

                h, w, _ = person_roi.shape

                left = face_landmarks.landmark[234]
                right = face_landmarks.landmark[454]
                chin = face_landmarks.landmark[152]

                lx = int(left.x * w) + x1
                rx = int(right.x * w) + x1
                ly = int(left.y * h) + y1

                cx = int(chin.x * w) + x1
                cy = int(chin.y * h) + y1

                neck_width = rx - lx

                if neck_width < 60 or neck_width > 300:
                    continue

                neck_detected = True

                # -------- NECKLACE --------
                h0, w0 = necklace.shape[:2]
                scale = (neck_width * 1.2) / w0

                new_w = int(w0 * scale)
                new_h = int(h0 * scale)

                resized_necklace = cv2.resize(necklace, (new_w, new_h))

                x_offset = cx - new_w // 2
                y_offset = cy + 5

                overlay_image(frame, resized_necklace, x_offset, y_offset)

                # -------- EARRINGS (FIXED) --------
                ear_size = int(neck_width * 0.18)
                ear_size = max(18, ear_size)

                resized_ear = cv2.resize(earring, (ear_size, ear_size))

                # Face edges
                lx_face = lx
                rx_face = rx
                ly_face = ly

                # Move outside face
                horizontal_offset = int(neck_width * 0.07)

                left_ear_x = lx_face - horizontal_offset
                right_ear_x = rx_face + horizontal_offset

                # Better vertical alignment
                ear_y = int(ly_face + (cy - ly_face) * 0.4)

                overlay_image(frame,
                              resized_ear,
                              left_ear_x - ear_size // 2,
                              ear_y - ear_size // 2)

                overlay_image(frame,
                              resized_ear,
                              right_ear_x - ear_size // 2,
                              ear_y - ear_size // 2)

        # ----------- HAND DETECTION -----------
        hand_results = hands.process(rgb_roi)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:

                h, w, _ = person_roi.shape

                p1 = hand_landmarks.landmark[13]
                p2 = hand_landmarks.landmark[14]

                x1_h, y1_h = int(p1.x * w) + x1, int(p1.y * h) + y1
                x2_h, y2_h = int(p2.x * w) + x1, int(p2.y * h) + y1

                cx_h = (x1_h + x2_h) // 2
                cy_h = (y1_h + y2_h) // 2

                dist = int(((x2_h - x1_h)**2 + (y2_h - y1_h)**2) ** 0.5)

                size = int(dist * 0.6)
                size = max(15, size)

                resized_ring = cv2.resize(ring, (size, size))

                overlay_image(frame,
                              resized_ring,
                              cx_h - size // 2,
                              cy_h - size // 2 + 2)

    # ----------- UI ----------------
    if not neck_detected:
        h, w, _ = frame.shape
        cv2.putText(frame, "Align your face",
                    (w//2 - 100, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)

    cv2.putText(frame, "Press Q or ESC to Exit",
                (20, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)

    display_frame = cv2.resize(frame, (900, 700))
    cv2.imshow("Virtual Jewellery Try-On", display_frame)

    key = cv2.waitKey(10) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()