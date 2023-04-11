import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def detect_l(hand_landmarks):
    thumb_extended = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP].x
    index_finger_extended = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP].y
    middle_finger_folded = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_finger_folded = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_PIP].y
    pinky_folded = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_PIP].y
    thumb_below_index = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y > hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP].y

    if thumb_extended and index_finger_extended and middle_finger_folded and ring_finger_folded and pinky_folded and thumb_below_index:
        return True
    return False



mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3, max_num_hands=1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if detect_l(hand_landmarks):
                    index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]

                    index_finger_tip_x, index_finger_tip_y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
                    thumb_tip_x, thumb_tip_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])

                    cv2.rectangle(image, (index_finger_tip_x - 10, index_finger_tip_y - 10), (index_finger_tip_x + 10, index_finger_tip_y + 10), (0, 255, 0), 2)
                    cv2.rectangle(image, (thumb_tip_x - 10, thumb_tip_y - 10), (thumb_tip_x + 10, thumb_tip_y + 10), (0, 255, 0), 2)
                    cv2.putText(image, 'VOCE FEZ O L', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Detector De L', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
