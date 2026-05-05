import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

label = input("Enter gesture label (fist/palm/thumbs): ")

with open('gesture_data.csv', 'a', newline='') as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                data = []

                for lm in hand_landmarks.landmark:
                    data.append(lm.x)
                    data.append(lm.y)

                data.append(label)
                writer.writerow(data)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Label: {label}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Collecting Data", frame)

        if cv2.waitKey(1) == 27:  # ESC to stop
            break

cap.release()
cv2.destroyAllWindows()