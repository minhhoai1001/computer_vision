import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hand = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

ptime = 0

while True:
    succes, img = cap.read()

    # Flip the image horizontally for a later selfie-view display, and convert
    img = cv2.flip(img, 1)
    # Convert image from BGR to RGB
    imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hand.process(imgRBG)

    #print("result: ", results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for hand_mp in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_mp.landmark):
                #print('ID: ', id)
                h, w, _ = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(id, cx, cy)
                if id == 0: # 0Wrist
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                elif id in [4, 8, 12, 16, 20]:
                    cv2.circle(img, (cx, cy), 8, (255, 0, 0), cv2.FILLED)

            mp_drawing.draw_landmarks(img, hand_mp, mp_hands.HAND_CONNECTIONS)

    ctime = time.time()
    fps = round(1/(ctime-ptime), 2)
    ptime = ctime

    cv2.putText(img, 'FPS: {}'.format(str(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow('Imgage:', img)

    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()