import cv2
import numpy as np
import mediapipe as mp
import vizdoom as vzd
import os
import threading
import time

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "pose_landmarker_lite.task"
HAND_MODEL_PATH = "hand_landmarker.task"

# Thresholds p/ precisão
EXTENSION_THRESHOLD = 0.25
CAM_DEAD_ZONE = 0.14
PIP_MCP_THRESOLD = 0.06
POSE_VIZ_THRESHOLD = 0.5

# Marcação dos pontos dedo indicador e ombro e pulso direito
INDEX_PIP = 6
INDEX_MCP = 5
RIGHT_SHOULDER = 12
RIGHT_WRIST = 16

# VizDoom actions
MOVE_FORWARD  = [True,  False, False, False]
MOVE_BACKWARD = [False, True,  False, False]
TURN_LEFT     = [False, False, True,  False]
TURN_RIGHT    = [False, False, False, True]
NO_OP         = [False, False, False, False]

# Threading
_latest_result = None
_result_lock = threading.Lock()

_latest_hand_result = None
_hand_result_lock = threading.Lock()
_returned_to_point = True

def _on_result(result, output_image, timestamp_ms: int):
    global _latest_result
    with _result_lock:
        _latest_result = result

def _on_hand_result(result, output_image, timestamp_ms: int):
    global _latest_hand_result
    with _hand_result_lock:
        _latest_hand_result = result

def get_latest_result():
    with _result_lock:
        return _latest_result

def get_latest_hand_result():
    with _hand_result_lock:
        return _latest_hand_result
    

# Verificações para o gatilho/disparo
def _is_pointing(lm) -> bool:
    return abs(lm[INDEX_PIP].x - lm[INDEX_MCP].x) < PIP_MCP_THRESOLD

def _is_side(lm) -> bool:
    return (lm[INDEX_PIP].x - lm[INDEX_MCP].x) > PIP_MCP_THRESOLD

def detect_trigger_pull(hand_result) -> bool:
    global _returned_to_point
    if not hand_result or not hand_result.hand_landmarks:
        _returned_to_point = True
        return False
    lm = hand_result.hand_landmarks[0]
    if _is_pointing(lm):
        _returned_to_point = True
        return False
    if _is_side(lm) and _returned_to_point:
        _returned_to_point = False
        return True
    return False

# Verificações para movimento e direção
def compute_arm_extension(landmarks) -> float:
    shoulder = landmarks[RIGHT_SHOULDER]
    wrist = landmarks[RIGHT_WRIST]
    dx = wrist.x - shoulder.x
    dy = wrist.y - shoulder.y
    return float(np.sqrt(dx ** 2 + dy ** 2))

def classify_arm_state(extension: float) -> str:
    if extension >= EXTENSION_THRESHOLD:
        return "FORWARD"
    return "STOP"

def classify_turn(wrist_x: float) -> str:
    x = 1.0 - wrist_x # Display flipado
    if x < 0.5 - CAM_DEAD_ZONE:
        return "TURN_LEFT"
    if x > 0.5 + CAM_DEAD_ZONE:
        return "TURN_RIGHT"
    return "STRAIGHT"

def get_action(move_state: str, turn_state: str, shoot: bool) -> list[bool]:
    forward = move_state == "FORWARD"
    turn_left = turn_state == "TURN_LEFT"
    turn_right = turn_state == "TURN_RIGHT"
    return [forward, False, turn_left, turn_right, shoot]

# Configs do VizDoom
def init_game() -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "freedoom1.cfg"))
    game.set_sound_enabled(True)
    game.add_game_args("+snd_efx 0")
    game.add_game_args("+snd_device 0") # Audio pro Windows (VzDoom funfa melhor no Linux)
    game.set_screen_resolution(vzd.ScreenResolution.RES_1920X1080)
    game.set_doom_map("E1M2") # Na fase 1 o elevador me confunde, 2 da pra reconhecer melhor
    game.set_doom_skill(1) # Dificuldade
    game.set_available_buttons([
        vzd.Button.MOVE_FORWARD,
        vzd.Button.MOVE_BACKWARD,
        vzd.Button.TURN_LEFT,
        vzd.Button.TURN_RIGHT,
        vzd.Button.ATTACK,
    ])
    game.init()
    time.sleep(2)
    return game

def run():
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        result_callback=_on_result,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        result_callback=_on_hand_result,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    DOOM_W, DOOM_H = 1920, 1080
    DOOM_X, DOOM_Y = 0, 0

    game = init_game()
    cap = cv2.VideoCapture(0)

    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    CAM_W = 640
    CAM_H = int(CAM_W * native_h / native_w)

    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Webcam", CAM_W, CAM_H)
    cv2.moveWindow("Webcam", DOOM_X + DOOM_W - CAM_W, DOOM_Y)

    with PoseLandmarker.create_from_options(options) as landmarker, \
         HandLandmarker.create_from_options(hand_options) as hand_landmarker:
        timestamp_ms = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms += 33
            landmarker.detect_async(mp_image, timestamp_ms)
            hand_landmarker.detect_async(mp_image, timestamp_ms)

            cv2.imshow("Webcam", cv2.flip(frame, 1))

            result = get_latest_result()
            hand_result = get_latest_hand_result()
            state = "STOP"
            turn_state = "STRAIGHT"

            if result and result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                wrist = landmarks[RIGHT_WRIST]
                if wrist.visibility >= POSE_VIZ_THRESHOLD:
                    extension = compute_arm_extension(landmarks)
                    state = classify_arm_state(extension)
                    turn_state = classify_turn(wrist.x)

            shoot = detect_trigger_pull(hand_result)
            action = get_action(state, turn_state, shoot)

            if not game.is_episode_finished():
                game.make_action(action)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if game.is_player_dead():
                game.respawn_player()

    cap.release()
    cv2.destroyAllWindows()
    game.close()

if __name__ == "__main__":
    run()
