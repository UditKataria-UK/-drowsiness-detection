import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time
import subprocess
from collections import deque


# =========================================================
# 1. CNN MODEL
# =========================================================

class DrowsinessCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================================================
# 2. LOAD MODEL
# =========================================================

model = DrowsinessCNN()

model.load_state_dict(
    torch.load(
        "drowsiness_model.pth",
        map_location="cpu"
    )
)

model.eval()


# =========================================================
# 3. IMAGE TRANSFORMATION
# =========================================================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# =========================================================
# 4. MEDIAPIPE FACE LANDMARKER
# =========================================================

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path="models/face_landmarker.task"
    ),

    running_mode=VisionRunningMode.IMAGE,

    num_faces=1
)


landmarker = (
    FaceLandmarker.create_from_options(
        options
    )
)


# =========================================================
# 5. WEBCAM
# =========================================================

cap = cv2.VideoCapture(0)


# =========================================================
# 6. SMOOTHING
# =========================================================

# Store recent eye predictions
prediction_history = deque(maxlen=8)


# =========================================================
# 7. DROWSINESS SETTINGS
# =========================================================

closed_start_time = None

DROWSY_TIME = 2.0

alarm_active = False


# =========================================================
# 8. MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # Mirror webcam
    frame = cv2.flip(frame, 1)


    height, width, _ = frame.shape


    # -----------------------------------------------------
    # BGR → RGB
    # -----------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # -----------------------------------------------------
    # MediaPipe image
    # -----------------------------------------------------

    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb_frame
    )


    # -----------------------------------------------------
    # Detect face
    # -----------------------------------------------------

    result = landmarker.detect(
        mp_image
    )


    # =====================================================
    # FACE FOUND
    # =====================================================

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]


        # -------------------------------------------------
        # Eye landmarks
        # -------------------------------------------------

        left_eye_indices = [
            33,
            133,
            159,
            145
        ]

        right_eye_indices = [
            362,
            263,
            386,
            374
        ]


        current_predictions = []


        # =================================================
        # PROCESS EACH EYE
        # =================================================

        for eye_indices in [

            left_eye_indices,
            right_eye_indices

        ]:

            # ---------------------------------------------
            # Landmark coordinates
            # ---------------------------------------------

            x_coords = [
                landmarks[i].x
                for i in eye_indices
            ]

            y_coords = [
                landmarks[i].y
                for i in eye_indices
            ]


            # ---------------------------------------------
            # Normalized → pixel coordinates
            # ---------------------------------------------

            x1 = int(
                min(x_coords) * width
            )

            x2 = int(
                max(x_coords) * width
            )

            y1 = int(
                min(y_coords) * height
            )

            y2 = int(
                max(y_coords) * height
            )


            # ---------------------------------------------
            # Padding
            # ---------------------------------------------

            padding_x = 35
            padding_y = 25


            x1 = max(
                0,
                x1 - padding_x
            )

            y1 = max(
                0,
                y1 - padding_y
            )

            x2 = min(
                width,
                x2 + padding_x
            )

            y2 = min(
                height,
                y2 + padding_y
            )


            # ---------------------------------------------
            # Crop eye
            # ---------------------------------------------

            eye = frame[
                y1:y2,
                x1:x2
            ]


            if eye.size == 0:
                continue


            # ---------------------------------------------
            # Draw eye box
            # ---------------------------------------------

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2
            )


            # ---------------------------------------------
            # BGR → RGB
            # ---------------------------------------------

            eye_rgb = cv2.cvtColor(
                eye,
                cv2.COLOR_BGR2RGB
            )


            # ---------------------------------------------
            # Convert to PIL
            # ---------------------------------------------

            eye_pil = Image.fromarray(
                eye_rgb
            )


            # ---------------------------------------------
            # Convert to Tensor
            # ---------------------------------------------

            eye_tensor = transform(
                eye_pil
            ).unsqueeze(0)


            # ---------------------------------------------
            # CNN prediction
            # ---------------------------------------------

            with torch.no_grad():

                output = model(
                    eye_tensor
                )

                predicted = torch.argmax(
                    output,
                    dim=1
                ).item()


            current_predictions.append(
                predicted
            )


        # =================================================
        # PROCESS CURRENT FRAME
        # =================================================

        if len(current_predictions) > 0:

            # ------------------------------------------------
            # If at least one eye is detected:
            #
            # 0 = Closed
            # 1 = Open
            # ------------------------------------------------

            closed_eyes = (
                current_predictions.count(0)
            )

            total_eyes = len(
                current_predictions
            )


            # ------------------------------------------------
            # Consider frame CLOSED if majority of detected
            # eyes are closed.
            # ------------------------------------------------

            if closed_eyes >= (
                total_eyes / 2
            ):

                frame_prediction = 0

            else:

                frame_prediction = 1


            # Add to history
            prediction_history.append(
                frame_prediction
            )


        # =================================================
        # MAJORITY VOTING
        # =================================================

        if len(prediction_history) >= 5:

            closed_count = (
                prediction_history.count(0)
            )

            open_count = (
                prediction_history.count(1)
            )


            # ---------------------------------------------
            # Recent frames mostly CLOSED
            # ---------------------------------------------

            if closed_count > open_count:

                eyes_closed = True

            else:

                eyes_closed = False


            # =================================================
            # DROWSINESS TIMER
            # =================================================

            if eyes_closed:

                if closed_start_time is None:

                    closed_start_time = (
                        time.time()
                    )


                closed_duration = (
                    time.time()
                    -
                    closed_start_time
                )


                # -----------------------------------------
                # DROWSY
                # -----------------------------------------

                if closed_duration >= DROWSY_TIME:

                    text = "DROWSY"


                    # -------------------------------------
                    # Alarm only once
                    # -------------------------------------

                    if not alarm_active:

                        subprocess.Popen(
                            [
                                "say",
                                "Wake up! You are feeling sleepy."
                            ]
                        )

                        alarm_active = True


                else:

                    text = "EYES CLOSED"


            # =================================================
            # EYES OPEN
            # =================================================

            else:

                closed_start_time = None

                alarm_active = False

                text = "ALERT"


        else:

            text = "DETECTING..."


    # =====================================================
    # NO FACE
    # =====================================================

    else:

        prediction_history.clear()

        closed_start_time = None

        alarm_active = False

        text = "NO FACE"


    # =====================================================
    # DISPLAY TEXT
    # =====================================================

    cv2.putText(

        frame,

        text,

        (30, 50),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.2,

        (0, 0, 255),

        3
    )


    # =====================================================
    # SHOW WINDOW
    # =====================================================

    cv2.imshow(

        "Driver Drowsiness Detection",

        frame
    )


    # =====================================================
    # PRESS Q TO EXIT
    # =====================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# =========================================================
# 9. CLEANUP
# =========================================================

cap.release()

landmarker.close()

cv2.destroyAllWindows()