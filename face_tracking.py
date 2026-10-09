r"""Webカメラで顔をトラッキングする最小サンプル (ローカル / Python 3.14)

セットアップ (仮想環境を推奨):
    # Windows
    py -3.14 -m venv .venv
    .venv\Scripts\activate

    # macOS / Linux
    python3.14 -m venv .venv
    source .venv/bin/activate

    # OpenCV 5系には CascadeClassifier がないので 4系を入れる。
    # ウィンドウ表示(imshow)が必要なので、headless ではなく opencv-python を使う。
    pip install "opencv-python<5"

使い方:
    python face_tracking.py
    q キーまたは ESC キーで終了
"""

import sys

import cv2
import numpy as np

CAMERA_INDEX = 0     # カメラが複数あるときは 1, 2 ... に変える
ALPHA = 0.3          # 平滑化の強さ (小さいほど滑らか、大きいほど追従が速い)
MAX_MISSES = 10      # 何フレーム連続で見失ったらトラッキングをリセットするか


def open_camera() -> cv2.VideoCapture:
    # Windows は DirectShow を指定すると、起動が速く安定しやすい
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    return cv2.VideoCapture(CAMERA_INDEX, backend)


def main() -> None:
    if not hasattr(cv2, "CascadeClassifier"):
        raise SystemExit(
            f"OpenCV {cv2.__version__} には CascadeClassifier がありません。\n"
            '次を実行して4系を入れてください: pip install "opencv-python<5"'
        )
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = open_camera()
    if not cap.isOpened():
        raise SystemExit(
            "カメラを開けませんでした。\n"
            "・他のアプリがカメラを使っていないか確認してください\n"
            "・macOSは、ターミナルにカメラの許可を出してください\n"
            "・カメラが複数あるときは CAMERA_INDEX を変えてください"
        )

    smoothed: np.ndarray | None = None  # 平滑化した (x, y, w, h)
    misses = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # 鏡のように左右反転
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )

            if len(faces) > 0:
                # 一番大きい顔を追跡対象にする
                target = np.array(max(faces, key=lambda f: f[2] * f[3]), dtype=float)
                smoothed = target if smoothed is None else ALPHA * target + (1 - ALPHA) * smoothed
                misses = 0
            else:
                misses += 1
                if misses > MAX_MISSES:
                    smoothed = None

            if smoothed is not None:
                x, y, w, h = smoothed.astype(int)
                cx, cy = x + w // 2, y + h // 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame, f"center=({cx}, {cy})", (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

            cv2.imshow("Face Tracking", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
