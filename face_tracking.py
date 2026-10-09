r"""Webカメラで複数人の顔をトラッキングする (ローカル / Python 3.14 / OpenCV 5系対応)

face_tracking.py の複数人版。顔検出は YuNet (FaceDetectorYN) で、
画面内の全員を検出し、人ごとにIDを付けて追跡します。
モデルファイルは face_tracking.py と同じものを使います (初回は自動ダウンロード)。

セットアップ (仮想環境を推奨):
    # Windows
    py -3.14 -m venv .venv
    .venv\Scripts\activate

    # macOS / Linux
    python3.14 -m venv .venv
    source .venv/bin/activate

    # ウィンドウ表示(imshow)が必要なので、headless ではなく opencv-python を使う
    pip install opencv-python

使い方:
    python face_tracking_multi.py
    q キーまたは ESC キーで終了
"""

import shutil
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_NAME = "face_detection_yunet_2023mar.onnx"
MODEL_URLS = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + MODEL_NAME,
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/" + MODEL_NAME,
)
MODEL_PATH = Path(__file__).resolve().with_name(MODEL_NAME)
MIN_MODEL_BYTES = 100_000  # 数百バイトのポインタファイルやエラーページを弾く

CAMERA_INDEX = 0        # カメラが複数あるときは 1, 2 ... に変える
SCORE_THRESHOLD = 0.8   # 顔らしさの下限 (上げると誤検出が減り、下げると見つけやすくなる)
NMS_THRESHOLD = 0.3
TOP_K = 5000
ALPHA = 0.3             # 平滑化の強さ (小さいほど滑らか、大きいほど追従が速い)
MAX_MISSES = 10         # 何フレーム連続で見失ったら、その人のトラックを消すか
MATCH_RATIO = 1.0       # 前フレームの顔から「顔の幅 x この値」以内なら同じ人とみなす

# 人ごとの枠の色 (BGR)。IDの順に使い回す
COLORS = [
    (0, 255, 0), (255, 128, 0), (0, 200, 255),
    (255, 0, 255), (0, 128, 255), (255, 255, 0),
]


class Track:
    """1人ぶんの追跡状態"""

    def __init__(self, track_id: int, state: np.ndarray) -> None:
        self.id = track_id
        self.state = state  # 平滑化した [x, y, w, h, 特徴点5つ(x, y)] の14個
        self.misses = 0     # 連続で見失っているフレーム数


def center(v: np.ndarray) -> np.ndarray:
    return np.array([v[0] + v[2] / 2, v[1] + v[3] / 2])


def update_tracks(
    tracks: list[Track], detections: list[np.ndarray], next_id: int
) -> tuple[list[Track], int]:
    """今フレームの検出結果を、既存のトラックに結びつける。

    1. 距離が近い (トラック, 検出) の組から順に、同じ人として割り当てる
    2. 割り当てられたトラックは平滑化して位置を更新する
    3. 割り当てられなかったトラックは見失い回数を増やし、MAX_MISSES を超えたら消す
    4. 割り当てられなかった検出は、新しい人としてIDを付けて追加する
    """
    pairs = []
    for ti, track in enumerate(tracks):
        for di, det in enumerate(detections):
            dist = float(np.linalg.norm(center(track.state) - center(det)))
            if dist <= MATCH_RATIO * max(track.state[2], det[2]):
                pairs.append((dist, ti, di))
    pairs.sort()

    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    for _, ti, di in pairs:
        if ti in used_tracks or di in used_dets:
            continue
        track = tracks[ti]
        track.state = ALPHA * detections[di] + (1 - ALPHA) * track.state
        track.misses = 0
        used_tracks.add(ti)
        used_dets.add(di)

    for ti, track in enumerate(tracks):
        if ti not in used_tracks:
            track.misses += 1
    tracks = [t for t in tracks if t.misses <= MAX_MISSES]

    for di, det in enumerate(detections):
        if di not in used_dets:
            tracks.append(Track(next_id, det.copy()))
            next_id += 1
    return tracks, next_id


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=30) as res, open(tmp, "wb") as f:
            shutil.copyfileobj(res, f)
        size = tmp.stat().st_size
        if size < MIN_MODEL_BYTES:
            raise ValueError(f"ファイルが小さすぎます ({size} バイト)")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def ensure_model() -> Path:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size >= MIN_MODEL_BYTES:
        return MODEL_PATH

    errors = []
    for url in MODEL_URLS:
        print(f"顔検出モデルをダウンロードしています: {url}")
        try:
            download(url, MODEL_PATH)
            return MODEL_PATH
        except Exception as e:
            errors.append(f"  {url}\n    -> {e}")

    raise SystemExit(
        "モデルを自動ダウンロードできませんでした。\n"
        + "\n".join(errors)
        + "\n\n次のURLをブラウザで開いて保存し、このスクリプトと同じフォルダに置いてください:\n"
        + f"  URL: {MODEL_URLS[0]}\n"
        + f"  保存名: {MODEL_NAME}\n"
        + f"  置き場所: {MODEL_PATH.parent}"
    )


def open_camera() -> cv2.VideoCapture:
    # Windows は DirectShow を指定すると、起動が速く安定しやすい
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    return cv2.VideoCapture(CAMERA_INDEX, backend)


def main() -> None:
    if not hasattr(cv2, "FaceDetectorYN"):
        raise SystemExit(
            f"OpenCV {cv2.__version__} には FaceDetectorYN がありません。\n"
            "次を実行して更新してください: pip install -U opencv-python"
        )

    model = ensure_model()
    input_size = (640, 480)
    detector = cv2.FaceDetectorYN.create(
        str(model), "", input_size, SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K
    )

    cap = open_camera()
    if not cap.isOpened():
        raise SystemExit(
            "カメラを開けませんでした。\n"
            "・他のアプリがカメラを使っていないか確認してください\n"
            "・macOSは、ターミナルにカメラの許可を出してください\n"
            "・カメラが複数あるときは CAMERA_INDEX を変えてください"
        )

    tracks: list[Track] = []
    next_id = 1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # 鏡のように左右反転
            frame_h, frame_w = frame.shape[:2]
            if (frame_w, frame_h) != input_size:
                input_size = (frame_w, frame_h)
                detector.setInputSize(input_size)

            # faces は顔がないと None。各行: [0-3]=枠(x,y,w,h) [4-13]=特徴点5つ [14]=スコア
            _, faces = detector.detect(frame)
            detections = [] if faces is None else [np.array(f[:14], dtype=float) for f in faces]
            tracks, next_id = update_tracks(tracks, detections, next_id)

            for track in tracks:
                color = COLORS[(track.id - 1) % len(COLORS)]
                x, y, box_w, box_h = track.state[:4].astype(int)
                cx, cy = x + box_w // 2, y + box_h // 2
                cv2.rectangle(frame, (int(x), int(y)), (int(x + box_w), int(y + box_h)), color, 2)
                for px, py in track.state[4:14].reshape(5, 2).astype(int):
                    cv2.circle(frame, (int(px), int(py)), 3, color, -1)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame, f"ID {track.id} ({cx}, {cy})", (int(x), max(int(y) - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )

            cv2.putText(
                frame, f"people: {len(tracks)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )
            cv2.imshow("Face Tracking (multi)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
