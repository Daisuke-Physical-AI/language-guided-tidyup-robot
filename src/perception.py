"""俯瞰カメラの画像を VLM（ollama の gemma3）に見せ、机の上に何があるかを認識する。

腕が作業域の真上にあると物体が隠れるので、撮影の前に腕を手前に畳む（LOOK_Q）。
位置の計測はシミュレータ側（モーションキャプチャ相当）に任せ、VLM は「何色のブロックがあるか」の
意味的な認識だけを受け持つ。認識結果は LLM の計画に渡す物体リストになる。
"""
import base64
import io
import json

import mujoco
import numpy as np
from PIL import Image

import planner
from video import top_camera

LOOK_Q = np.array([0.0, -1.3, 0.0, -2.9, 0.0, 1.65, 0.785])   # 腕を畳んでカメラの視界から外す姿勢
KNOWN_COLORS = ["red", "green", "blue", "yellow", "orange", "purple"]

PROMPT = (
    "This is a top-down camera image of a robot's table. The robot base is at the bottom edge of the image, "
    "and the two empty trays are on the left and right. List every small colored cube lying on the table. "
    f"Possible colors: {', '.join(KNOWN_COLORS)}. Do not list anything that is not a cube.\n"
    'Answer in JSON only: {"cubes": ["<color>", ...]}'
)


class TopCamera:
    def __init__(self, model, size=720):
        self.r = mujoco.Renderer(model, size, size)
        self.cam = top_camera()

    def shoot(self, data):
        self.r.update_scene(data, camera=self.cam)
        img = self.r.render()
        return np.rot90(img, 2).copy()          # 台座が画面下・+y が画面左になる向きに回す

    def close(self):
        self.r.close()


def take_top_image(arm, top_cam):
    arm.move_q(LOOK_Q, 1.2)
    arm.settle(0.3)
    return top_cam.shoot(arm.d)


def perceive(img):
    """画像 → {"colors": [...], "raw": LLM の生出力, "stats": 所要時間など}"""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    content, stats = planner._chat([{"role": "user", "content": PROMPT}], images=[b64])
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {}
    seen = []
    for c in data.get("cubes", []) or []:
        c = str(c).lower().strip()
        if c in KNOWN_COLORS and c not in seen:
            seen.append(c)
    return {"colors": seen, "raw": content, "stats": stats}
