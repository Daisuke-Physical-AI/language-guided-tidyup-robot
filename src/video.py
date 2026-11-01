"""オフスクリーン描画 → 字幕の帯 → MP4。実行中の状態（指示文・AI の解釈・今の段階と動作）を画面に出す。"""
import os
import mujoco
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

import scene

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)

# 同梱の Noto Sans JP（OFL、assets/fonts/OFL.txt）を優先。無ければシステムのフォントへ。
FONT_CANDIDATES = [
    os.path.join(PROJ, "assets", "fonts", "NotoSansJP.ttf"),
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

JP_COLOR = {"red": "赤", "green": "緑", "blue": "青", "yellow": "黄", "orange": "橙", "purple": "紫"}
JP_TRAY = {"tray_left": "左トレイ", "tray_right": "右トレイ"}
PHASES = ["① 見る", "② 考える", "③ 動かす", "④ 確かめる"]     # 表紙の図と同じ番号・名前


def _wrap(text, font, max_w, max_lines=2):
    """画面幅に合わせて折り返す。空白で切れないとき（日本語）は文字単位で切る。超えた分は … で省略。"""
    import re
    lines, cur = [], ""
    for word in re.split(r"(\s+)", text):
        if font.getlength(cur + word) <= max_w:
            cur += word
            continue
        if cur:
            lines.append(cur.rstrip())
            cur = ""
        while font.getlength(word) > max_w:
            k = len(word)
            while k > 1 and font.getlength(word[:k]) > max_w:
                k -= 1
            lines.append(word[:k])
            word = word[k:]
        cur = word
    if cur:
        lines.append(cur.rstrip())
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:-1] + "…"
    return lines


def _font(size, weight="Regular"):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
            except OSError:
                continue
            try:                      # 可変フォントは既定が Thin なので指定の太さに寄せる
                f.set_variation_by_name(weight)
            except Exception:         # noqa: BLE001  可変でない／未対応なら無視
                pass
            return f
    return ImageFont.load_default()


def _rgb(color):
    """scene.COLORS（0〜1）を 0〜255 に。知らない色は灰色。"""
    c = scene.COLORS.get(color)
    return tuple(int(v * 255) for v in c) if c else (128, 128, 128)


def _chip(draw, x, y, color, side):
    """ブロックの色を示す小さな四角。文字の代わりに色そのものを見せる。"""
    draw.rounded_rectangle([x, y, x + side, y + side], radius=5, fill=_rgb(color), outline=(255, 255, 255), width=2)
    return x + side


def _status_text(parts):
    """status の部品列を文字だけにする（1 画面モードとログ用）。"""
    return "".join(JP_COLOR.get(p[1], p[1]) if isinstance(p, (tuple, list)) else p for p in parts or [])


# 3 画面レイアウト（views を渡したとき）: 上に字幕の帯、下に 640px 幅の画面を 3 つ、最下段に名札
PANEL_W, BAND_H, CAP_H, TRIPLE_H = 640, 200, 44, 1080


class Recorder:
    """物理ステップごとに tick() を呼ぶと、fps に合わせた間隔でフレームを取る。

    views=None なら 1 画面に字幕を重ねる。views（triple_views() の戻り値）を渡すと、
    字幕を映像に重ねず上の帯に置き、3 つのカメラを横に並べた 1920x1080 にする。

    overlay の中身（set_overlay で更新する）:
      instruction  指示文
      goal         [(色, トレイ名), ...]  AI の解釈。無ければ「これから決める」と出す
      phase        0〜3（PHASES の添字）。None なら段階名を出さない
      status       今の動作。文字列と ("chip", 色) を並べたリスト
      inset        1 画面モードで右上に出す俯瞰画像
    """

    def __init__(self, model, out_path, width=1280, height=720, fps=30, speed=1.0, cam=None, text_scale=1.0,
                 views=None):
        self.m = model
        self.out_path = out_path
        self.fps = fps
        self.views = views
        if views:
            self.W, self.H = PANEL_W * len(views), TRIPLE_H
            self.panel_h = self.H - BAND_H - CAP_H
            self.r = mujoco.Renderer(model, self.panel_h, PANEL_W)
        else:
            self.W, self.H = width, height
            self.r = mujoco.Renderer(model, height, width)
        self.cam = cam or default_camera()
        # 倍速 speed のとき、実時間 1/fps 秒ぶんの物理ステップ × speed ごとに 1 フレーム
        self.steps_per_frame = max(int(round(speed / (fps * model.opt.timestep))), 1)
        self.counter = 0
        self.frames = []
        self.overlay = {"status": []}
        self.set_text_scale(text_scale)

    def set_text_scale(self, s):
        """字幕の大きさ。帯の文字は 1920 幅・text_scale 1.3 のときに 32/28/24px になるよう設計した。"""
        k = s / 1.3
        self.font_big = _font(int(32 * k))
        self.font_mid = _font(int(28 * k))
        self.font_bold = _font(int(28 * k), "Bold")
        self.font_dim = _font(int(24 * k))
        self.font_label = _font(int(21 * k))
        self.chip = int(28 * k)
        self.font = _font(int(26 * s))          # 1 画面モード用
        self.font_small = _font(int(20 * s))
        self.inset_side = int(230 * s)

    def set_overlay(self, **kw):
        self.overlay.update(kw)

    def tick(self, data):
        self.counter += 1
        if self.counter % self.steps_per_frame == 0:
            self.capture(data)

    def capture(self, data, hold_frames=1):
        if self.views:
            img = self._compose_views(data)
        else:
            self.r.update_scene(data, camera=self.cam)
            img = self._draw_overlay(self.r.render())
        for _ in range(hold_frames):
            self.frames.append(img)

    def _draw_status(self, draw, x, y):
        for part in self.overlay.get("status") or []:
            if isinstance(part, (tuple, list)):
                x = _chip(draw, x, y + 2, part[1], self.chip) + 8
            else:
                draw.text((x, y), part, font=self.font_mid, fill=(235, 235, 235))
                x += self.font_mid.getlength(part)
        return x

    def _draw_band(self, draw):
        """上の帯（3 段）: 指示文 ／ AI の解釈 ／ 今の段階と動作。囲みや太字の強調は使わない。"""
        k = self.font_mid.size / 28
        if self.overlay.get("instruction"):
            draw.text((20, int(12 * k)), "「" + self.overlay["instruction"] + "」", font=self.font_big,
                      fill=(255, 245, 200))
        y, x = int(66 * k), 20
        draw.text((x, y + 4), "AI の解釈：", font=self.font_dim, fill=(150, 152, 165))
        x += self.font_dim.getlength("AI の解釈：")
        goal = self.overlay.get("goal")
        if goal:
            by_tray = {}
            for color, tray in goal:                      # 同じトレイ行きをまとめて「■■ → 左トレイ」
                by_tray.setdefault(tray, []).append(color)
            for tray, colors in by_tray.items():
                for c in colors:
                    x = _chip(draw, x, y + 2, c, self.chip) + 6
                s = "→ " + JP_TRAY.get(tray, tray)
                draw.text((x + 6, y), s, font=self.font_mid, fill=(235, 235, 235))
                x += 6 + self.font_mid.getlength(s) + 36
        else:
            draw.text((x, y), "（これから決める）", font=self.font_mid, fill=(140, 142, 155))
        y, x = int(124 * k), 20
        if self.overlay.get("phase") is not None:
            s = PHASES[self.overlay["phase"]]
            draw.text((x, y), s, font=self.font_bold, fill=(255, 255, 255))
            x += self.font_bold.getlength(s) + 30
        self._draw_status(draw, x, y)

    def _compose_views(self, data):
        """3 画面を描いて並べ、上の帯に字幕、下の段に各画面の名札を置く。映像に文字は重ねない。"""
        bg = (16, 18, 24)
        canvas = Image.new("RGB", (self.W, self.H), bg)
        for i, v in enumerate(self.views):
            self.r.update_scene(data, camera=v["cam"])
            img = self.r.render()
            if v.get("rot180"):
                img = np.rot90(img, 2)
            canvas.paste(Image.fromarray(np.ascontiguousarray(img)), (i * PANEL_W, BAND_H))
        draw = ImageDraw.Draw(canvas, "RGBA")
        y_cap = BAND_H + self.panel_h
        for i, v in enumerate(self.views):
            if i:
                draw.line([(i * PANEL_W, BAND_H), (i * PANEL_W, y_cap)], fill=bg, width=4)
            draw.text((i * PANEL_W + 12, y_cap + 10), v["label"], font=self.font_label, fill=(225, 225, 225))
        self._draw_band(draw)
        return np.asarray(canvas)

    def hold(self, data, seconds):
        self.capture(data, hold_frames=int(seconds * self.fps))

    def _band_lines(self):
        """1 画面モード用。帯の内容を文字だけの行にする。"""
        lines = []
        if self.overlay.get("instruction"):
            lines.append(("指示: " + self.overlay["instruction"], self.font, (255, 230, 120)))
        goal = self.overlay.get("goal")
        if goal:
            txt = ", ".join(f"{JP_COLOR.get(c, c)}→{JP_TRAY.get(t, t)}" for c, t in goal)
            lines.append(("AI の解釈: " + txt, self.font_small, (180, 230, 255)))
        st = _status_text(self.overlay.get("status"))
        if self.overlay.get("phase") is not None:
            st = PHASES[self.overlay["phase"]] + "  " + st
        if st:
            lines.append((st, self.font_small, (220, 220, 220)))
        return lines

    def _draw_overlay(self, img):
        im = Image.fromarray(img)
        draw = ImageDraw.Draw(im, "RGBA")
        W, H = im.size
        y = 14
        pad = 10
        wrapped = []
        for text, font, color in self._band_lines():
            for ln in _wrap(text, font, W - 2 * (pad + 6)):
                wrapped.append((ln, font, color))
        box_h = sum(f.size + 10 for _, f, _ in wrapped) + pad * 2
        draw.rectangle([0, 0, W, box_h], fill=(0, 0, 0, 140))
        for text, font, color in wrapped:
            draw.text((pad + 6, y), text, font=font, fill=color)
            y += font.size + 10
        inset = self.overlay.get("inset")
        if inset is not None:                      # VLM に見せた俯瞰画像を右上に小窓で出す
            side = self.inset_side
            x0, y0 = W - side - 14, box_h + 12
            draw.rectangle([x0 - 3, y0 - 3, x0 + side + 3, y0 + side + 26], fill=(0, 0, 0, 150))
            im.paste(Image.fromarray(inset).resize((side, side)), (x0, y0))
            draw.text((x0 + 4, y0 + side + 2), "俯瞰カメラ（VLM の入力）", font=self.font_label, fill=(220, 220, 220))
        return np.asarray(im)

    def save(self):
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        w = imageio.get_writer(self.out_path, fps=self.fps, codec="libx264", quality=8, pixelformat="yuv420p",
                               macro_block_size=1)
        for f in self.frames:
            w.append_data(f)
        w.close()
        return self.out_path, len(self.frames) / self.fps

    def close(self):
        self.r.close()


# 動画用カメラの候補。すべてロボットの背後側（azimuth < 0）で、ロボットの左（+y）が画面の左に来る。
CAMERAS = {
    "back_left":       dict(lookat=[0.42, 0.03, 0.05], distance=1.35, azimuth=-40, elevation=-30),  # 初期の画角
    "back_left_close": dict(lookat=[0.46, 0.00, 0.03], distance=1.10, azimuth=-38, elevation=-40),
    "side_left":       dict(lookat=[0.45, 0.00, 0.03], distance=1.20, azimuth=-65, elevation=-32),
    "high":            dict(lookat=[0.46, 0.00, 0.00], distance=1.15, azimuth=-30, elevation=-52),
}


def make_camera(spec="back_left"):
    """名前（CAMERAS のキー）か、lookat/distance/azimuth/elevation の辞書から MjvCamera を作る。"""
    if isinstance(spec, str):
        spec = CAMERAS[spec]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = spec["lookat"]
    cam.distance = spec["distance"]
    cam.azimuth = spec["azimuth"]
    cam.elevation = spec["elevation"]
    return cam


def default_camera():
    return make_camera("back_left")


def triple_views():
    """3 画面: ロボット視点（俯瞰カメラ＝VLM の入力）／ ISO ／ 正面。ロボット全体が常に入る距離にしてある。"""
    # 俯角は 23° より深くする（fovy 45° の上端が地平線を超えると、空が無いので黒く映る）
    return [
        {"label": "ロボット視点（VLM入力映像）",
         "cam": make_camera(dict(lookat=[0.47, 0.0, 0.0], distance=1.6, azimuth=180, elevation=-89)), "rot180": True},
        {"label": "ISO ビュー",
         "cam": make_camera(dict(lookat=[0.35, 0.0, 0.25], distance=1.75, azimuth=-135, elevation=-28))},
        {"label": "正面ビュー",
         "cam": make_camera(dict(lookat=[0.35, 0.0, 0.22], distance=1.80, azimuth=180, elevation=-30))},
    ]


def top_camera():
    """VLM に見せる真上からの図。take_top_image 側で 180° 回し、ロボット台座が画面下・+y が画面左になる。"""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.47, 0.0, 0.0]
    cam.distance = 1.35
    cam.azimuth = 180
    cam.elevation = -89
    return cam
