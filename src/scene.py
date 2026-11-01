"""机上シーンを MjSpec で組み立てる。

MuJoCo Menagerie の panda.xml を読み込み、手先の TCP サイト・床・作業マット・2つのトレイ・
色つきブロック（乱数配置）をプログラムから足す。XML を書き換えずに済むので、
Menagerie 側はそのまま（git 管理外の third_party/）に置ける。
"""
import os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
PANDA_XML = os.path.join(PROJ, "third_party", "mujoco_menagerie", "franka_emika_panda", "panda.xml")

BOX = mujoco.mjtGeom.mjGEOM_BOX

CUBE_HALF = 0.02                     # 4 cm 角
MAT_TOP = 0.006                      # 作業マットの上面
CUBE_Z0 = MAT_TOP + CUBE_HALF + 0.001
COLORS = {
    "red": (0.90, 0.15, 0.15),
    "green": (0.15, 0.70, 0.25),
    "blue": (0.15, 0.40, 0.90),
    "yellow": (0.95, 0.85, 0.10),
    "orange": (0.95, 0.50, 0.10),
    "purple": (0.55, 0.20, 0.75),
}
TRAYS = {"tray_left": (0.45, 0.36), "tray_right": (0.45, -0.36)}
TRAY_HALF = 0.11
TRAY_BASE_TOP = 0.008
TRAY_RIM_H = 0.012
SLOTS = [(-0.05, -0.05), (0.05, -0.05), (-0.05, 0.05), (0.05, 0.05)]   # トレイ内の置き場
WORKSPACE = ((0.36, 0.60), (-0.20, 0.20))                              # ブロックの初期範囲
MIN_DIST = 0.11                                                        # ブロック同士の最小間隔


class SceneInfo:
    def __init__(self, cubes, trays):
        self.cubes = list(cubes)                 # body 名のリスト（例 "red_cube"）
        self.trays = dict(trays)                 # 名前 → (x, y)
        self.used_slots = {t: 0 for t in trays}

    @property
    def tray_names(self):
        return list(self.trays)

    def next_slot(self, tray):
        k = self.used_slots[tray] % len(SLOTS)
        self.used_slots[tray] += 1
        tx, ty = self.trays[tray]
        sx, sy = SLOTS[k]
        return np.array([tx + sx, ty + sy, TRAY_BASE_TOP + CUBE_HALF])

    def in_tray(self, pos, tray):
        tx, ty = self.trays[tray]
        return abs(pos[0] - tx) < TRAY_HALF and abs(pos[1] - ty) < TRAY_HALF and pos[2] < 0.08

    def which_tray(self, pos):
        for t in self.trays:
            if self.in_tray(pos, t):
                return t
        return None


def _sample_positions(rng, n):
    (x0, x1), (y0, y1) = WORKSPACE
    pts = []
    for _ in range(5000):
        p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        if all(np.linalg.norm(p - q) >= MIN_DIST for q in pts):
            pts.append(p)
            if len(pts) == n:
                return pts
    raise RuntimeError("ブロックの配置に失敗（間隔条件が厳しすぎる）")


def build(seed=0, cubes=("red", "green", "blue", "yellow"), offscreen=(1920, 1080)):
    if not os.path.exists(PANDA_XML):
        raise FileNotFoundError(f"{PANDA_XML} が無い。scripts/setup.sh で取得する")
    spec = mujoco.MjSpec.from_file(PANDA_XML)
    for k in list(spec.keys):          # nq が変わるので Menagerie のキーフレームは捨てる
        spec.delete(k)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = offscreen

    spec.body("hand").add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.004] * 3, rgba=[0, 1, 0, 0])

    wb = spec.worldbody
    wb.add_light(pos=[0, 0, 2.0], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                 diffuse=[0.6] * 3, specular=[0.1] * 3)
    wb.add_light(pos=[1.2, -1.0, 1.6], dir=[-0.6, 0.5, -0.8], diffuse=[0.45] * 3, specular=[0.05] * 3)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], rgba=[0.80, 0.82, 0.86, 1])
    # 作業マット（ブロックの初期位置の下）
    (x0, x1), (y0, y1) = WORKSPACE
    # 色は明るめ: 暗いと青ブロックとのコントラストが落ちて VLM が数え落とす（2026-10-28 実測）
    wb.add_geom(name="mat", type=BOX, pos=[(x0 + x1) / 2, 0, MAT_TOP / 2],
                size=[(x1 - x0) / 2 + 0.06, (y1 - y0) / 2 + 0.06, MAT_TOP / 2], rgba=[0.62, 0.64, 0.66, 1])
    # トレイ（底＋4辺の低い縁）
    for name, (tx, ty) in TRAYS.items():
        b = wb.add_body(name=name, pos=[tx, ty, 0])
        b.add_geom(name=f"{name}_base", type=BOX, pos=[0, 0, TRAY_BASE_TOP / 2],
                   size=[TRAY_HALF, TRAY_HALF, TRAY_BASE_TOP / 2], rgba=[0.93, 0.90, 0.80, 1])
        t = 0.006
        for i, (px, py, sx, sy) in enumerate([
            (TRAY_HALF - t, 0, t, TRAY_HALF), (-(TRAY_HALF - t), 0, t, TRAY_HALF),
            (0, TRAY_HALF - t, TRAY_HALF, t), (0, -(TRAY_HALF - t), TRAY_HALF, t),
        ]):
            b.add_geom(name=f"{name}_rim{i}", type=BOX, pos=[px, py, TRAY_BASE_TOP + TRAY_RIM_H / 2],
                       size=[sx, sy, TRAY_RIM_H / 2], rgba=[0.75, 0.68, 0.55, 1])
    # ブロック
    rng = np.random.default_rng(seed)
    names = []
    for color, (x, y) in zip(cubes, _sample_positions(rng, len(cubes))):
        name = f"{color}_cube"
        b = wb.add_body(name=name, pos=[x, y, CUBE_Z0])
        b.add_freejoint()
        b.add_geom(name=f"{name}_geom", type=BOX, size=[CUBE_HALF] * 3, rgba=[*COLORS[color], 1],
                   mass=0.05, friction=[1.5, 0.01, 0.001], condim=4)
        names.append(name)

    model = spec.compile()
    return model, SceneInfo(names, TRAYS)


def body_pos(model, data, name):
    return data.xpos[model.body(name).id].copy()
