"""言語指示 → 俯瞰カメラを VLM が見る → LLM が計画 → Panda が実行 → もう一度見て検証 → 必要なら再計画。
全部を 1 本の動画にする。

使い方:
  MUJOCO_GL=egl .venv/bin/python src/main.py --instruction "暖色系を左のトレイに、寒色系を右に" --seed 1
  --disturb を付けると、1 回目の実行後にブロックを 1 つトレイから弾き出し、回復する様子を撮る。
"""
import argparse
import json
import os
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scene            # noqa: E402
import planner          # noqa: E402
import perception       # noqa: E402
from control import Arm, cube_yaw  # noqa: E402
from video import Recorder, triple_views, JP_COLOR, JP_TRAY  # noqa: E402

PROJ = os.path.dirname(HERE)
MAX_ATTEMPTS = 4        # 観測→計画→実行 の最大回数（通常は 2 回: 実行＋確認）


def color_of(name):
    """"red_cube" → "red"。帯の色チップはこの名前で引く。"""
    return name.replace("_cube", "")


def jp_obj(name):
    return JP_COLOR.get(color_of(name), name)


def jp_tray(name):
    return JP_TRAY.get(name, name)


def kick(model, data, name, target_xy=(0.48, 0.0), speed=1.6, up=1.2):
    """外乱: ブロックをトレイから作業マットの方へ弾き飛ばす（デモ用）。"""
    jid = model.body(name).jntadr[0]
    adr = model.jnt_dofadr[jid]
    p = scene.body_pos(model, data, name)
    d = np.array([target_xy[0] - p[0], target_xy[1] - p[1]])
    d /= max(np.linalg.norm(d), 1e-6)
    data.qvel[adr:adr + 3] = [d[0] * speed, d[1] * speed, up]


def run(instruction, seed, out_path, speed=1.0, cubes=("red", "green", "blue", "yellow"), title=None,
        disturb=False, text_scale=1.0, cam=None, views=None, fps=30):
    t_start = time.time()
    planner.check_server()
    model, info = scene.build(seed=seed, cubes=cubes)
    data = mujoco.MjData(model)
    rec = Recorder(model, out_path, speed=speed, text_scale=text_scale, cam=cam, views=views, fps=fps)
    top_cam = perception.TopCamera(model)
    log = {"instruction": instruction, "seed": seed, "model": planner.MODEL, "disturb": disturb, "attempts": []}

    def say(msg):
        print(msg, flush=True)

    def location(name):
        if name not in info.cubes:
            return "not found"
        t = info.which_tray(scene.body_pos(model, data, name))
        return f"in {t}" if t else "on the table"

    arm = Arm(model, data, recorder=rec, log=say)
    rec.set_overlay(instruction=instruction, phase=None, status=[])
    arm.settle(0.4)
    rec.hold(data, 1.0)

    # 指示文の解釈（どの物体をどの容器へ）は LLM に 1 回だけ確定させる。
    # 以後の再計画は「目標 − 現在の状態」をコードで決定的に求める。
    # （再計画のたびに LLM に聞き直すと、小さなモデルは解釈を変えてしまう: 2026-10-28 シナリオ 3 で実測）
    goal = {}            # 物体 → 目的の容器
    told = set()         # LLM に伝えた物体。新しく見えた物体があるときだけ LLM に聞き足す
    done = False
    disturbed = False
    for attempt in range(1, MAX_ATTEMPTS + 1):
        a = {"attempt": attempt}
        log["attempts"].append(a)

        # 1) 観測: 腕を畳んで俯瞰カメラで撮り、VLM に何があるか聞く
        rec.set_overlay(phase=0, status=["カメラで机を撮影する"])
        img = perception.take_top_image(arm, top_cam)
        rec.set_overlay(inset=img)
        per = perception.perceive(img)
        objects = [f"{c}_cube" for c in per["colors"]]
        truth, seen = set(info.cubes), set(objects)
        a["vlm"] = {"colors": per["colors"], "missed": sorted(truth - seen), "extra": sorted(seen - truth),
                    "raw": per["raw"], "stats": per["stats"]}
        say(f"[{attempt}] VLM sees: {per['colors']}  missed={sorted(truth - seen)} extra={sorted(seen - truth)}"
            f"  ({per['stats']['wall_s']:.1f}s)")
        seen_parts = [("chip", c) for c in per["colors"]] or ["なし"]
        rec.set_overlay(status=["カメラに映ったもの："] + seen_parts)
        rec.hold(data, 0.8)

        # 2) 計画: LLM に聞くのは、まだ伝えていない物体があるときだけ（通常は最初の 1 回）
        new = [o for o in objects if o not in told]
        if new:
            state = {o: location(o) for o in objects}
            rec.set_overlay(phase=1, status=["指示文を読み解いている"])
            res = planner.plan(instruction, state, info.tray_names)
            a["plan"] = {k: res[k] for k in ("reasoning", "steps", "dropped", "raw", "stats", "prompt")}
            for s in res["steps"]:
                goal.setdefault(s["pick"], s["place"])      # 一度決めた解釈は上書きしない
            told.update(objects)
            say(f"[{attempt}] LLM: {res['reasoning']}  ({res['stats']['wall_s']:.1f}s)")
        goal_txt = ", ".join(f"{jp_obj(o)}→{jp_tray(t)}" for o, t in goal.items()) or "動かすものなし"

        # 3) 差分: 目標のうち、まだ目的の容器に無いものだけを実行する
        steps = [{"pick": o, "place": t} for o, t in goal.items() if location(o) != f"in {t}"]
        remain_txt = ", ".join(f"{jp_obj(s['pick'])}→{jp_tray(s['place'])}" for s in steps) or "なし"
        remain_parts = ["残り："]
        for s in steps:
            remain_parts += [("chip", color_of(s["pick"])), f"→ {jp_tray(s['place'])}　"]
        if not steps:
            remain_parts = ["残り：なし → 完了"]
        rec.set_overlay(goal=[(color_of(o), t) for o, t in goal.items()], phase=1, status=remain_parts)
        a["remaining"] = steps
        say(f"[{attempt}] goal: {goal_txt}\n    remaining: {remain_txt}")
        rec.hold(data, 1.2)
        if not steps:
            done = True
            break

        # 4) 実行
        arm.go_home()
        a["steps"] = []
        for i, s in enumerate(steps, 1):
            pick, place = s["pick"], s["place"]
            rec.set_overlay(phase=2, status=[("chip", color_of(pick)), f"を {jp_tray(place)}へ（{i} / {len(steps)}）"])
            rec_step = {"pick": pick, "place": place}
            if pick not in info.cubes:
                rec_step["result"] = "not found"
                goal.pop(pick, None)              # VLM の誤認識と判断して目標から外す
                say(f"    step {i}: {pick} は机上に無い（VLM の誤認識）→ 目標から外す")
            else:
                bid = model.body(pick).id
                p0 = scene.body_pos(model, data, pick)
                yaw = cube_yaw(data.xmat[bid])
                arm.pick(p0, yaw=yaw, grasp_dz=-0.004)
                lifted = scene.body_pos(model, data, pick)[2] > scene.CUBE_Z0 + 0.05
                if lifted:
                    arm.place(info.next_slot(place))
                ok = info.in_tray(scene.body_pos(model, data, pick), place)
                rec_step["result"] = "ok" if ok else ("dropped" if lifted else "grasp failed")
                say(f"    step {i}: {pick} → {place}: {rec_step['result']}")
            res_jp = {"ok": "成功", "not found": "見つからない", "dropped": "落とした", "grasp failed": "掴めず"}
            rec.set_overlay(status=[("chip", color_of(pick)),
                                    f"→ {jp_tray(place)}　{res_jp.get(rec_step['result'], rec_step['result'])}"])
            rec.hold(data, 0.4)
            a["steps"].append(rec_step)

        # 5) 外乱（デモ用）: 置けたブロックを 1 つ弾き出す。ロボットには知らせない
        if disturb and not disturbed:
            placed = [s for s in a["steps"] if s["result"] == "ok"]
            if placed:
                victim = placed[0]["pick"]
                arm.go_home()
                rec.set_overlay(phase=None, status=["外乱：", ("chip", color_of(victim)), "を机の上に戻す"])
                kick(model, data, victim)
                arm.settle(1.2)
                rec.hold(data, 1.0)
                disturbed = True
                a["disturbance"] = victim
                say(f"    外乱: {victim} を弾き出した → {location(victim)}")

        # 6) 検証（記録用）。次の周回で観測し直し、差分を取り直す
        failed = [s for s in a["steps"] if location(s["pick"]) != f"in {s['place']}"]
        a["verify"] = {"failed": failed}
        say(f"[{attempt}] verify: {'all ok' if not failed else 'failed: ' + ', '.join(s['pick'] for s in failed)}")
        if failed:
            verify_parts = [("chip", color_of(s["pick"])) for s in failed] + ["がトレイに無い → ① へ戻る"]
        else:
            verify_parts = ["すべて置けた → ① で確認する"]
        rec.set_overlay(phase=3, status=verify_parts)
        rec.hold(data, 0.8)

    arm.go_home()
    final = {c: location(c) for c in info.cubes}
    rec.set_overlay(phase=None, status=["完了：すべて目標の位置にある"] if done else ["打ち切り"])
    rec.hold(data, 2.0)
    path, dur = rec.save()
    rec.close()
    top_cam.close()
    log.update({"final": final, "success": done, "video": os.path.relpath(path, PROJ), "video_seconds": dur,
                "wall_seconds": time.time() - t_start})
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    say(f"[done] success={done} video={path} ({dur:.1f}s), wall {time.time() - t_start:.0f}s")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", "-i", default="暖色系のブロックを左のトレイに、寒色系のブロックを右のトレイに片付けて")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--speed", type=float, default=1.0, help="動画の倍速（2 なら 2 倍速。動画内に明記される）")
    ap.add_argument("--cubes", default="red,green,blue,yellow")
    ap.add_argument("--title", default=None)
    ap.add_argument("--disturb", action="store_true", help="1 回目の実行後にブロックを弾き出して回復させる")
    ap.add_argument("--text-scale", type=float, default=1.0, help="字幕と小窓の倍率")
    ap.add_argument("--layout", choices=["single", "triple"], default="triple",
                    help="triple: 上に字幕の帯、ISO/正面/ロボット視点の 3 画面（1920x1080, 24fps）")
    a = ap.parse_args()
    out = a.out or os.path.join(PROJ, "out", f"run_seed{a.seed}{'_disturb' if a.disturb else ''}.mp4")
    triple = a.layout == "triple"
    run(a.instruction, a.seed, out, speed=a.speed, cubes=tuple(a.cubes.split(",")), title=a.title,
        disturb=a.disturb, text_scale=(1.3 if triple and a.text_scale == 1.0 else a.text_scale),
        views=(triple_views() if triple else None), fps=(24 if triple else 30))


if __name__ == "__main__":
    main()
