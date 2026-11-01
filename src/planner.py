"""言語指示を、ローカル LLM（ollama）で「どのブロックをどのトレイへ」という手順列に変換する。

AI の使いどころはここ。曖昧な指示（「暖色系を左に」「原色だけ片付けて」など）の解釈を LLM に任せ、
実行と検証は物理シミュレーション側が受け持つ。失敗時はその状況を伝えて計画し直させる（閉ループ）。
"""
import json
import os
import time

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("PAI_MODEL", "gemma3:4b-it-q4_K_M")
# ollama は num_ctx 未指定だと VRAM 量から自動で決める（親の手順書）。必ず明示する。
NUM_CTX = int(os.environ.get("PAI_NUM_CTX", "4096"))

SYSTEM_PROMPT = """You are the task planner of a tabletop robot arm.
The robot can only do one kind of action: pick one object and place it into one container.
You will be given the objects the camera currently sees (with where each one is now), the containers,
and a natural-language instruction (possibly in Japanese). Decide which objects must be moved to which
container so that the instruction becomes satisfied.
Rules:
- Never move an object that is already in the container the instruction wants it in.
- Never move objects the instruction does not ask to move.
- If the instruction is already satisfied, return an empty steps list.
Answer in JSON only, with this exact shape:
{"reasoning": "<one or two short English sentences>", "steps": [{"pick": "<object>", "place": "<container>"}, ...]}
Use only the object and container names given."""


class PlannerError(RuntimeError):
    pass


def check_server():
    """起動時に ollama とモデルの実在を確かめる。無ければ止める（黙って別物を使わない）。"""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise PlannerError(f"ollama に接続できない ({OLLAMA_URL}): {e}. ollama serve で起動する") from e
    names = {m["name"] for m in r.json().get("models", [])}
    if MODEL not in names:
        raise PlannerError(f"モデル {MODEL} が ollama に無い。入っているもの: {sorted(names)}")


def _chat(messages, images=None):
    if images:
        messages = [dict(m) for m in messages]
        messages[-1]["images"] = images
    payload = {
        "model": MODEL,
        "stream": False,
        "format": "json",
        "options": {"num_ctx": NUM_CTX, "temperature": 0},
        "messages": messages,
    }
    t0 = time.time()
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
    r.raise_for_status()
    body = r.json()
    content = body["message"]["content"]
    stats = {
        "wall_s": time.time() - t0,
        "prompt_tokens": body.get("prompt_eval_count"),
        "output_tokens": body.get("eval_count"),
    }
    return content, stats


def _parse_steps(content, objects, containers):
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise PlannerError(f"LLM の出力が JSON でない: {content[:200]}") from e
    steps = []
    dropped = []
    for s in data.get("steps", []) or []:
        pick, place = s.get("pick"), s.get("place")
        if pick in objects and place in containers:
            steps.append({"pick": pick, "place": place})
        else:
            dropped.append(s)
    return {"reasoning": data.get("reasoning", ""), "steps": steps, "dropped": dropped, "raw": content}


def plan(instruction, state, containers, feedback=None):
    """state は {物体名: 今の場所}（場所は "on the table" / "in tray_left" など）。
    containers は名前のリスト。feedback は前回の実行結果（再計画のとき）。

    戻り値: {"reasoning", "steps": [{"pick","place"}...], "dropped", "raw", "stats", "prompt"}
    """
    objects = list(state)
    if objects:
        obj_lines = "\n".join(f"  - {o}: {loc}" for o, loc in state.items())
    else:
        obj_lines = "  (the camera sees no cubes)"
    user = (
        f"Objects seen by the camera, with their current location:\n{obj_lines}\n"
        f"Containers: {', '.join(containers)}\n"
        f"Instruction: {instruction}"
    )
    if feedback:
        user += f"\n\nResult of the previous attempt: {feedback}\nPlan only what still needs to be done."
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    content, stats = _chat(messages)
    out = _parse_steps(content, objects, containers)
    out["stats"] = stats
    out["prompt"] = user
    return out


if __name__ == "__main__":
    import sys

    check_server()
    instr = sys.argv[1] if len(sys.argv) > 1 else "暖色系のブロックを左のトレイに、寒色系を右のトレイに片付けて"
    state = {c: "on the table" for c in ["red_cube", "green_cube", "blue_cube", "yellow_cube"]}
    res = plan(instr, state, ["tray_left", "tray_right"])
    print(json.dumps(res, ensure_ascii=False, indent=2))
