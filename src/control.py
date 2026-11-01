"""Panda の手先（TCP サイト）を目標位置へ動かす逆運動学と、pick / place の動作列。

逆運動学は減衰最小二乗法（DLS）。物理とは切り離した作業用 MjData で関節角を解き、
位置制御アクチュエータの目標値（ctrl）を滑らかに補間して物理を進める。
"""
import numpy as np
import mujoco

N_ARM = 7                      # 腕の関節数。指は dof 7, 8
GRIPPER_OPEN = 255.0           # actuator8 の ctrl。255 = 開口 0.08 m、0 = 閉
GRIPPER_CLOSED = 0.0
HOME_Q = np.array([0.0, -0.2, 0.0, -2.0, 0.0, 1.8, 0.785])


class Arm:
    def __init__(self, model, data, recorder=None, log=print):
        self.m, self.d = model, data
        self.rec = recorder
        self.log = log
        self.sid = model.site("tcp").id
        self.scratch = mujoco.MjData(model)
        self.lo = model.jnt_range[:N_ARM, 0]
        self.hi = model.jnt_range[:N_ARM, 1]
        self.dt = model.opt.timestep
        # ホーム姿勢の手先の向き（真下向き）を「置く・掴む」ときの基準にする
        self.go_home(instant=True)
        self.R_down = data.site_xmat[self.sid].reshape(3, 3).copy()

    # ---- 基本 ----
    def tcp_pos(self):
        return self.d.site_xpos[self.sid].copy()

    def go_home(self, instant=False):
        if instant:
            self.d.qpos[:N_ARM] = HOME_Q
            self.d.qvel[:N_ARM] = 0
            self.d.ctrl[:N_ARM] = HOME_Q
            self.d.ctrl[N_ARM] = GRIPPER_OPEN
            mujoco.mj_forward(self.m, self.d)
        else:
            self._interp_to(HOME_Q, seconds=1.5)

    def solve_ik(self, pos, R=None, q0=None, iters=300, tol=1e-4, damping=0.05, rot_weight=0.5):
        """目標位置 pos（と向き R）に手先を合わせる関節角を返す。(q, 残差)"""
        R = self.R_down if R is None else R
        q = np.array(self.d.ctrl[:N_ARM] if q0 is None else q0, dtype=float)
        s = self.scratch
        s.qpos[:] = self.d.qpos
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        err = None
        for _ in range(iters):
            s.qpos[:N_ARM] = q
            mujoco.mj_kinematics(self.m, s)
            mujoco.mj_comPos(self.m, s)
            p = s.site_xpos[self.sid]
            Rc = s.site_xmat[self.sid].reshape(3, 3)
            e_pos = pos - p
            e_rot = 0.5 * (np.cross(Rc[:, 0], R[:, 0]) + np.cross(Rc[:, 1], R[:, 1]) + np.cross(Rc[:, 2], R[:, 2]))
            err = np.concatenate([e_pos, rot_weight * e_rot])
            if np.linalg.norm(e_pos) < tol and np.linalg.norm(e_rot) < 1e-3:
                break
            mujoco.mj_jacSite(self.m, s, jacp, jacr, self.sid)
            J = np.vstack([jacp[:, :N_ARM], rot_weight * jacr[:, :N_ARM]])
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), err)
            step = np.linalg.norm(dq)
            if step > 0.2:
                dq *= 0.2 / step
            q = np.clip(q + dq, self.lo + 0.02, self.hi - 0.02)
        return q, float(np.linalg.norm(err[:3]))

    def _interp_to(self, q_target, seconds, gripper=None):
        """ctrl を現在値から q_target へ滑らかに補間しながら物理を進める。"""
        n = max(int(seconds / self.dt), 1)
        q_start = self.d.ctrl[:N_ARM].copy()
        for i in range(n):
            a = (i + 1) / n
            a = a * a * (3 - 2 * a)         # smoothstep
            self.d.ctrl[:N_ARM] = q_start + a * (q_target - q_start)
            if gripper is not None:
                self.d.ctrl[N_ARM] = gripper
            self.step()
        self.settle(0.15)

    def step(self):
        mujoco.mj_step(self.m, self.d)
        if self.rec is not None:
            self.rec.tick(self.d)

    def settle(self, seconds):
        for _ in range(int(seconds / self.dt)):
            self.step()

    def move_q(self, q, seconds=1.0):
        """関節角を直接指定して動かす（観測姿勢など）。"""
        self._interp_to(np.asarray(q, dtype=float), seconds)

    def move_to(self, pos, seconds=1.0, yaw=0.0):
        """手先を真下向きのまま pos へ。yaw は手首の z 回転（ブロックの向きに合わせる）。"""
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        q, res = self.solve_ik(np.asarray(pos, dtype=float), R=Rz @ self.R_down)
        if res > 0.01:
            self.log(f"  [IK] 残差 {res*100:.1f} cm @ {np.round(pos, 3)}")
        self._interp_to(q, seconds)
        return res

    def set_gripper(self, value, seconds=0.6):
        n = int(seconds / self.dt)
        for _ in range(n):
            self.d.ctrl[N_ARM] = value
            self.step()

    # ---- 動作列 ----
    def pick(self, obj_pos, yaw=0.0, hover=0.12, grasp_dz=0.0):
        p = np.asarray(obj_pos, dtype=float)
        self.set_gripper(GRIPPER_OPEN, 0.3)
        self.move_to(p + [0, 0, hover], 1.0, yaw)
        self.move_to(p + [0, 0, grasp_dz], 0.8, yaw)
        self.set_gripper(GRIPPER_CLOSED, 0.7)
        self.move_to(p + [0, 0, hover], 0.8, yaw)

    def place(self, target_pos, hover=0.12, release_dz=0.03):
        p = np.asarray(target_pos, dtype=float)
        self.move_to(p + [0, 0, hover], 1.2)
        self.move_to(p + [0, 0, release_dz], 0.8)
        self.set_gripper(GRIPPER_OPEN, 0.5)
        self.move_to(p + [0, 0, hover], 0.7)


def cube_yaw(xmat):
    """立方体の向き（z 軸回りの回転）を、対称性を使って [-45°, 45°] に畳んで返す。"""
    R = np.asarray(xmat).reshape(3, 3)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return float(yaw)
