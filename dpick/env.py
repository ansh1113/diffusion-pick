"""MuJoCo tabletop environment: Franka Panda (MuJoCo Menagerie), 3 colored cubes, a bowl.

Task: "put the red cube in the bowl".

Action (5-D, absolute end-effector targets, 10 Hz):
    [x, y, z, yaw, grip]   grip in [0, 1], 1 = close.
A damped-least-squares IK controller running at 250 Hz converts the target pose
into joint-position targets for the Panda's position servos.

Observation:
    front_rgb  (H, W, 3) uint8   fixed third-person camera
    wrist_rgb  (H, W, 3) uint8   camera mounted on the hand
    proprio    (12,)  float32    [ee_xyz(3), sin(yaw), cos(yaw), finger_width, q1..q6]
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "franka_emika_panda"

CUBE_HALF = 0.02
BOWL_R_IN = 0.075  # inner radius of the bowl wall
BOWL_WALL_H = 0.04
BOWL_BASE_H = 0.006
TABLE_TOP = 0.0
HOME_Q = np.array([0.0, 0.1, 0.0, -2.2, 0.0, 2.3, 0.785])
CONTROL_HZ = 10
PHYS_DT = 0.002
N_SUBSTEPS = int(round(1.0 / (CONTROL_HZ * PHYS_DT)))  # 50
IK_EVERY = 2

CUBE_COLORS = {
    "red": (0.85, 0.08, 0.08, 1.0),
    "green": (0.1, 0.7, 0.15, 1.0),
    "blue": (0.1, 0.25, 0.85, 1.0),
}
DISTRACTOR_SPECS = [  # (geom type, size, rgba)
    ("box", (0.018, 0.018, 0.018), (0.95, 0.85, 0.1, 1.0)),  # yellow cube
    ("cylinder", (0.02, 0.025, 0), (0.6, 0.2, 0.75, 1.0)),  # purple cylinder
    ("box", (0.03, 0.012, 0.012), (1.0, 0.5, 0.05, 1.0)),  # orange bar
    ("sphere", (0.02, 0, 0), (0.1, 0.8, 0.8, 1.0)),  # cyan ball
    ("box", (0.02, 0.02, 0.02), (0.55, 0.05, 0.05, 1.0)),  # dark-red cube (hard distractor)
]
WORKSPACE = dict(x=(0.38, 0.64), y=(-0.22, 0.22))


@dataclass
class EnvConfig:
    img_size: int = 64
    n_distractors: int = 0
    max_steps: int = 200
    # robustness perturbations (all off by default)
    light_perturb: bool = False
    camera_shift: bool = False
    viz_size: int = 0  # >0 additionally renders a high-res 'viz' frame for GIFs
    extra: dict = field(default_factory=dict)


def _quat_from_mat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def down_rot(yaw: float) -> np.ndarray:
    """Hand orientation pointing straight down with a given yaw (columns = hand axes in world)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def _look_at_xyaxes(pos, target, up=(0, 0, 1)):
    pos, target, up = map(np.asarray, (pos, target, up))
    fwd = target - pos
    fwd = fwd / np.linalg.norm(fwd)
    x = np.cross(fwd, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, fwd)
    return np.concatenate([x, y])


FRONT_CAM_POS = np.array([1.05, 0.0, 0.62])
FRONT_CAM_TARGET = np.array([0.47, 0.0, 0.02])


# "lite" uses decimated visual meshes (scripts/make_lite_meshes.py) -> ~6x faster software rendering.
# The dataset and the policy must use the same setting.
PANDA_VISUAL = os.environ.get("PANDA_VISUAL", "lite")


def build_model(n_distractors: int = 0) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(ASSET_DIR / "panda.xml"))
    if PANDA_VISUAL == "lite":
        lite = ASSET_DIR / "assets_lite"
        for mesh in spec.meshes:
            if mesh.file.endswith(".obj") and (lite / mesh.file).exists():
                mesh.file = str(lite / mesh.file)
    spec.option.timestep = PHYS_DT
    spec.option.noslip_iterations = 3
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 640
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.quality.shadowsize = 1024

    # gravity compensation on the arm so the position servos track IK targets accurately
    for b in spec.bodies:
        if b.name and (b.name.startswith("link") or b.name in ("hand", "left_finger", "right_finger")):
            b.gravcomp = 1.0

    hand = spec.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.005, 0, 0], rgba=[1, 0, 0, 0], group=4)
    # wrist camera: mounted on the hand, looking along the fingers
    hand.add_camera(name="wrist", pos=[0.065, 0.0, 0.02], xyaxes=[0, -1, 0, -0.97, 0, -0.25], fovy=75)

    wb = spec.worldbody
    tex = spec.add_texture(name="wood", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=[0.62, 0.48, 0.33],
                           rgb2=[0.5, 0.38, 0.25], width=64, height=64)
    spec.add_material(name="table", rgba=[0.66, 0.52, 0.38, 1], specular=0.1, shininess=0.1)
    spec.add_material(name="floor", rgba=[0.25, 0.27, 0.3, 1])
    del tex
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], pos=[0, 0, -0.75],
                material="floor")
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.6, 0.65, 0.02],
                pos=[0.45, 0, TABLE_TOP - 0.02], material="table", friction=[1.0, 0.005, 0.0001])
    for i, (lx, ly) in enumerate([(-0.1, -0.6), (-0.1, 0.6), (1.0, -0.6), (1.0, 0.6)]):
        wb.add_geom(name=f"leg{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, 0.365],
                    pos=[lx, ly, -0.405], material="table", contype=0, conaffinity=0)

    wb.add_light(name="key", pos=[0.5, -0.5, 1.6], dir=[-0.1, 0.4, -1], diffuse=[0.55, 0.55, 0.55],
                 specular=[0.2, 0.2, 0.2], castshadow=True)
    wb.add_light(name="fill", pos=[1.2, 0.6, 1.2], dir=[-0.6, -0.4, -1], diffuse=[0.25, 0.25, 0.25],
                 specular=[0, 0, 0], castshadow=False)

    wb.add_camera(name="front", pos=FRONT_CAM_POS.tolist(),
                  xyaxes=_look_at_xyaxes(FRONT_CAM_POS, FRONT_CAM_TARGET).tolist(), fovy=45)
    viz_pos = np.array([1.1, -0.75, 0.75])
    wb.add_camera(name="viz", pos=viz_pos.tolist(),
                  xyaxes=_look_at_xyaxes(viz_pos, np.array([0.4, 0.0, 0.1])).tolist(), fovy=50)

    # cubes
    for name, rgba in CUBE_COLORS.items():
        b = wb.add_body(name=f"{name}_cube", pos=[0.5, 0, CUBE_HALF])
        b.add_freejoint(name=f"{name}_cube_joint")
        b.add_geom(name=f"{name}_cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3,
                   rgba=list(rgba), mass=0.05, friction=[1.5, 0.01, 0.001], condim=4,
                   solref=[0.01, 1])

    # bowl: static body built from a base disk + ring of wall segments
    bowl = wb.add_body(name="bowl", pos=[0.5, 0.15, TABLE_TOP])
    bowl_rgba = [0.92, 0.9, 0.85, 1]
    bowl.add_geom(name="bowl_base", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[BOWL_R_IN + 0.008, BOWL_BASE_H / 2, 0], pos=[0, 0, BOWL_BASE_H / 2], rgba=bowl_rgba)
    n_seg = 20
    seg_half_len = (BOWL_R_IN + 0.006) * np.tan(np.pi / n_seg) + 0.002
    for k in range(n_seg):
        a = 2 * np.pi * k / n_seg
        r = BOWL_R_IN + 0.004
        bowl.add_geom(name=f"bowl_wall{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[0.004, seg_half_len, BOWL_WALL_H / 2],
                      pos=[r * np.cos(a), r * np.sin(a), BOWL_WALL_H / 2],
                      quat=[np.cos(a / 2), 0, 0, np.sin(a / 2)], rgba=[0.75, 0.55, 0.35, 1] if k % 2 else [0.8, 0.6, 0.4, 1])

    for i in range(n_distractors):
        gtype, size, rgba = DISTRACTOR_SPECS[i % len(DISTRACTOR_SPECS)]
        b = wb.add_body(name=f"distractor{i}", pos=[0.5, -0.3, 0.05])
        b.add_freejoint(name=f"distractor{i}_joint")
        gt = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
              "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}[gtype]
        b.add_geom(name=f"distractor{i}_geom", type=gt, size=list(size), rgba=list(rgba), mass=0.05,
                   friction=[1.0, 0.01, 0.001], condim=4)

    return spec.compile()


class PickPlaceEnv:
    """Put the red cube in the bowl."""

    def __init__(self, cfg: EnvConfig | None = None, seed: int | None = None):
        self.cfg = cfg or EnvConfig()
        self.model = build_model(self.cfg.n_distractors)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.rng_aux = np.random.default_rng(None if seed is None else seed + 7_777_777)
        m = self.model
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.arm_dadr = np.array([m.jnt_dofadr[m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.finger_qadr = [m.jnt_qposadr[m.joint(n).id] for n in ("finger_joint1", "finger_joint2")]
        self.tcp_id = m.site("tcp").id
        self.bowl_id = m.body("bowl").id
        self.cube_qadr = {c: m.jnt_qposadr[m.joint(f"{c}_cube_joint").id] for c in CUBE_COLORS}
        self.dis_qadr = [m.jnt_qposadr[m.joint(f"distractor{i}_joint").id] for i in range(self.cfg.n_distractors)]
        self.front_cam = m.camera("front").id
        self._nominal = dict(
            light_diffuse=m.light_diffuse.copy(), light_dir=m.light_dir.copy(), light_pos=m.light_pos.copy(),
            head_diffuse=m.vis.headlight.diffuse.copy(), head_ambient=m.vis.headlight.ambient.copy(),
            cam_pos=m.cam_pos.copy(), cam_quat=m.cam_quat.copy())
        self._renderer = None
        self._viz_renderer = None
        self._jac = np.zeros((6, m.nv))
        self.target = None
        self.t = 0

    # ------------------------------------------------------------------ rendering
    def _get_renderer(self):
        if self._renderer is None:
            s = self.cfg.img_size
            self._renderer = mujoco.Renderer(self.model, s, s)
        return self._renderer

    def render(self, camera: str, size: int | None = None) -> np.ndarray:
        if size is not None and size != self.cfg.img_size:
            if self._viz_renderer is None or self._viz_renderer.width != size:
                if self._viz_renderer is not None:
                    self._viz_renderer.close()
                self._viz_renderer = mujoco.Renderer(self.model, size, size)
            r = self._viz_renderer
        else:
            r = self._get_renderer()
        r.update_scene(self.data, camera=camera)
        return r.render().copy()

    def close(self):
        for r in (self._renderer, self._viz_renderer):
            if r is not None:
                r.close()
        self._renderer = self._viz_renderer = None

    # ------------------------------------------------------------------ state helpers
    def tcp_pos(self):
        return self.data.site_xpos[self.tcp_id].copy()

    def tcp_rot(self):
        return self.data.site_xmat[self.tcp_id].reshape(3, 3).copy()

    def tcp_yaw(self):
        R = self.tcp_rot()
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def finger_width(self):
        return float(self.data.qpos[self.finger_qadr[0]] + self.data.qpos[self.finger_qadr[1]])

    def cube_pos(self, color="red"):
        a = self.cube_qadr[color]
        return self.data.qpos[a:a + 3].copy()

    def cube_yaw(self, color="red"):
        """Yaw of the cube folded into [-pi/4, pi/4] (cube has 90-degree symmetry)."""
        a = self.cube_qadr[color]
        q = self.data.qpos[a + 3:a + 7]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        R = R.reshape(3, 3)
        # pick the body axis most horizontal & use its heading
        yaw = np.arctan2(R[1, 0], R[0, 0])
        return float((yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)

    def bowl_pos(self):
        return self.model.body_pos[self.bowl_id].copy()

    def proprio(self):
        p = self.tcp_pos()
        yaw = self.tcp_yaw()
        q = self.data.qpos[self.arm_qadr[:6]]
        return np.concatenate([p, [np.sin(yaw), np.cos(yaw), self.finger_width()], q]).astype(np.float32)

    def obs(self):
        o = dict(front_rgb=self.render("front"), wrist_rgb=self.render("wrist"), proprio=self.proprio())
        if self.cfg.viz_size:
            o["viz_rgb"] = self.render("viz", self.cfg.viz_size)
        return o

    def is_success(self):
        c = self.cube_pos("red")
        b = self.bowl_pos()
        in_xy = np.linalg.norm(c[:2] - b[:2]) < BOWL_R_IN - 0.01
        resting = c[2] < BOWL_BASE_H + CUBE_HALF + 0.015
        released = self.finger_width() > 0.05 or np.linalg.norm(self.tcp_pos() - c) > 0.04
        return bool(in_xy and resting and released)

    # ------------------------------------------------------------------ reset
    def _sample_layout(self):
        """Bowl + 3 cubes from the main RNG (identical across robustness conditions for a seed)."""
        rng = self.rng
        for _ in range(1000):
            bowl = np.array([rng.uniform(*WORKSPACE["x"]), rng.uniform(*WORKSPACE["y"])])
            pts = []
            ok = True
            for _k in range(3):
                for _t in range(200):
                    p = np.array([rng.uniform(*WORKSPACE["x"]), rng.uniform(*WORKSPACE["y"])])
                    if np.linalg.norm(p - bowl) < BOWL_R_IN + 0.06:
                        continue
                    if any(np.linalg.norm(p - q) < 0.075 for q in pts):
                        continue
                    pts.append(p)
                    break
                else:
                    ok = False
                    break
            if ok:
                return bowl, pts
        raise RuntimeError("could not sample layout")

    def _sample_distractors(self, bowl, pts):
        """Distractor positions from the auxiliary RNG, avoiding the bowl and the cubes."""
        rng, out = self.rng_aux, []
        x_rng = (WORKSPACE["x"][0] - 0.04, WORKSPACE["x"][1] + 0.04)
        y_rng = (WORKSPACE["y"][0] - 0.04, WORKSPACE["y"][1] + 0.04)
        for _ in range(self.cfg.n_distractors):
            for _t in range(500):
                p = np.array([rng.uniform(*x_rng), rng.uniform(*y_rng)])
                if np.linalg.norm(p - bowl) < BOWL_R_IN + 0.05:
                    continue
                if any(np.linalg.norm(p - q) < 0.065 for q in pts + out):
                    continue
                break
            out.append(p)
        return out

    def _apply_visual_perturbations(self):
        m, nom, rng = self.model, self._nominal, self.rng_aux
        m.light_diffuse[:] = nom["light_diffuse"]
        m.light_dir[:] = nom["light_dir"]
        m.light_pos[:] = nom["light_pos"]
        m.vis.headlight.diffuse[:] = nom["head_diffuse"]
        m.vis.headlight.ambient[:] = nom["head_ambient"]
        m.cam_pos[:] = nom["cam_pos"]
        m.cam_quat[:] = nom["cam_quat"]
        if self.cfg.light_perturb:
            scale = rng.uniform(0.35, 1.7)
            tint = rng.uniform(0.75, 1.25, size=3)
            m.light_diffuse[:] = np.clip(nom["light_diffuse"] * scale * tint, 0, 1)
            m.vis.headlight.diffuse[:] = np.clip(nom["head_diffuse"] * rng.uniform(0.3, 1.5) * tint, 0, 1)
            m.vis.headlight.ambient[:] = np.clip(nom["head_ambient"] * rng.uniform(0.3, 1.5), 0, 1)
            d = nom["light_dir"][0] + rng.normal(0, 0.5, 3)
            d[2] = -abs(d[2]) - 0.5
            m.light_dir[0] = d / np.linalg.norm(d)
            m.light_pos[0] = nom["light_pos"][0] + rng.uniform(-0.5, 0.5, 3) * [1, 1, 0]
        if self.cfg.camera_shift:
            mag = self.cfg.extra.get("cam_shift_m", 0.03)
            ang = self.cfg.extra.get("cam_shift_deg", 3.0)
            cid = self.front_cam
            m.cam_pos[cid] = nom["cam_pos"][cid] + rng.uniform(-mag, mag, 3)
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            dq = np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, axis, np.deg2rad(rng.uniform(-ang, ang)))
            out = np.zeros(4)
            mujoco.mju_mulQuat(out, dq, nom["cam_quat"][cid])
            m.cam_quat[cid] = out

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.rng_aux = np.random.default_rng(seed + 7_777_777)
        rng = self.rng
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self._apply_visual_perturbations()
        bowl, pts = self._sample_layout()
        m.body_pos[self.bowl_id][:2] = bowl
        colors = list(CUBE_COLORS)
        for c, p in zip(colors, pts[:3]):
            a = self.cube_qadr[c]
            yaw = rng.uniform(-np.pi / 4, np.pi / 4)
            d.qpos[a:a + 3] = [p[0], p[1], CUBE_HALF + 0.001]
            d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        for a, p in zip(self.dis_qadr, self._sample_distractors(bowl, pts)):
            yaw = self.rng_aux.uniform(-np.pi, np.pi)
            d.qpos[a:a + 3] = [p[0], p[1], 0.035]
            d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        q0 = HOME_Q + rng.normal(0, 0.06, 7)
        d.qpos[self.arm_qadr] = q0
        d.qpos[self.finger_qadr] = 0.04
        d.ctrl[:7] = q0
        d.ctrl[7] = 255
        mujoco.mj_forward(m, d)
        # settle with the arm holding its pose
        p, yaw = self.tcp_pos(), self.tcp_yaw()
        self.target = np.array([p[0], p[1], p[2], yaw, 0.0])
        for _ in range(int(0.3 / PHYS_DT)):
            mujoco.mj_step(m, d)
        self.t = 0
        return self.obs()

    # ------------------------------------------------------------------ control
    def _ik_step(self, pos_t, yaw_t):
        m, d = self.model, self.data
        jacp = self._jac[:3]
        jacr = self._jac[3:]
        mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp_id)
        J = self._jac[:, self.arm_dadr]
        err_p = pos_t - d.site_xpos[self.tcp_id]
        n = np.linalg.norm(err_p)
        if n > 0.04:
            err_p *= 0.04 / n
        R = d.site_xmat[self.tcp_id].reshape(3, 3)
        Rt = down_rot(yaw_t)
        Re = Rt @ R.T
        qe = _quat_from_mat(Re)
        if qe[0] < 0:
            qe = -qe
        err_r = 2.0 * qe[1:]
        n = np.linalg.norm(err_r)
        if n > 0.4:
            err_r *= 0.4 / n
        err = np.concatenate([err_p, err_r])
        lam = 1e-3
        JJt = J @ J.T + lam * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, err)
        # null-space posture regulation
        q = d.qpos[self.arm_qadr]
        Jpinv = J.T @ np.linalg.inv(JJt)
        dq += (np.eye(7) - Jpinv @ J) @ (0.2 * (HOME_Q - q))
        qd = q + dq
        lo = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 8)], 0]
        hi = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 8)], 1]
        d.ctrl[:7] = np.clip(qd, lo + 0.01, hi - 0.01)

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).copy()
        a[0] = np.clip(a[0], 0.2, 0.8)
        a[1] = np.clip(a[1], -0.4, 0.4)
        a[2] = np.clip(a[2], 0.0, 0.45)
        a[3] = np.clip(a[3], -np.pi / 2, np.pi / 2)
        a[4] = np.clip(a[4], 0.0, 1.0)
        self.target = a
        d = self.data
        d.ctrl[7] = 255.0 * (1.0 - a[4])
        for k in range(N_SUBSTEPS):
            if k % IK_EVERY == 0:
                self._ik_step(a[:3], a[3])
            mujoco.mj_step(self.model, d)
        self.t += 1
        success = self.is_success()
        done = success or self.t >= self.cfg.max_steps
        return self.obs(), float(success), done, dict(success=success)
