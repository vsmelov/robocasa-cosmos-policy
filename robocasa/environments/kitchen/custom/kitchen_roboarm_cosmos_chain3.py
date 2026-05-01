"""Kitchen-roboarm Cosmos multi-stage envs (one MuJoCo episode, no reset between stages).

- ``PnPRoboarmCosmosChain3``: stove PnP → counter→microwave → start button.
- ``PnPRoboarmCosmosChain2MicrowaveCloseOn``: close microwave door → start button.
- ``PnPRoboarmCosmosChain2DrawerOpenClose``: open top drawer → close it.
- ``PnPRoboarmCosmosChain4MicrowaveCloseOnOffOpen``: close MW → arm home → on → arm home → off → arm home → open door.
- ``PnPRoboarmCosmosChain6PotatoMwPlate``: counter→MW (potato) → close → on → off → open → MW→counter (plate).

Each chain class defines ``CHAIN_STAGE_HORIZON_NAMES`` for horizon / T5 lookup in eval.

Canonical copy lives in the ``vsmelov/robocasa-cosmos-policy`` fork (git submodule
``external/robocasa-cosmos-policy`` in ``kitchen-roboarm``); Docker mounts that tree at ``/opt/robocasa-cosmos-policy``.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from robocasa.environments.kitchen.kitchen import *
from robocasa.environments.kitchen.single_stage.kitchen_drawer import OpenDrawer
from robocasa.environments.kitchen.single_stage.kitchen_pnp import PnP

# Turn-off stage (chain4): success if gripper touches stop OR min distance (eef / finger geoms / sites) ≤ this (m).
CHAIN4_TURNOFF_PROXIMITY_M = 0.01

_FINGER_GEOM_KEYS = ("pad", "finger", "fingertip", "tip", "touch")
_FINGER_SITE_KEYS = ("pad", "finger", "fingertip", "tip", "touch", "grip")


def _distance_eef_to_microwave_button(env, microwave, button: str) -> float:
    assert button in ("start_button", "stop_button")
    gid = env.sim.model.geom_name2id("{}{}".format(microwave.naming_prefix, button))
    btn = env.sim.data.geom_xpos[gid]
    eef = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]
    return float(np.linalg.norm(eef - btn))


def _min_finger_geom_dist_to_mw_button(env, microwave, button: str) -> tuple[float, int]:
    """Min |geom_xpos - button_center| for robot geoms under naming_prefix that look like pads/fingers."""
    robot = env.robots[0]
    pf = robot.robot_model.naming_prefix
    model, data = env.sim.model, env.sim.data
    gid = model.geom_name2id("{}{}".format(microwave.naming_prefix, button))
    btn = data.geom_xpos[gid]
    best = float("inf")
    n = 0
    for j in range(model.ngeom):
        name = model.geom_id2name(j)
        if not name or not name.startswith(pf):
            continue
        tail = name[len(pf) :].lower()
        if not any(k in tail for k in _FINGER_GEOM_KEYS):
            continue
        n += 1
        best = min(best, float(np.linalg.norm(data.geom_xpos[j] - btn)))
    return best, n


def _min_finger_site_dist_to_mw_button(env, microwave, button: str) -> tuple[float, int]:
    """Same for sites (some grippers expose pads as sites only)."""
    robot = env.robots[0]
    pf = robot.robot_model.naming_prefix
    model, data = env.sim.model, env.sim.data
    gid = model.geom_name2id("{}{}".format(microwave.naming_prefix, button))
    btn = data.geom_xpos[gid]
    best = float("inf")
    n = 0
    for sid in range(model.nsite):
        name = model.site_id2name(sid)
        if not name or not name.startswith(pf):
            continue
        tail = name[len(pf) :].lower()
        if not any(k in tail for k in _FINGER_SITE_KEYS):
            continue
        n += 1
        best = min(best, float(np.linalg.norm(data.site_xpos[sid] - btn)))
    return best, n


def chain4_turnoff_best_distance_to_stop(env) -> float:
    """Smallest proxy distance used for success (eef vs finger-like geoms/sites)."""
    mw = env.microwave
    d_eef = _distance_eef_to_microwave_button(env, mw, "stop_button")
    d_g, ng = _min_finger_geom_dist_to_mw_button(env, mw, "stop_button")
    d_s, ns = _min_finger_site_dist_to_mw_button(env, mw, "stop_button")
    parts = [d_eef]
    if ng > 0:
        parts.append(d_g)
    if ns > 0:
        parts.append(d_s)
    return float(min(parts))


def chain4_turnoff_debug_snapshot(env) -> dict:
    """Read-only metrics for stage-2 logging (one line per env.step when debug fp is open)."""
    mw = env.microwave
    robot = env.robots[0]
    d_eef = _distance_eef_to_microwave_button(env, mw, "stop_button")
    d_geom, n_geom = _min_finger_geom_dist_to_mw_button(env, mw, "stop_button")
    d_site, n_site = _min_finger_site_dist_to_mw_button(env, mw, "stop_button")
    d_best = chain4_turnoff_best_distance_to_stop(env)
    door = {k: float(v) for k, v in mw.get_door_state(env).items()}
    turned_on = bool(mw.get_state()["turned_on"])
    far_stop = bool(mw.gripper_button_far(env, button="stop_button"))
    stop_c = bool(env.check_contact(robot.gripper["right"], f"{mw.name}_stop_button"))
    start_c = bool(env.check_contact(robot.gripper["right"], f"{mw.name}_start_button"))
    return {
        "chain_stage": int(getattr(env, "chain_stage")),
        "distance_best_to_stop_m": d_best,
        "distance_eef_to_stop_m": d_eef,
        "distance_min_finger_geom_to_stop_m": float(d_geom) if n_geom else None,
        "distance_min_finger_site_to_stop_m": float(d_site) if n_site else None,
        "finger_geom_candidates": int(n_geom),
        "finger_site_candidates": int(n_site),
        "door_state": door,
        "gripper_far_stop_th0_15m": far_stop,
        "proximity_lte_1cm": d_best <= CHAIN4_TURNOFF_PROXIMITY_M,
        "start_button_contact": start_c,
        "stop_button_contact": stop_c,
        "turned_on": turned_on,
    }


def write_chain4_turnoff_stage_debug_log(attempt_dir: Path, env, episode_success: bool) -> None:
    """Append episode summary after per-step JSONL (chain4 stage 2 only)."""
    if type(env).__name__ != "PnPRoboarmCosmosChain4MicrowaveCloseOnOffOpen":
        return
    if int(getattr(env, "chain_stage")) != 2:
        return
    path = Path(attempt_dir) / "turnoff_stage_debug.log"
    snap = chain4_turnoff_debug_snapshot(env)
    snap["episode_reported_success"] = bool(episode_success)
    snap["min_distance_eef_seen"] = getattr(env, "_chain4_turnoff_min_eef", None)
    snap["min_distance_finger_geom_seen"] = getattr(env, "_chain4_turnoff_min_finger_geom", None)
    snap["min_distance_best_seen"] = getattr(env, "_chain4_turnoff_min_best", None)
    snap["stop_contact_ever"] = bool(getattr(env, "_chain4_turnoff_contact_ever", False))
    snap["utc"] = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as f:
        f.write("=== episode_summary (after last step) ===\n")
        f.write(json.dumps(snap, indent=2, sort_keys=True))
        f.write("\n")


def _chain_flat_gripper_qpos_indices(robot):
    g = getattr(robot, "_ref_gripper_joint_pos_indexes", None)
    if g is None:
        return np.array([], dtype=int)
    if isinstance(g, dict):
        out = []
        for v in g.values():
            out.extend(int(x) for x in v)
        return np.asarray(out, dtype=int)
    return np.asarray([int(x) for x in g], dtype=int)


def _chain_flat_gripper_vel_indices(robot):
    g = getattr(robot, "_ref_gripper_joint_vel_indexes", None)
    if g is None:
        return np.array([], dtype=int)
    if isinstance(g, dict):
        out = []
        for v in g.values():
            out.extend(int(x) for x in v)
        return np.asarray(out, dtype=int)
    return np.asarray([int(x) for x in g], dtype=int)


def _snapshot_chain_init_arm_gripper(env):
    """After env.reset layout: remember arm+gripper qpos so we can hard-restore before the next stage."""
    robot = env.robots[0]
    arm_i = np.asarray(robot._ref_arm_joint_pos_indexes, dtype=int)
    grip_i = _chain_flat_gripper_qpos_indices(robot)
    env._chain_init_arm_qpos = np.array(env.sim.data.qpos[arm_i], dtype=np.float64, copy=True)
    env._chain_init_gripper_qpos = (
        np.array(env.sim.data.qpos[grip_i], dtype=np.float64, copy=True) if grip_i.size else None
    )


def _restore_chain_init_arm_gripper(env):
    """Hard-set arm+gripper to values captured at episode init (distribution shift hack for button press)."""
    if getattr(env, "_chain_init_arm_qpos", None) is None:
        return
    robot = env.robots[0]
    sim = env.sim
    arm_i = np.asarray(robot._ref_arm_joint_pos_indexes, dtype=int)
    grip_i = _chain_flat_gripper_qpos_indices(robot)
    sim.data.qpos[arm_i] = env._chain_init_arm_qpos
    if grip_i.size and env._chain_init_gripper_qpos is not None:
        sim.data.qpos[grip_i] = env._chain_init_gripper_qpos
    arm_vi = np.asarray(getattr(robot, "_ref_arm_joint_vel_indexes", []), dtype=int)
    if arm_vi.size:
        sim.data.qvel[arm_vi] = 0.0
    grip_vi = _chain_flat_gripper_vel_indices(robot)
    if grip_vi.size:
        sim.data.qvel[grip_vi] = 0.0
    sim.forward()


# Default duration for linear joint blend to episode-init arm pose between chain stages (seconds).
CHAIN_ARM_HOME_DURATION_S = 1.0


def _restore_chain_init_arm_gripper_smooth(env, duration_s=None):
    """
    Linearly interpolate arm (+ gripper) qpos to the snapshot from ``_snapshot_chain_init_arm_gripper``
    over ``duration_s`` seconds at ``env.control_freq`` substeps (no OSC ``step`` — only ``sim.forward``).

    Optional ``env._chain_arm_home_capture_cb`` callable ``f(env)`` invoked after each substep forward
    (e.g. Cosmos appends camera frames for run-level video).
    """
    dur = float(CHAIN_ARM_HOME_DURATION_S if duration_s is None else duration_s)
    if getattr(env, "_chain_init_arm_qpos", None) is None:
        return
    if dur <= 0.0:
        _restore_chain_init_arm_gripper(env)
        return
    robot = env.robots[0]
    sim = env.sim
    cf = float(getattr(env, "control_freq", 20) or 20)
    n = max(2, int(round(dur * cf)))
    arm_i = np.asarray(robot._ref_arm_joint_pos_indexes, dtype=int)
    grip_i = _chain_flat_gripper_qpos_indices(robot)
    start_arm = np.array(sim.data.qpos[arm_i], dtype=np.float64, copy=True)
    tgt_arm = np.asarray(env._chain_init_arm_qpos, dtype=np.float64)
    start_grip = (
        np.array(sim.data.qpos[grip_i], dtype=np.float64, copy=True) if grip_i.size else None
    )
    tgt_grip = (
        np.asarray(env._chain_init_gripper_qpos, dtype=np.float64)
        if (grip_i.size and env._chain_init_gripper_qpos is not None)
        else None
    )
    arm_vi = np.asarray(getattr(robot, "_ref_arm_joint_vel_indexes", []), dtype=int)
    grip_vi = _chain_flat_gripper_vel_indices(robot)
    cap = getattr(env, "_chain_arm_home_capture_cb", None)
    for k in range(1, n + 1):
        alpha = k / float(n)
        sim.data.qpos[arm_i] = (1.0 - alpha) * start_arm + alpha * tgt_arm
        if grip_i.size and tgt_grip is not None and start_grip is not None:
            sim.data.qpos[grip_i] = (1.0 - alpha) * start_grip + alpha * tgt_grip
        if arm_vi.size:
            sim.data.qvel[arm_vi] = 0.0
        if grip_vi.size:
            sim.data.qvel[grip_vi] = 0.0
        sim.forward()
        if cap is not None:
            try:
                cap(env)
            except Exception:
                pass


class PnPRoboarmCosmosChain3(PnP):
    """
    Stove → counter (PnP pan), then counter → microwave, then turn on microwave,
    in one simulation without reset between stages.
    """

    CHAIN_STAGE_HORIZON_NAMES = ("PnPStoveToCounter", "PnPCounterToMicrowave", "TurnOnMicrowave")
    # After completing stage 1 (object in MW), snap arm back to post-reset pose before «press start».
    CHAIN_RESET_ARM_AFTER_STAGES = (1,)
    EXCLUDE_LAYOUTS = [8]

    def __init__(self, obj_groups="food", *args, **kwargs):
        super().__init__(obj_groups=obj_groups, *args, **kwargs)
        self._chain_stage = 0
        self._chain_init_arm_qpos = None
        self._chain_init_gripper_qpos = None

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter_stove = self.register_fixture_ref(
            "counter_stove", dict(id=FixtureType.COUNTER, ref=self.stove, size=[0.30, 0.40])
        )
        self.microwave = self.register_fixture_ref("microwave", dict(id=FixtureType.MICROWAVE))
        self.counter_mw = self.register_fixture_ref(
            "counter_mw", dict(id=FixtureType.COUNTER, ref=self.microwave)
        )
        self.init_robot_base_pos = self.stove

    def _reset_internal(self):
        super()._reset_internal()
        self._chain_stage = 0
        self.microwave.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)
        _snapshot_chain_init_arm_gripper(self)

    def advance_chain_stage(self):
        """Call after stage success; no simulation reset."""
        last = len(type(self).CHAIN_STAGE_HORIZON_NAMES) - 1
        if self._chain_stage < last:
            completed = self._chain_stage
            self._chain_stage += 1
            if completed in getattr(type(self), "CHAIN_RESET_ARM_AFTER_STAGES", ()):
                _restore_chain_init_arm_gripper_smooth(self)

    @property
    def chain_stage(self) -> int:
        return self._chain_stage

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self._chain_stage == 0:
            obj_lang = self.get_obj_lang()
            obj_cont_lang = self.get_obj_lang(obj_name="obj_container")
            cont_lang, preposition = self.get_obj_lang(
                obj_name="container", get_preposition=True
            )
            ep_meta["lang"] = (
                f"pick the {obj_lang} from the {obj_cont_lang} and place it {preposition} the {cont_lang}"
            )
        elif self._chain_stage == 1:
            obj_lang = self.get_obj_lang()
            ep_meta["lang"] = f"pick the {obj_lang} from the counter and place it in the microwave"
        else:
            ep_meta["lang"] = "press the start button on the microwave"
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                cookable=True,
                microwavable=True,
                max_size=(0.15, 0.15, None),
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    size=(0.02, 0.02),
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                    try_to_place_in="pan",
                ),
            )
        )
        cfgs.append(
            dict(
                name="container",
                obj_groups=("plate", "bowl"),
                placement=dict(
                    fixture=self.counter_stove,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self):
        if self._chain_stage == 0:
            return OU.check_obj_in_receptacle(self, "obj", "container", th=0.07) and OU.gripper_obj_far(
                self
            )
        if self._chain_stage == 1:
            return OU.obj_inside_of(self, "obj", self.microwave) and OU.gripper_obj_far(self)
        turned_on = self.microwave.get_state()["turned_on"]
        far = self.microwave.gripper_button_far(self, button="start_button")
        return bool(turned_on and far)


class PnPRoboarmCosmosChain2MicrowaveCloseOn(Kitchen):
    """Microwave door starts open; stage 0 = close door, stage 1 = press start (same pose as atomic)."""

    CHAIN_STAGE_HORIZON_NAMES = ("CloseSingleDoor", "TurnOnMicrowave")
    CHAIN_RESET_ARM_AFTER_STAGES = (0,)
    EXCLUDE_LAYOUTS = [8]

    def __init__(self, *args, **kwargs):
        # Potato (and many microwavable foods) ship MJCFs under aigen_objs; default Kitchen uses objaverse only,
        # which can leave zero candidates and break rng.choice(probabilities) with NaN.
        kwargs.setdefault("obj_registries", ("aigen", "objaverse"))
        super().__init__(*args, **kwargs)
        self._chain_stage = 0
        self._chain_init_arm_qpos = None
        self._chain_init_gripper_qpos = None

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref("microwave", dict(id=FixtureType.MICROWAVE))
        self.init_robot_base_pos = self.microwave

    def _reset_internal(self):
        super()._reset_internal()
        self._chain_stage = 0
        self.microwave.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)
        _snapshot_chain_init_arm_gripper(self)

    def advance_chain_stage(self):
        last = len(type(self).CHAIN_STAGE_HORIZON_NAMES) - 1
        if self._chain_stage < last:
            completed = self._chain_stage
            self._chain_stage += 1
            if completed in getattr(type(self), "CHAIN_RESET_ARM_AFTER_STAGES", ()):
                _restore_chain_init_arm_gripper_smooth(self)

    @property
    def chain_stage(self) -> int:
        return self._chain_stage

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self._chain_stage == 0:
            ep_meta["lang"] = "close the microwave door"
        else:
            ep_meta["lang"] = "press the start button on the microwave"
        return ep_meta

    def _get_obj_cfgs(self):
        # Align with Cosmos stove benchmarks (potato); ``obj_groups=all`` pulls unrelated props (e.g. candles).
        return [
            dict(
                name="obj",
                obj_groups=("potato",),
                heatable=True,
                microwavable=True,
                placement=dict(
                    fixture=self.microwave,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                    try_to_place_in="container",
                ),
            )
        ]

    def _check_success(self):
        if self._chain_stage == 0:
            for joint_p in self.microwave.get_door_state(env=self).values():
                if joint_p > 0.05:
                    return False
            return True
        turned_on = self.microwave.get_state()["turned_on"]
        far = self.microwave.gripper_button_far(self, button="start_button")
        return bool(turned_on and far)


class PnPRoboarmCosmosChain4MicrowaveCloseOnOffOpen(Kitchen):
    """
    One episode: close door → (arm home) → start → (arm home) → stop → (arm home) → open door.
    Arm+gripper ``qpos`` snapshotted after ``reset``; hard-restored after stages 0–2 before the next instruction.
    """

    CHAIN_STAGE_HORIZON_NAMES = (
        "CloseSingleDoor",
        "TurnOnMicrowave",
        "TurnOffMicrowave",
        "OpenSingleDoor",
    )
    CHAIN_RESET_ARM_AFTER_STAGES = (0, 1, 2)
    EXCLUDE_LAYOUTS = [8]

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("obj_registries", ("aigen", "objaverse"))
        super().__init__(*args, **kwargs)
        self._chain_stage = 0
        self._chain_init_arm_qpos = None
        self._chain_init_gripper_qpos = None

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref("microwave", dict(id=FixtureType.MICROWAVE))
        self.init_robot_base_pos = self.microwave

    def _reset_internal(self):
        super()._reset_internal()
        self._chain_stage = 0
        self.microwave.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)
        _snapshot_chain_init_arm_gripper(self)

    def advance_chain_stage(self):
        last = len(type(self).CHAIN_STAGE_HORIZON_NAMES) - 1
        if self._chain_stage < last:
            completed = self._chain_stage
            self._chain_stage += 1
            if completed in getattr(type(self), "CHAIN_RESET_ARM_AFTER_STAGES", ()):
                _restore_chain_init_arm_gripper_smooth(self)
                # Arm motion can crack the door; open door forces off in Microwave.update_state.
                # Re-seal before start/stop so door_state does not fight the turn-off stage.
                if self._chain_stage in (1, 2):
                    self.microwave.set_door_state(min=0.0, max=0.0, env=self, rng=self.rng)
                    self.sim.forward()

    @property
    def chain_stage(self) -> int:
        return self._chain_stage

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self._chain_stage == 0:
            ep_meta["lang"] = "close the microwave door"
        elif self._chain_stage == 1:
            ep_meta["lang"] = "press the start button on the microwave"
        elif self._chain_stage == 2:
            ep_meta["lang"] = "press the stop button on the microwave"
        else:
            ep_meta["lang"] = "open the microwave door"
        return ep_meta

    def step(self, action):
        obs, reward, done, info = super().step(action)
        fp = getattr(self, "_chain4_turnoff_debug_fp", None)
        if fp is not None and self._chain_stage == 2:
            row = chain4_turnoff_debug_snapshot(self)
            fp.write(json.dumps(row, sort_keys=True) + "\n")
            self._chain4_turnoff_min_eef = min(
                getattr(self, "_chain4_turnoff_min_eef", float("inf")), float(row["distance_eef_to_stop_m"])
            )
            mg = row.get("distance_min_finger_geom_to_stop_m")
            if mg is not None:
                self._chain4_turnoff_min_finger_geom = min(
                    getattr(self, "_chain4_turnoff_min_finger_geom", float("inf")), float(mg)
                )
            self._chain4_turnoff_min_best = min(
                getattr(self, "_chain4_turnoff_min_best", float("inf")), float(row["distance_best_to_stop_m"])
            )
            if row["stop_button_contact"]:
                self._chain4_turnoff_contact_ever = True
        return obs, reward, done, info

    def _get_obj_cfgs(self):
        return [
            dict(
                name="obj",
                obj_groups=("potato",),
                heatable=True,
                microwavable=True,
                placement=dict(
                    fixture=self.microwave,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                    try_to_place_in="container",
                ),
            )
        ]

    def _check_success(self):
        door_state = self.microwave.get_door_state(env=self)
        if self._chain_stage == 0:
            for joint_p in door_state.values():
                if joint_p > 0.05:
                    return False
            return True
        if self._chain_stage == 1:
            turned_on = self.microwave.get_state()["turned_on"]
            far = self.microwave.gripper_button_far(self, button="start_button")
            return bool(turned_on and far)
        if self._chain_stage == 2:
            mw = self.microwave
            robot = self.robots[0]
            d_best = chain4_turnoff_best_distance_to_stop(self)
            contact = bool(self.check_contact(robot.gripper["right"], f"{mw.name}_stop_button"))
            if contact or d_best <= CHAIN4_TURNOFF_PROXIMITY_M:
                mw._turned_on = False
                return True
            return False
        for joint_p in door_state.values():
            if joint_p < 0.90:
                return False
        return True


class PnPRoboarmCosmosChain6PotatoMwPlate(Kitchen):
    """
    Counter → microwave (potato + plate in MW), close, on, off, open, then pick to counter plate.
    Same uninterrupted sim as other chain envs; arm returns to post-reset pose smoothly between stages.
    """

    CHAIN_STAGE_HORIZON_NAMES = (
        "PnPCounterToMicrowave",
        "CloseSingleDoor",
        "TurnOnMicrowave",
        "TurnOffMicrowave",
        "OpenSingleDoor",
        "PnPMicrowaveToCounter",
    )
    CHAIN_RESET_ARM_AFTER_STAGES = (0, 1, 2, 3, 4)
    EXCLUDE_LAYOUTS = [8]

    def __init__(self, obj_groups=("potato",), exclude_obj_groups=None, *args, **kwargs):
        self.obj_groups = obj_groups
        self.exclude_obj_groups = exclude_obj_groups
        kwargs.setdefault("obj_registries", ("aigen", "objaverse"))
        super().__init__(*args, **kwargs)
        self._chain_stage = 0
        self._chain_init_arm_qpos = None
        self._chain_init_gripper_qpos = None

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.microwave = self.register_fixture_ref(
            "microwave",
            dict(id=FixtureType.MICROWAVE),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.distr_counter = self.register_fixture_ref(
            "distr_counter",
            dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.init_robot_base_pos = self.microwave

    def _reset_internal(self):
        super()._reset_internal()
        self._chain_stage = 0
        self.microwave.set_door_state(min=0.90, max=1.0, env=self, rng=self.rng)
        _snapshot_chain_init_arm_gripper(self)

    def advance_chain_stage(self):
        last = len(type(self).CHAIN_STAGE_HORIZON_NAMES) - 1
        if self._chain_stage < last:
            completed = self._chain_stage
            self._chain_stage += 1
            if completed in getattr(type(self), "CHAIN_RESET_ARM_AFTER_STAGES", ()):
                _restore_chain_init_arm_gripper_smooth(self)
                if self._chain_stage in (2, 3):
                    self.microwave.set_door_state(min=0.0, max=0.0, env=self, rng=self.rng)
                    self.sim.forward()

    @property
    def chain_stage(self) -> int:
        return self._chain_stage

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self._chain_stage == 0:
            obj_lang = self.get_obj_lang()
            ep_meta["lang"] = f"pick the {obj_lang} from the counter and place it in the microwave"
        elif self._chain_stage == 1:
            ep_meta["lang"] = "close the microwave door"
        elif self._chain_stage == 2:
            ep_meta["lang"] = "press the start button on the microwave"
        elif self._chain_stage == 3:
            ep_meta["lang"] = "press the stop button on the microwave"
        elif self._chain_stage == 4:
            ep_meta["lang"] = "open the microwave door"
        else:
            obj_lang = self.get_obj_lang()
            cont_lang = self.get_obj_lang(obj_name="container")
            ep_meta["lang"] = (
                f"pick the {obj_lang} from the microwave and place it on {cont_lang} located on the counter"
            )
        return ep_meta

    def _get_obj_cfgs(self):
        return [
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                microwavable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.microwave),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="container",
                ),
            ),
            dict(
                name="container",
                obj_groups=("plate",),
                placement=dict(
                    fixture=self.microwave,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                ),
            ),
        ]

    def _check_success(self):
        if self._chain_stage == 0:
            obj = self.objects["obj"]
            container = self.objects["container"]
            oc = self.check_contact(obj, container)
            mc = self.check_contact(container, self.microwave)
            return oc and mc and OU.gripper_obj_far(self)
        if self._chain_stage == 1:
            door_state = self.microwave.get_door_state(env=self)
            for joint_p in door_state.values():
                if joint_p > 0.05:
                    return False
            return True
        if self._chain_stage == 2:
            turned_on = self.microwave.get_state()["turned_on"]
            far = self.microwave.gripper_button_far(self, button="start_button")
            return bool(turned_on and far)
        if self._chain_stage == 3:
            mw = self.microwave
            robot = self.robots[0]
            d_best = chain4_turnoff_best_distance_to_stop(self)
            contact = bool(self.check_contact(robot.gripper["right"], f"{mw.name}_stop_button"))
            if contact or d_best <= CHAIN4_TURNOFF_PROXIMITY_M:
                mw._turned_on = False
                return True
            return False
        if self._chain_stage == 4:
            door_state = self.microwave.get_door_state(env=self)
            for joint_p in door_state.values():
                if joint_p < 0.90:
                    return False
            return True
        return OU.check_obj_in_receptacle(self, "obj", "container") and OU.gripper_obj_far(self)


class PnPRoboarmCosmosChain2DrawerOpenClose(OpenDrawer):
    """Same spawn/robot placement as ``OpenDrawer``; stage 1 language matches ``CloseDrawer``."""

    CHAIN_STAGE_HORIZON_NAMES = ("OpenDrawer", "CloseDrawer")
    EXCLUDE_LAYOUTS = [8]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chain_stage = 0

    def _reset_internal(self):
        super()._reset_internal()
        self._chain_stage = 0

    def advance_chain_stage(self):
        last = len(type(self).CHAIN_STAGE_HORIZON_NAMES) - 1
        if self._chain_stage < last:
            self._chain_stage += 1

    @property
    def chain_stage(self) -> int:
        return self._chain_stage

    def get_ep_meta(self):
        ep_meta = Kitchen.get_ep_meta(self)
        if self._chain_stage == 0:
            ep_meta["lang"] = f"open the {self.drawer_side} drawer"
        else:
            ep_meta["lang"] = f"close the {self.drawer_side} drawer"
        return ep_meta

    def _check_success(self):
        door_state = self.drawer.get_door_state(env=self)
        if self._chain_stage == 0:
            for joint_p in door_state.values():
                if joint_p < 0.95:
                    return False
            return True
        for joint_p in door_state.values():
            if joint_p > 0.05:
                return False
        return True
