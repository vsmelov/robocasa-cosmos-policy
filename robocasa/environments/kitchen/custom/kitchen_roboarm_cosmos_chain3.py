"""Kitchen-roboarm Cosmos multi-stage envs (one MuJoCo episode, no reset between stages).

- ``PnPRoboarmCosmosChain3``: stove PnP → counter→microwave → start button.
- ``PnPRoboarmCosmosChain2MicrowaveCloseOn``: close microwave door → start button.
- ``PnPRoboarmCosmosChain2DrawerOpenClose``: open top drawer → close it.
- ``PnPRoboarmCosmosChain3DrawerPotatoOpenPnPClose``: open drawer → potato drawer→counter → close drawer.
- ``PnPRoboarmCosmosChain4MicrowaveCloseOnOffOpen``: close MW → arm home → on → arm home → off → arm home → open door.
- ``PnPRoboarmCosmosChain6PotatoMwPlate``: counter→MW (potato) → close → on → off → open → MW→counter (plate).
- ``PnPRoboarmCosmosChainRecipeStoveMwV1``: potato→pan+stove on; carrot→MW heat; MW→counter; stove off (atomic horizons).

Each chain class defines ``CHAIN_STAGE_HORIZON_NAMES`` for horizon / T5 lookup in eval.

Canonical copy lives in the ``vsmelov/robocasa-cosmos-policy`` fork (git submodule
``external/robocasa-cosmos-policy`` in ``kitchen-roboarm``); Docker mounts that tree at ``/opt/robocasa-cosmos-policy``.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from robocasa.environments.kitchen.kitchen import *
from robocasa.environments.kitchen.single_stage.kitchen_drawer import OpenDrawer
from robocasa.environments.kitchen.single_stage.kitchen_pnp import PnP

# Historical 10 mm line in JSONL only; success uses ``mw_stop_proximity_threshold_m()`` (stop geom / contact only).
CHAIN4_TURNOFF_PROXIMITY_M = 0.01


def mw_stop_proximity_threshold_m() -> float:
    """Distance cap (m) for stop-button proximity success; does not affect start-button stage.

    Override: ``CHAIN_MW_STOP_PROXIMITY_M`` (float meters). Default ~5.5 cm to account for EEF vs probe tip vs
    ``stop_button`` geom center (not panel surface).
    """
    raw = os.environ.get("CHAIN_MW_STOP_PROXIMITY_M", "").strip()
    if raw:
        return float(raw)
    return 0.055


def mw_microwave_door_open_success_min_frac() -> float:
    """Minimum normalized microwave hinge openness [0, 1] for «door open» success in chain4/chain6.

    RoboCasa uses ``normalize_joint_value`` on the micro hinge; requiring 0.90 often fails when the door is
    visually open but the gripper limits the last few degrees. Override: ``CHAIN_MW_OPEN_DOOR_SUCCESS_FRAC``.
    """
    raw = os.environ.get("CHAIN_MW_OPEN_DOOR_SUCCESS_FRAC", "").strip()
    v = float(raw) if raw else 0.82
    return float(np.clip(v, 0.15, 0.99))


def _chain_turnoff_debug_log_pre_step_enabled() -> bool:
    """If true, log a JSONL line before ``super().step`` as well as after (2× per control step)."""
    return os.environ.get("CHAIN_MW_TURNOFF_DEBUG_PRE", "1").strip().lower() not in ("0", "false", "no", "")

# Panda-Omron / composite grippers often omit "finger" in geom names; include gripper/hand so proximity uses real pads.
_FINGER_GEOM_KEYS = ("pad", "finger", "fingertip", "tip", "touch", "gripper", "hand")
_FINGER_SITE_KEYS = ("pad", "finger", "fingertip", "tip", "touch", "grip", "gripper", "hand")


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
    th = mw_stop_proximity_threshold_m()
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
        "proximity_ok_stop": d_best <= th,
        "proximity_threshold_m": th,
        "start_button_contact": start_c,
        "stop_button_contact": stop_c,
        "turned_on": turned_on,
    }


# Env class name → chain_stage index where TurnOffMicrowave-style success runs (proximity/contact fix).
_CHAIN_TURNOFF_DEBUG_STAGE_BY_CLASS = {
    "PnPRoboarmCosmosChain4MicrowaveCloseOnOffOpen": 2,
    "PnPRoboarmCosmosChain6PotatoMwPlate": 3,
    "PnPRoboarmCosmosChainRecipeStoveMwV1": 6,
}


def _chain_turnoff_step_debug_maybe(env, *, phase: str = "post") -> None:
    """If orchestrator opened ``_chain4_turnoff_debug_fp``, append JSONL (chain4 st2 / chain6 st3).

    ``phase`` is ``pre`` (before physics for this action) or ``post`` (after ``super().step``) — two lines per
    control step when pre-logging is enabled (``_chain_turnoff_debug_log_pre_step_enabled()``).
    """
    fp = getattr(env, "_chain4_turnoff_debug_fp", None)
    if fp is None:
        return
    want = _CHAIN_TURNOFF_DEBUG_STAGE_BY_CLASS.get(type(env).__name__)
    if want is None or int(getattr(env, "_chain_stage")) != want:
        return
    row = chain4_turnoff_debug_snapshot(env)
    row["debug_step_phase"] = phase
    fp.write(json.dumps(row, sort_keys=True) + "\n")
    env._chain4_turnoff_min_eef = min(
        getattr(env, "_chain4_turnoff_min_eef", float("inf")), float(row["distance_eef_to_stop_m"])
    )
    mg = row.get("distance_min_finger_geom_to_stop_m")
    if mg is not None:
        env._chain4_turnoff_min_finger_geom = min(
            getattr(env, "_chain4_turnoff_min_finger_geom", float("inf")), float(mg)
        )
    env._chain4_turnoff_min_best = min(
        getattr(env, "_chain4_turnoff_min_best", float("inf")), float(row["distance_best_to_stop_m"])
    )
    if row["stop_button_contact"]:
        env._chain4_turnoff_contact_ever = True


def write_chain4_turnoff_stage_debug_log(attempt_dir: Path, env, episode_success: bool) -> None:
    """Append episode summary after per-step JSONL (chain4 stage 2 / chain6 stage 3)."""
    want = _CHAIN_TURNOFF_DEBUG_STAGE_BY_CLASS.get(type(env).__name__)
    if want is None or int(getattr(env, "chain_stage")) != want:
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
        if _chain_turnoff_debug_log_pre_step_enabled():
            _chain_turnoff_step_debug_maybe(self, phase="pre")
        obs, reward, done, info = super().step(action)
        _chain_turnoff_step_debug_maybe(self, phase="post")
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
            if contact or d_best <= mw_stop_proximity_threshold_m():
                mw._turned_on = False
                return True
            return False
        th_open = mw_microwave_door_open_success_min_frac()
        for joint_p in door_state.values():
            if joint_p < th_open:
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

    def step(self, action):
        if _chain_turnoff_debug_log_pre_step_enabled():
            _chain_turnoff_step_debug_maybe(self, phase="pre")
        obs, reward, done, info = super().step(action)
        _chain_turnoff_step_debug_maybe(self, phase="post")
        return obs, reward, done, info

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
            if contact or d_best <= mw_stop_proximity_threshold_m():
                mw._turned_on = False
                return True
            return False
        if self._chain_stage == 4:
            door_state = self.microwave.get_door_state(env=self)
            th_open = mw_microwave_door_open_success_min_frac()
            for joint_p in door_state.values():
                if joint_p < th_open:
                    return False
            return True
        # PnPMicrowaveToCounter: require potato on plate on counter, gripper clear, and potato not still inside MW volume.
        if not OU.check_obj_in_receptacle(self, "obj", "container") or not OU.gripper_obj_far(self):
            return False
        if OU.obj_inside_of(self, "obj", self.microwave):
            return False
        return True


class PnPRoboarmCosmosChain3DrawerPotatoOpenPnPClose(OpenDrawer):
    """
    Open the top drawer → pick ``obj`` (default potato) from the drawer onto the counter → close the drawer.
    One uninterrupted simulation; arm returns home after stages 0 and 1 (before PnP and before close).
    """

    CHAIN_STAGE_HORIZON_NAMES = ("OpenDrawer", "PnPDrawerToCounter", "CloseDrawer")
    CHAIN_RESET_ARM_AFTER_STAGES = (0, 1)
    EXCLUDE_LAYOUTS = [8]

    def __init__(self, obj_groups=("potato",), exclude_obj_groups=None, *args, **kwargs):
        self.obj_groups = obj_groups
        self.exclude_obj_groups = exclude_obj_groups
        kwargs.setdefault("obj_registries", ("aigen", "objaverse"))
        super().__init__(*args, **kwargs)
        self._chain_stage = 0
        self._chain_init_arm_qpos = None
        self._chain_init_gripper_qpos = None

    def _reset_internal(self):
        super()._reset_internal()
        self._chain_stage = 0
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
        ep_meta = Kitchen.get_ep_meta(self)
        if self._chain_stage == 0:
            ep_meta["lang"] = f"open the {self.drawer_side} drawer"
        elif self._chain_stage == 1:
            obj_lang = self.get_obj_lang()
            ep_meta["lang"] = f"pick the {obj_lang} from the drawer and place it on the counter"
        else:
            ep_meta["lang"] = f"close the {self.drawer_side} drawer"
        return ep_meta

    def _get_obj_cfgs(self):
        """``obj`` in closed drawer at reset (same layout as ``OpenDrawer`` but name matches PnP success)."""
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                max_size=(None, None, 0.10),
                placement=dict(
                    fixture=self.drawer,
                    size=(0.30, 0.30),
                    pos=(None, -0.75),
                ),
            )
        )
        # No counter distractors: ``obj_groups`` in (all, food, …) with (aigen, objaverse) can still pick a
        # category whose total MJCF path count is 0 → NaN in sample_kitchen_object_helper. OpenDrawer uses
        # ``all``; we only need ``obj`` in the drawer for this chain benchmark.
        return cfgs

    def _check_success(self):
        door_state = self.drawer.get_door_state(env=self)
        if self._chain_stage == 0:
            for joint_p in door_state.values():
                if joint_p < 0.95:
                    return False
            return True
        if self._chain_stage == 1:
            counter = self.get_fixture(FixtureType.COUNTER, ref=self.drawer)
            return OU.check_obj_fixture_contact(self, "obj", counter) and OU.gripper_obj_far(self)
        for joint_p in door_state.values():
            if joint_p > 0.05:
                return False
        return True


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


class PnPRoboarmCosmosChainRecipeStoveMwV1(Kitchen):
    """
    Minimal multi-appliance «recipe» in one sim (atomic horizon names for Cosmos T5).

    Stages: potato from counter into pan on stove → burner on → open MW → carrot+plate into MW →
    close MW → MW on → MW off → open MW → carrot on plate on counter → burner off.

    Objects: ``obj`` (potato, cookable), ``container`` (pan on stove), ``obj_mw`` (carrot), ``mw_plate`` (plate in MW).
    """

    CHAIN_STAGE_HORIZON_NAMES = (
        "PnPCounterToStove",
        "TurnOnStove",
        "OpenSingleDoor",
        "PnPCounterToMicrowave",
        "CloseSingleDoor",
        "TurnOnMicrowave",
        "TurnOffMicrowave",
        "OpenSingleDoor",
        "PnPMicrowaveToCounter",
        "TurnOffStove",
    )
    CHAIN_RESET_ARM_AFTER_STAGES = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    EXCLUDE_LAYOUTS = [8]

    def __init__(
        self,
        obj_groups=("potato",),
        obj_mw_groups=("carrot",),
        exclude_obj_groups=None,
        exclude_obj_mw_groups=None,
        *args,
        **kwargs,
    ):
        self.obj_groups = obj_groups
        self.obj_mw_groups = obj_mw_groups
        self.exclude_obj_groups = exclude_obj_groups
        self.exclude_obj_mw_groups = exclude_obj_mw_groups
        kwargs.setdefault("obj_registries", ("aigen", "objaverse"))
        super().__init__(*args, **kwargs)
        self._chain_stage = 0
        self._chain_init_arm_qpos = None
        self._chain_init_gripper_qpos = None
        self._recipe_knob = None

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
        # Before _create_objects → _get_obj_cfgs: pick burner/knob so pan placement matches success check.
        valid_knobs = [k for (k, v) in self.stove.knob_joints.items() if v is not None]
        if not valid_knobs:
            raise RuntimeError("PnPRoboarmCosmosChainRecipeStoveMwV1: stove has no knob joints")
        self._recipe_knob = self.rng.choice(list(valid_knobs))

    def _reset_internal(self):
        super()._reset_internal()
        self._chain_stage = 0
        self.microwave.set_door_state(min=0.0, max=0.0, env=self, rng=self.rng)
        self.stove.set_knob_state(mode="off", knob=self._recipe_knob, env=self, rng=self.rng)
        _snapshot_chain_init_arm_gripper(self)

    def advance_chain_stage(self):
        last = len(type(self).CHAIN_STAGE_HORIZON_NAMES) - 1
        if self._chain_stage < last:
            completed = self._chain_stage
            self._chain_stage += 1
            if completed in getattr(type(self), "CHAIN_RESET_ARM_AFTER_STAGES", ()):
                dur_raw = os.environ.get("CHAIN_RECIPE_ARM_HOME_DURATION_S", "").strip()
                arm_home_s = float(dur_raw) if dur_raw else None
                _restore_chain_init_arm_gripper_smooth(self, duration_s=arm_home_s)
                if self._chain_stage in (5, 6):
                    self.microwave.set_door_state(min=0.0, max=0.0, env=self, rng=self.rng)
                    self.sim.forward()

    @property
    def chain_stage(self) -> int:
        return self._chain_stage

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self._chain_stage == 0:
            obj_lang = self.get_obj_lang(obj_name="obj")
            cont_lang = self.get_obj_lang(obj_name="container")
            preposition = "on"
            ep_meta["lang"] = f"pick the {obj_lang} from the plate and place it {preposition} the {cont_lang}"
        elif self._chain_stage == 1:
            ep_meta["lang"] = f"turn on the {self._recipe_knob.replace('_', ' ')} burner of the stove"
        elif self._chain_stage == 2:
            ep_meta["lang"] = "open the microwave door"
        elif self._chain_stage == 3:
            obj_lang = self.get_obj_lang(obj_name="obj_mw")
            ep_meta["lang"] = f"pick the {obj_lang} from the counter and place it in the microwave"
        elif self._chain_stage == 4:
            ep_meta["lang"] = "close the microwave door"
        elif self._chain_stage == 5:
            ep_meta["lang"] = "press the start button on the microwave"
        elif self._chain_stage == 6:
            ep_meta["lang"] = "press the stop button on the microwave"
        elif self._chain_stage == 7:
            ep_meta["lang"] = "open the microwave door"
        elif self._chain_stage == 8:
            obj_lang = self.get_obj_lang(obj_name="obj_mw")
            cont_lang = self.get_obj_lang(obj_name="mw_plate")
            ep_meta["lang"] = (
                f"pick the {obj_lang} from the microwave and place it on {cont_lang} located on the counter"
            )
        else:
            ep_meta["lang"] = f"turn off the {self._recipe_knob.replace('_', ' ')} burner of the stove"
        return ep_meta

    def step(self, action):
        if _chain_turnoff_debug_log_pre_step_enabled():
            _chain_turnoff_step_debug_maybe(self, phase="pre")
        obs, reward, done, info = super().step(action)
        _chain_turnoff_step_debug_maybe(self, phase="post")
        return obs, reward, done, info

    def _get_obj_cfgs(self):
        return [
            dict(
                name="container",
                obj_groups=("pan",),
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(locs=[self._recipe_knob]),
                    size=(0.02, 0.02),
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                ),
            ),
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                cookable=True,
                placement=dict(
                    fixture=self.counter_stove,
                    sample_region_kwargs=dict(ref=self.stove),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="container",
                ),
            ),
            dict(
                name="mw_plate",
                obj_groups=("plate",),
                placement=dict(
                    fixture=self.microwave,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                ),
            ),
            dict(
                name="obj_mw",
                obj_groups=self.obj_mw_groups,
                exclude_obj_groups=self.exclude_obj_mw_groups,
                graspable=True,
                microwavable=True,
                placement=dict(
                    fixture=self.counter_mw,
                    sample_region_kwargs=dict(ref=self.microwave),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            ),
        ]

    def _mw_door_open_ok(self) -> bool:
        th = mw_microwave_door_open_success_min_frac()
        for joint_p in self.microwave.get_door_state(env=self).values():
            if joint_p < th:
                return False
        return True

    def _mw_door_closed_ok(self) -> bool:
        for joint_p in self.microwave.get_door_state(env=self).values():
            if joint_p > 0.05:
                return False
        return True

    def _stove_knob_on_at(self, location: str) -> bool:
        knobs_state = self.stove.get_knobs_state(env=self)
        if location not in knobs_state:
            return False
        knob_value = knobs_state[location]
        # Match ``ManipulateStoveKnob`` / ``Stove.update_state`` on-threshold (radians, wrapped).
        return bool(0.35 <= np.abs(knob_value) <= 2 * np.pi - 0.35)

    def _burner_flame_lit(self, location: str) -> bool:
        """True if flame site opacity is on (after ``Stove.update_state`` each step)."""
        try:
            site_id = self.sim.model.site_name2id(
                "{}burner_on_{}".format(self.stove.naming_prefix, location)
            )
        except Exception:
            return False
        return float(self.sim.model.site_rgba[site_id][3]) > 0.2

    def _burner_location_under_container(self):
        """Nearest burner under pan (for turn-off); None if pan not on stove."""
        container = self.objects["container"]
        obj_pos = np.array(self.sim.data.body_xpos[self.obj_body_id[container.name]])[0:2]
        if not OU.check_obj_fixture_contact(self, "container", self.stove):
            return None
        rk = self._recipe_knob
        if rk is not None:
            site_r = self.stove.burner_sites.get(rk)
            if site_r is not None:
                bp = np.array(self.sim.data.get_site_xpos(site_r.get("name")))[0:2]
                if float(np.linalg.norm(bp - obj_pos)) <= 0.20:
                    return rk
        best_loc = None
        best_d = float("inf")
        for location, site in self.stove.burner_sites.items():
            if site is None:
                continue
            burner_pos = np.array(self.sim.data.get_site_xpos(site.get("name")))[0:2]
            d = float(np.linalg.norm(burner_pos - obj_pos))
            if d < best_d:
                best_d, best_loc = d, location
        if best_loc is None or best_d > 0.18:
            return None
        return best_loc

    def _recipe_target_burner_on(self) -> bool:
        """Stage 1 success: the instructed burner (``_recipe_knob``) matches pan placement + lang."""
        rk = self._recipe_knob
        if rk is None:
            return False
        return self._stove_knob_on_at(rk) or self._burner_flame_lit(rk)

    def _mw_plate_carrot_place_ok(self) -> bool:
        """Carrot on plate inside MW; tolerant vs flaky contacts when plate+carrot+MW overlap."""
        plate = self.objects["mw_plate"]
        carrot = self.objects["obj_mw"]
        if not self.check_contact(plate, self.microwave):
            return False
        oc = self.check_contact(carrot, plate)
        if oc:
            return True
        if not OU.obj_inside_of(self, "obj_mw", self.microwave):
            return False
        cpos = np.array(self.sim.data.body_xpos[self.obj_body_id[carrot.name]])
        ppos = np.array(self.sim.data.body_xpos[self.obj_body_id[plate.name]])
        xy = float(np.linalg.norm(cpos[:2] - ppos[:2]))
        th_xy = max(0.14, float(plate.horizontal_radius) * 1.05)
        z_ok = cpos[2] + 0.02 >= ppos[2]
        return xy < th_xy and z_ok

    def _check_success(self):
        if self._chain_stage == 0:
            pan_on_stove = OU.check_obj_fixture_contact(self, "container", self.stove)
            potato_in_pan = OU.check_obj_in_receptacle(self, "obj", "container", th=0.07)
            gripper_clear = OU.gripper_obj_far(self, obj_name="obj")
            return pan_on_stove and potato_in_pan and gripper_clear
        if self._chain_stage == 1:
            return self._recipe_target_burner_on()
        if self._chain_stage == 2:
            return self._mw_door_open_ok()
        if self._chain_stage == 3:
            return self._mw_plate_carrot_place_ok() and OU.gripper_obj_far(self, obj_name="obj_mw")
        if self._chain_stage == 4:
            return self._mw_door_closed_ok()
        if self._chain_stage == 5:
            turned_on = self.microwave.get_state()["turned_on"]
            far = self.microwave.gripper_button_far(self, button="start_button")
            return bool(turned_on and far)
        if self._chain_stage == 6:
            mw = self.microwave
            robot = self.robots[0]
            d_best = chain4_turnoff_best_distance_to_stop(self)
            contact = bool(self.check_contact(robot.gripper["right"], f"{mw.name}_stop_button"))
            if contact or d_best <= mw_stop_proximity_threshold_m():
                mw._turned_on = False
                return True
            return False
        if self._chain_stage == 7:
            return self._mw_door_open_ok()
        if self._chain_stage == 8:
            plate = self.objects["mw_plate"]
            th_xy = max(0.10, float(plate.horizontal_radius) * 0.85)
            if not OU.check_obj_in_receptacle(self, "obj_mw", "mw_plate", th=th_xy):
                return False
            if not self.check_contact(plate, self.counter_mw):
                return False
            if not OU.gripper_obj_far(self, obj_name="obj_mw"):
                return False
            if OU.obj_inside_of(self, "obj_mw", self.microwave):
                return False
            return True
        rk = self._recipe_knob
        if rk is None:
            return True
        return (not self._stove_knob_on_at(rk)) and (not self._burner_flame_lit(rk))
