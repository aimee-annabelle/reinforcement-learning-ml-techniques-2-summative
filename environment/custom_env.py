"""Custom Rwanda health clinic environment for reinforcement learning experiments."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from typing import Any, Dict, List, Optional, Tuple

try:
    from environment.rendering import ClinicRenderer
except ImportError:  # Rendering is optional when pygame is unavailable
    ClinicRenderer = None  # type: ignore

REQUEST_NCD_TEST = 0
REQUEST_INFECTION_TEST = 1
DIAGNOSE_CHRONIC = 2
DIAGNOSE_INFECTION = 3
ALLOCATE_MED = 4
REFER_PATIENT = 5
WAIT = 6

ACTION_MEANINGS = {
    REQUEST_NCD_TEST: "Request NCD test",
    REQUEST_INFECTION_TEST: "Request infection test",
    DIAGNOSE_CHRONIC: "Diagnose chronic condition",
    DIAGNOSE_INFECTION: "Diagnose infection",
    ALLOCATE_MED: "Allocate medication",
    REFER_PATIENT: "Refer patient",
    WAIT: "Wait / no action",
}

CONDITION_HEALTHY_OR_MILD = 0
CONDITION_CHRONIC = 1
CONDITION_INFECTION = 2
CONDITION_BOTH_SERIOUS = 3


class RwandaHealthEnv(gym.Env):
    """Mission-aligned clinic environment used across all experiments."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        max_steps: int = 50,
        initial_test_kits: int = 10,
        initial_meds: int = 20,
        max_queue: int = 30,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.initial_test_kits = initial_test_kits
        self.initial_meds = initial_meds
        self.max_queue = max_queue
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)

        self.np_random = np.random.RandomState()
        self.renderer: Optional[ClinicRenderer] = None

        self._reset_internal_state()

    def _reset_internal_state(self) -> None:
        self.test_kits = self.initial_test_kits
        self.meds = self.initial_meds
        self.queue_length = 0
        self.step_count = 0
        self.last_action = WAIT
        self.last_reward = 0.0
        self.true_condition = CONDITION_HEALTHY_OR_MILD

        self.patient_age = 0
        self.symptom_severity = 0.0
        self.chronic_risk = 0.0
        self.infection_risk = 0.0
        self.comorbidity_flag = 0.0

        self.ncd_tested = False
        self.infection_tested = False
        self.diagnosed = False
        self.treated = False

    def seed(self, seed: Optional[int] = None) -> List[int]:
        self.np_random = np.random.RandomState(seed)
        return [seed] if seed is not None else []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        self._reset_internal_state()
        self.queue_length = int(self.np_random.randint(0, 5))
        self._generate_new_patient()
        observation = self._get_obs()
        return observation, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action: {action}"
        self.step_count += 1
        self.last_action = int(action)

        reward = -1.0 * (self.queue_length / max(self.max_queue, 1))

        if action == REQUEST_NCD_TEST:
            reward += self._handle_ncd_test()
        elif action == REQUEST_INFECTION_TEST:
            reward += self._handle_infection_test()
        elif action in [DIAGNOSE_CHRONIC, DIAGNOSE_INFECTION]:
            reward += self._reward_for_diagnosis(action)
        elif action == ALLOCATE_MED:
            reward += self._handle_treatment()
        elif action == REFER_PATIENT:
            reward += self._handle_referral()
        elif action == WAIT:
            reward += self._handle_wait()

        if self.diagnosed and self.treated and action != REFER_PATIENT:
            reward += 5.0
            self._generate_new_patient()

        self._update_queue()

        terminated = False
        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True
        if self.test_kits == 0 and self.meds == 0:
            terminated = True
            reward -= 10.0
        if self.queue_length >= self.max_queue:
            terminated = True
            reward -= 10.0

        self.last_reward = reward
        observation = self._get_obs()
        info: Dict[str, Any] = {
            "true_condition": self.true_condition,
            "test_kits": self.test_kits,
            "meds": self.meds,
            "queue_length": self.queue_length,
            "ncd_tested": self.ncd_tested,
            "infection_tested": self.infection_tested,
            "diagnosed": self.diagnosed,
            "treated": self.treated,
        }
        return observation, reward, terminated, truncated, info

    def _handle_ncd_test(self) -> float:
        if self.ncd_tested:
            return -3.0
        if self.test_kits <= 0:
            return -5.0
        self.test_kits -= 1
        self.ncd_tested = True
        bonus = 2.0 if self.true_condition in [CONDITION_CHRONIC, CONDITION_BOTH_SERIOUS] else -0.5
        if bonus > 0.0:
            self.chronic_risk = float(np.clip(self.chronic_risk + 0.2, 0.0, 1.0))
        return -0.5 + bonus

    def _handle_infection_test(self) -> float:
        if self.infection_tested:
            return -3.0
        if self.test_kits <= 0:
            return -5.0
        self.test_kits -= 1
        self.infection_tested = True
        bonus = 2.0 if self.true_condition in [CONDITION_INFECTION, CONDITION_BOTH_SERIOUS] else -0.5
        if bonus > 0.0:
            self.infection_risk = float(np.clip(self.infection_risk + 0.2, 0.0, 1.0))
        return -0.5 + bonus

    def _handle_treatment(self) -> float:
        if self.treated:
            return -3.0
        if not self.diagnosed:
            return -5.0
        if self.meds <= 0:
            return -5.0
        self.meds -= 1
        self.treated = True
        return self._reward_for_treatment()

    def _handle_referral(self) -> float:
        if not self.diagnosed:
            return -5.0
        if self.true_condition == CONDITION_BOTH_SERIOUS:
            reward = 8.0
        elif self.true_condition in [CONDITION_CHRONIC, CONDITION_INFECTION]:
            reward = 3.0
        else:
            reward = -2.0
        self._generate_new_patient()
        return reward

    def _handle_wait(self) -> float:
        if self.test_kits == 0 or self.meds == 0:
            return -0.2
        return -1.0

    def _reward_for_diagnosis(self, action: int) -> float:
        if action == DIAGNOSE_CHRONIC:
            if self.diagnosed:
                return -5.0
            if not self.ncd_tested:
                self.diagnosed = True
                return 3.0 if self.true_condition in [CONDITION_CHRONIC, CONDITION_BOTH_SERIOUS] else -10.0
            self.diagnosed = True
            return 10.0 if self.true_condition in [CONDITION_CHRONIC, CONDITION_BOTH_SERIOUS] else -6.0
        if action == DIAGNOSE_INFECTION:
            if self.diagnosed:
                return -5.0
            if not self.infection_tested:
                self.diagnosed = True
                return 3.0 if self.true_condition in [CONDITION_INFECTION, CONDITION_BOTH_SERIOUS] else -10.0
            self.diagnosed = True
            return 10.0 if self.true_condition in [CONDITION_INFECTION, CONDITION_BOTH_SERIOUS] else -6.0
        return 0.0

    def _reward_for_treatment(self) -> float:
        if self.true_condition in [CONDITION_CHRONIC, CONDITION_INFECTION]:
            return 6.0
        if self.true_condition == CONDITION_BOTH_SERIOUS:
            return 8.0
        return -4.0

    def _update_queue(self) -> None:
        served = 1
        arrivals = int(self.np_random.randint(0, 3))
        self.queue_length = max(0, self.queue_length - served + arrivals)

    def _generate_new_patient(self) -> None:
        self.true_condition = int(
            self.np_random.choice(
                [
                    CONDITION_HEALTHY_OR_MILD,
                    CONDITION_CHRONIC,
                    CONDITION_INFECTION,
                    CONDITION_BOTH_SERIOUS,
                ],
                p=[0.4, 0.25, 0.25, 0.10],
            )
        )
        self.patient_age = int(self.np_random.randint(15, 80))
        self.symptom_severity = float(self.np_random.uniform(0.0, 1.0))

        if self.true_condition in [CONDITION_CHRONIC, CONDITION_BOTH_SERIOUS]:
            self.chronic_risk = float(self.np_random.uniform(0.6, 1.0))
        else:
            self.chronic_risk = float(self.np_random.uniform(0.0, 0.7))

        if self.true_condition in [CONDITION_INFECTION, CONDITION_BOTH_SERIOUS]:
            self.infection_risk = float(self.np_random.uniform(0.6, 1.0))
        else:
            self.infection_risk = float(self.np_random.uniform(0.0, 0.7))

        self.comorbidity_flag = 1.0 if self.true_condition == CONDITION_BOTH_SERIOUS else float(self.np_random.rand() < 0.2)
        self.ncd_tested = False
        self.infection_tested = False
        self.diagnosed = False
        self.treated = False

    def _get_obs(self) -> np.ndarray:
        age_norm = (self.patient_age - 15) / (80 - 15)
        test_kits_norm = float(np.clip(self.test_kits / max(self.initial_test_kits, 1), 0.0, 1.0))
        meds_norm = float(np.clip(self.meds / max(self.initial_meds, 1), 0.0, 1.0))
        queue_norm = float(np.clip(self.queue_length / max(self.max_queue, 1), 0.0, 1.0))
        time_norm = self.step_count / max(self.max_steps, 1)
        last_action_norm = self.last_action / 6.0
        obs = np.array(
            [
                age_norm,
                self.symptom_severity,
                self.chronic_risk,
                self.infection_risk,
                self.comorbidity_flag,
                test_kits_norm,
                meds_norm,
                queue_norm,
                time_norm,
                last_action_norm,
                float(self.ncd_tested),
                float(self.infection_tested),
            ],
            dtype=np.float32,
        )
        return obs

    def render(self) -> Optional[str]:
        if self.render_mode == "ansi":
            return (
                f"Step: {self.step_count}, Condition: {self.true_condition}, "
                f"Tests: {self.test_kits}, Meds: {self.meds}, Queue: {self.queue_length}, "
                f"Last action: {ACTION_MEANINGS[self.last_action]}, Reward: {self.last_reward:.2f}"
            )
        if self.render_mode == "human" and ClinicRenderer is not None:
            if self.renderer is None:
                self.renderer = ClinicRenderer()
            state = {
                "patient_age": self.patient_age,
                "symptom_severity": self.symptom_severity,
                "chronic_risk": self.chronic_risk,
                "infection_risk": self.infection_risk,
                "comorbidity_flag": self.comorbidity_flag,
                "ncd_tested": self.ncd_tested,
                "infection_tested": self.infection_tested,
                "diagnosed": self.diagnosed,
                "treated": self.treated,
                "test_kits": self.test_kits,
                "initial_test_kits": self.initial_test_kits,
                "meds": self.meds,
                "initial_meds": self.initial_meds,
                "queue_length": self.queue_length,
                "max_queue": self.max_queue,
                "last_action": self.last_action,
                "last_reward": self.last_reward,
                "step_count": self.step_count,
                "max_steps": self.max_steps,
            }
            self.renderer.render(state)
            return None
        print(
            f"[Step {self.step_count}] {ACTION_MEANINGS[self.last_action]} | Reward={self.last_reward:.2f} | Queue={self.queue_length}"
        )
        return None

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def __del__(self) -> None:  # Defensive cleanup when interpreter exits
        self.close()


def make_env(
    *,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
    monitor: bool = False,
):
    """Utility factory to create monitored environments when needed."""

    from stable_baselines3.common.monitor import Monitor

    def _init():
        env = RwandaHealthEnv(render_mode=render_mode)
        if seed is not None:
            env.reset(seed=seed)
        return Monitor(env) if monitor else env

    return _init


__all__ = [
    "RwandaHealthEnv",
    "make_env",
    "ACTION_MEANINGS",
    "REQUEST_NCD_TEST",
    "REQUEST_INFECTION_TEST",
    "DIAGNOSE_CHRONIC",
    "DIAGNOSE_INFECTION",
    "ALLOCATE_MED",
    "REFER_PATIENT",
    "WAIT",
    "CONDITION_HEALTHY_OR_MILD",
    "CONDITION_CHRONIC",
    "CONDITION_INFECTION",
    "CONDITION_BOTH_SERIOUS",
]
