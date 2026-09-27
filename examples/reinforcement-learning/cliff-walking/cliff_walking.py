from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transition:
    next_state: int
    reward: float
    terminated: bool
    fell_from_cliff: bool


class CliffWalkingEnv:
    """The 3x5 variant used by the original Dr.trillion post."""

    rows = 3
    cols = 5
    num_states = rows * cols
    num_actions = 4
    start_state = 10
    goal_state = 14
    cliff_states = frozenset({11, 12, 13})

    def reset(self) -> int:
        return self.start_state

    def step(self, state: int, action: int) -> Transition:
        if not 0 <= state < self.num_states:
            raise ValueError(f"invalid state: {state}")
        if not 0 <= action < self.num_actions:
            raise ValueError(f"invalid action: {action}")

        row, col = divmod(state, self.cols)
        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # down
            row = min(self.rows - 1, row + 1)
        elif action == 2:  # left
            col = max(0, col - 1)
        else:  # right
            col = min(self.cols - 1, col + 1)

        next_state = row * self.cols + col
        if next_state in self.cliff_states:
            return Transition(self.start_state, -100.0, False, True)
        if next_state == self.goal_state:
            return Transition(next_state, 10.0, True, False)
        return Transition(next_state, -1.0, False, False)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def grad_log_softmax(probs: np.ndarray, action: int) -> np.ndarray:
    gradient = -probs.copy()
    gradient[action] += 1.0
    return gradient


def epsilon_greedy_probs(q_row: np.ndarray, epsilon: float) -> np.ndarray:
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1")
    probs = np.full(q_row.size, epsilon / q_row.size)
    greedy = np.flatnonzero(np.isclose(q_row, np.max(q_row)))
    probs[greedy] += (1.0 - epsilon) / greedy.size
    return probs


def sample_action(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(probs.size, p=probs))


def greedy_action(values: np.ndarray, rng: np.random.Generator) -> int:
    candidates = np.flatnonzero(np.isclose(values, np.max(values)))
    return int(rng.choice(candidates))


def deterministic_greedy_action(values: np.ndarray) -> int:
    """Fixed tie rule for the deterministic target policy in off-policy MC."""

    return int(np.flatnonzero(np.isclose(values, np.max(values)))[0])


def evaluate_greedy(
    table: np.ndarray,
    *,
    episodes: int = 200,
    seed: int = 10_000,
    max_steps: int = 500,
) -> dict[str, float]:
    env = CliffWalkingEnv()
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    steps: list[int] = []
    cliff_falls: list[int] = []
    successes = 0

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        falls = 0
        used_steps = 0
        for used_steps in range(1, max_steps + 1):
            action = greedy_action(table[state], rng)
            transition = env.step(state, action)
            total_reward += transition.reward
            falls += int(transition.fell_from_cliff)
            state = transition.next_state
            if transition.terminated:
                successes += 1
                break
        returns.append(total_reward)
        steps.append(used_steps)
        cliff_falls.append(falls)

    return {
        "mean_return": float(np.mean(returns)),
        "mean_steps": float(np.mean(steps)),
        "mean_cliff_falls": float(np.mean(cliff_falls)),
        "success_rate": successes / episodes,
    }


def policy_grid(table: np.ndarray) -> str:
    env = CliffWalkingEnv()
    arrows = np.array(["↑", "↓", "←", "→"])
    rows: list[str] = []
    for row in range(env.rows):
        cells: list[str] = []
        for col in range(env.cols):
            state = row * env.cols + col
            if state == env.start_state:
                cells.append("S")
            elif state == env.goal_state:
                cells.append("G")
            elif state in env.cliff_states:
                cells.append("C")
            else:
                cells.append(str(arrows[deterministic_greedy_action(table[state])]))
        rows.append(" ".join(cells))
    return "\n".join(rows)
