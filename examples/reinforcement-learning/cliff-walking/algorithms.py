from __future__ import annotations

from collections.abc import Callable

import numpy as np

from cliff_walking import (
    CliffWalkingEnv,
    deterministic_greedy_action,
    epsilon_greedy_probs,
    grad_log_softmax,
    sample_action,
    softmax,
)


def _metrics() -> dict[str, list[float]]:
    return {"returns": [], "steps": [], "cliff_falls": [], "truncated": []}


def _record(
    metrics: dict[str, list[float]],
    total_reward: float,
    steps: int,
    cliff_falls: int,
    truncated: bool,
) -> None:
    metrics["returns"].append(total_reward)
    metrics["steps"].append(float(steps))
    metrics["cliff_falls"].append(float(cliff_falls))
    metrics["truncated"].append(float(truncated))


def _finish(table: np.ndarray, metrics: dict[str, list[float]]) -> dict[str, object]:
    return {"table": table, **metrics}


def _discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    returns = np.zeros(len(rewards), dtype=float)
    value = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        value = rewards[index] + gamma * value
        returns[index] = value
    return returns


def first_visit_mc_control(
    *,
    episodes: int,
    seed: int,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    max_steps: int = 500,
) -> dict[str, object]:
    env = CliffWalkingEnv()
    rng = np.random.default_rng(seed)
    q = np.zeros((env.num_states, env.num_actions))
    counts = np.zeros_like(q)
    metrics = _metrics()

    for _ in range(episodes):
        state = env.reset()
        trajectory: list[tuple[int, int, float]] = []
        total_reward = 0.0
        falls = 0
        terminated = False

        for step in range(1, max_steps + 1):
            action = sample_action(epsilon_greedy_probs(q[state], epsilon), rng)
            transition = env.step(state, action)
            trajectory.append((state, action, transition.reward))
            total_reward += transition.reward
            falls += int(transition.fell_from_cliff)
            state = transition.next_state
            terminated = transition.terminated
            if terminated:
                break

        returns = _discounted_returns([item[2] for item in trajectory], gamma)
        first_visits: dict[tuple[int, int], int] = {}
        for index, (visited_state, visited_action, _) in enumerate(trajectory):
            first_visits.setdefault((visited_state, visited_action), index)
        for (visited_state, visited_action), index in first_visits.items():
            counts[visited_state, visited_action] += 1.0
            q[visited_state, visited_action] += (
                returns[index] - q[visited_state, visited_action]
            ) / counts[visited_state, visited_action]

        _record(metrics, total_reward, step, falls, not terminated)

    return _finish(q, metrics)


def sarsa(
    *,
    episodes: int,
    seed: int,
    gamma: float = 0.9,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    max_steps: int = 500,
) -> dict[str, object]:
    env = CliffWalkingEnv()
    rng = np.random.default_rng(seed)
    q = np.zeros((env.num_states, env.num_actions))
    metrics = _metrics()

    for _ in range(episodes):
        state = env.reset()
        action = sample_action(epsilon_greedy_probs(q[state], epsilon), rng)
        total_reward = 0.0
        falls = 0
        terminated = False

        for step in range(1, max_steps + 1):
            transition = env.step(state, action)
            total_reward += transition.reward
            falls += int(transition.fell_from_cliff)
            cutoff = step == max_steps
            if transition.terminated or cutoff:
                target = transition.reward
                next_action = 0
            else:
                next_action = sample_action(
                    epsilon_greedy_probs(q[transition.next_state], epsilon), rng
                )
                target = transition.reward + gamma * q[transition.next_state, next_action]
            q[state, action] += alpha * (target - q[state, action])
            state, action = transition.next_state, next_action
            terminated = transition.terminated
            if terminated:
                break

        _record(metrics, total_reward, step, falls, not terminated)

    return _finish(q, metrics)


def q_learning(
    *,
    episodes: int,
    seed: int,
    gamma: float = 0.9,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    max_steps: int = 500,
) -> dict[str, object]:
    env = CliffWalkingEnv()
    rng = np.random.default_rng(seed)
    q = np.zeros((env.num_states, env.num_actions))
    metrics = _metrics()

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        falls = 0
        terminated = False

        for step in range(1, max_steps + 1):
            action = sample_action(epsilon_greedy_probs(q[state], epsilon), rng)
            transition = env.step(state, action)
            total_reward += transition.reward
            falls += int(transition.fell_from_cliff)
            cutoff = step == max_steps
            bootstrap = 0.0 if transition.terminated or cutoff else np.max(q[transition.next_state])
            target = transition.reward + gamma * bootstrap
            q[state, action] += alpha * (target - q[state, action])
            state = transition.next_state
            terminated = transition.terminated
            if terminated:
                break

        _record(metrics, total_reward, step, falls, not terminated)

    return _finish(q, metrics)


def reinforce_with_baseline(
    *,
    episodes: int,
    seed: int,
    gamma: float = 0.99,
    alpha_policy: float = 0.001,
    alpha_value: float = 0.01,
    max_steps: int = 500,
) -> dict[str, object]:
    env = CliffWalkingEnv()
    rng = np.random.default_rng(seed)
    logits = np.zeros((env.num_states, env.num_actions))
    values = np.zeros(env.num_states)
    metrics = _metrics()

    for _ in range(episodes):
        state = env.reset()
        trajectory: list[tuple[int, int, float, np.ndarray]] = []
        total_reward = 0.0
        falls = 0
        terminated = False

        for step in range(1, max_steps + 1):
            probs = softmax(logits[state])
            action = sample_action(probs, rng)
            transition = env.step(state, action)
            trajectory.append((state, action, transition.reward, probs))
            total_reward += transition.reward
            falls += int(transition.fell_from_cliff)
            state = transition.next_state
            terminated = transition.terminated
            if terminated:
                break

        returns = _discounted_returns([item[2] for item in trajectory], gamma)
        for index, (visited_state, action, _, probs) in enumerate(trajectory):
            advantage = returns[index] - values[visited_state]
            values[visited_state] += alpha_value * advantage
            logits[visited_state] += (
                alpha_policy
                * (gamma**index)
                * advantage
                * grad_log_softmax(probs, action)
            )

        _record(metrics, total_reward, step, falls, not terminated)

    return _finish(logits, metrics)


def actor_critic(
    *,
    episodes: int,
    seed: int,
    gamma: float = 0.99,
    alpha_actor: float = 0.01,
    alpha_critic: float = 0.1,
    max_steps: int = 500,
) -> dict[str, object]:
    env = CliffWalkingEnv()
    rng = np.random.default_rng(seed)
    logits = np.zeros((env.num_states, env.num_actions))
    values = np.zeros(env.num_states)
    metrics = _metrics()

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        falls = 0
        discount = 1.0
        terminated = False

        for step in range(1, max_steps + 1):
            probs = softmax(logits[state])
            action = sample_action(probs, rng)
            transition = env.step(state, action)
            total_reward += transition.reward
            falls += int(transition.fell_from_cliff)
            cutoff = step == max_steps
            bootstrap = 0.0 if transition.terminated or cutoff else values[transition.next_state]
            delta = transition.reward + gamma * bootstrap - values[state]
            values[state] += alpha_critic * delta
            logits[state] += (
                alpha_actor * discount * delta * grad_log_softmax(probs, action)
            )
            discount *= gamma
            state = transition.next_state
            terminated = transition.terminated
            if terminated:
                break

        _record(metrics, total_reward, step, falls, not terminated)

    return _finish(logits, metrics)


def weighted_importance_sampling_mc_control(
    *,
    episodes: int,
    seed: int,
    gamma: float = 0.9,
    epsilon: float = 0.2,
    max_steps: int = 500,
) -> dict[str, object]:
    env = CliffWalkingEnv()
    rng = np.random.default_rng(seed)
    q = np.zeros((env.num_states, env.num_actions))
    cumulative_weights = np.zeros_like(q)
    metrics = _metrics()

    for _ in range(episodes):
        state = env.reset()
        trajectory: list[tuple[int, int, float, float]] = []
        total_reward = 0.0
        falls = 0
        terminated = False

        for step in range(1, max_steps + 1):
            target_action = deterministic_greedy_action(q[state])
            behavior_probs = np.full(env.num_actions, epsilon / env.num_actions)
            behavior_probs[target_action] += 1.0 - epsilon
            action = sample_action(behavior_probs, rng)
            transition = env.step(state, action)
            trajectory.append((state, action, transition.reward, behavior_probs[action]))
            total_reward += transition.reward
            falls += int(transition.fell_from_cliff)
            state = transition.next_state
            terminated = transition.terminated
            if terminated:
                break

        return_value = 0.0
        weight = 1.0
        for visited_state, action, reward, behavior_prob in reversed(trajectory):
            return_value = reward + gamma * return_value
            cumulative_weights[visited_state, action] += weight
            q[visited_state, action] += (
                weight / cumulative_weights[visited_state, action]
            ) * (return_value - q[visited_state, action])
            if action != deterministic_greedy_action(q[visited_state]):
                break
            weight /= behavior_prob

        _record(metrics, total_reward, step, falls, not terminated)

    return _finish(q, metrics)


def off_policy_actor_critic(
    *,
    episodes: int,
    seed: int,
    gamma: float = 0.99,
    alpha_actor: float = 0.005,
    alpha_critic: float = 0.05,
    max_steps: int = 500,
) -> dict[str, object]:
    """Tabular actor-critic with action-level importance-sampling correction.

    The behavior policy is uniform and fixed. This is a small didactic analogue of
    OffPAC, not a reproduction of its linear GTD critic or eligibility traces.
    """

    env = CliffWalkingEnv()
    rng = np.random.default_rng(seed)
    logits = np.zeros((env.num_states, env.num_actions))
    values = np.zeros(env.num_states)
    behavior_prob = 1.0 / env.num_actions
    metrics = _metrics()

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        falls = 0
        terminated = False

        for step in range(1, max_steps + 1):
            action = int(rng.integers(env.num_actions))
            target_probs = softmax(logits[state])
            ratio = target_probs[action] / behavior_prob
            transition = env.step(state, action)
            total_reward += transition.reward
            falls += int(transition.fell_from_cliff)
            cutoff = step == max_steps
            bootstrap = 0.0 if transition.terminated or cutoff else values[transition.next_state]
            delta = transition.reward + gamma * bootstrap - values[state]
            values[state] += alpha_critic * ratio * delta
            logits[state] += (
                alpha_actor
                * ratio
                * delta
                * grad_log_softmax(target_probs, action)
            )
            state = transition.next_state
            terminated = transition.terminated
            if terminated:
                break

        _record(metrics, total_reward, step, falls, not terminated)

    return _finish(logits, metrics)


ALGORITHMS: dict[str, Callable[..., dict[str, object]]] = {
    "first_visit_mc": first_visit_mc_control,
    "sarsa": sarsa,
    "q_learning": q_learning,
    "reinforce_baseline": reinforce_with_baseline,
    "actor_critic": actor_critic,
    "weighted_is_mc": weighted_importance_sampling_mc_control,
    "off_policy_actor_critic": off_policy_actor_critic,
}
