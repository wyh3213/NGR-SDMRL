from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_


EPS = 1e-8


def _to_device_graph(data, device):
    return data.to(device) if hasattr(data, "to") else data


@dataclass
class ObjectiveSpec:
    """
    direction='max' means larger metric is better.
    direction='min' means smaller metric is better; the trainer converts it to reward by negation.
    """
    name: str
    fn: Callable[[Tensor, Tensor], Tensor]
    weight: float = 1.0
    direction: str = "max"


@dataclass
class RolloutItem:
    latent_action: Tensor
    old_log_prob: Tensor
    old_mean: Tensor
    old_std: Tensor
    reward_vector: Tensor
    scalar_reward: Optional[Tensor] = None
    advantage: Optional[Tensor] = None


def binary_f1_reward(y_prob: Tensor, y_true: Tensor, threshold: float = 0.5, eps: float = EPS) -> Tensor:
    y_prob = y_prob.float()
    y_true = y_true.float()
    pred = (y_prob >= threshold).float()

    tp = (pred * y_true).sum()
    fp = (pred * (1.0 - y_true)).sum()
    fn = ((1.0 - pred) * y_true).sum()

    return (2.0 * tp) / (2.0 * tp + fp + fn + eps)


def precision_at_k_reward(y_prob: Tensor, y_true: Tensor, k: int) -> Tensor:
    y_prob = y_prob.float().reshape(-1)
    y_true = y_true.float().reshape(-1)
    k = max(1, min(k, y_prob.numel()))
    topk_idx = torch.topk(y_prob, k=k).indices
    return y_true[topk_idx].mean()


def recall_at_k_reward(y_prob: Tensor, y_true: Tensor, k: int, eps: float = EPS) -> Tensor:
    y_prob = y_prob.float().reshape(-1)
    y_true = y_true.float().reshape(-1)
    k = max(1, min(k, y_prob.numel()))
    topk_idx = torch.topk(y_prob, k=k).indices
    hits = y_true[topk_idx].sum()
    total_pos = y_true.sum()
    return hits / (total_pos + eps)


def negative_bce_reward(y_prob: Tensor, y_true: Tensor, eps: float = EPS) -> Tensor:
    y_prob = y_prob.float().clamp(min=eps, max=1.0 - eps)
    y_true = y_true.float()
    return -F.binary_cross_entropy(y_prob, y_true)


def density_match_reward(y_prob: Tensor, y_true: Tensor) -> Tensor:
    y_prob = y_prob.float()
    y_true = y_true.float()
    return -(y_prob.mean() - y_true.mean()).abs()


def compute_reward_vector(y_prob: Tensor, y_true: Tensor, objectives: Sequence[ObjectiveSpec]) -> Tensor:
    values = []
    for obj in objectives:
        score = obj.fn(y_prob, y_true)
        if not torch.is_tensor(score):
            score = torch.tensor(score, device=y_prob.device, dtype=y_prob.dtype)
        score = score.float().reshape(())
        if obj.direction.lower() == "min":
            score = -score
        values.append(score)
    return torch.stack(values, dim=0)


def gaussian_kl(old_mean: Tensor, old_std: Tensor, new_mean: Tensor, new_std: Tensor, eps: float = EPS) -> Tensor:
    old_std = old_std.clamp_min(eps)
    new_std = new_std.clamp_min(eps)
    old_var = old_std.pow(2)
    new_var = new_std.pow(2)

    kl = torch.log(new_std / old_std) + (old_var + (old_mean - new_mean).pow(2)) / (2.0 * new_var) - 0.5
    if kl.ndim > 1:
        return kl.mean(dim=-1).mean()
    return kl.mean()


class DynamicWeightController:
    """
    Dynamic weights:
    objectives that improve more slowly automatically get larger weights.
    Manual base weights still act as the prior preference.
    """
    def __init__(
        self,
        objectives: Sequence[ObjectiveSpec],
        mode: str = "manual",
        ema: float = 0.9,
        eps: float = EPS,
        progress_clip: float = 2.0,
    ):
        if mode not in {"manual", "dynamic"}:
            raise ValueError("mode must be 'manual' or 'dynamic'")

        self.mode = mode
        self.ema = ema
        self.eps = eps
        self.progress_clip = progress_clip

        base = torch.tensor(
            [max(float(obj.weight), eps) for obj in objectives],
            dtype=torch.float32,
        )
        self.manual_weights = base / base.sum().clamp_min(eps)
        self.running_weights = self.manual_weights.clone()
        self.initial_reward = None

    def get_weights(self, reward_matrix: Tensor) -> Tensor:
        manual = self.manual_weights.to(device=reward_matrix.device, dtype=reward_matrix.dtype)

        if self.mode == "manual":
            return manual

        current_reward = reward_matrix.mean(dim=0).detach()
        if self.initial_reward is None:
            self.initial_reward = current_reward
            self.running_weights = manual.detach().cpu()
            return manual

        progress = (current_reward - self.initial_reward) / (self.initial_reward.abs() + self.eps)
        progress = torch.nan_to_num(progress, nan=0.0, posinf=0.0, neginf=0.0)
        progress = progress.clamp(-self.progress_clip, self.progress_clip)

        hardness = torch.exp(-progress)
        weights = hardness * manual
        weights = weights / weights.sum().clamp_min(self.eps)

        running = self.ema * self.running_weights.to(weights.device) + (1.0 - self.ema) * weights
        running = running / running.sum().clamp_min(self.eps)

        self.running_weights = running.detach().cpu()
        return running


class MORLPPOTrainer:
    """
    Stable PPO-style MORL trainer for GANIB.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        objectives: Sequence[ObjectiveSpec],
        criterion: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        clip_epsilon: float = 0.1,
        ppo_epochs: int = 1,
        n_rollouts: int = 4,
        supervised_coef: float = 1.0,
        rl_coef: float = 0.2,
        entropy_coef: float = 5e-4,
        kl_coef: float = 1e-3,
        max_grad_norm: float = 1.0,
        dynamic_weight: bool = False,
        normalize_objective_rewards: bool = True,
        rl_std_scale: float = 1.0,
        ratio_log_clip: float = 5.0,
        reward_clip: Optional[float] = 5.0,
        advantage_clip: Optional[float] = 5.0,
        target_kl: float = 0.03,
        device: Optional[torch.device] = None,
        eps: float = EPS,
    ):
        if len(objectives) == 0:
            raise ValueError("At least one objective is required.")

        self.model = model
        self.optimizer = optimizer
        self.objectives = list(objectives)
        self.criterion = criterion if criterion is not None else nn.BCELoss()
        self.scheduler = scheduler

        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.n_rollouts = n_rollouts
        self.supervised_coef = supervised_coef
        self.rl_coef = rl_coef
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_objective_rewards = normalize_objective_rewards
        self.rl_std_scale = rl_std_scale
        self.ratio_log_clip = ratio_log_clip
        self.reward_clip = reward_clip
        self.advantage_clip = advantage_clip
        self.target_kl = target_kl
        self.eps = eps

        self.device = device if device is not None else next(model.parameters()).device
        self.weight_controller = DynamicWeightController(
            objectives=self.objectives,
            mode="dynamic" if dynamic_weight else "manual",
            eps=eps,
        )

    def _prepare_inputs(self, processed_features, dis_data, meta_data, target):
        processed_features = processed_features.to(self.device)
        dis_data = _to_device_graph(dis_data, self.device)
        meta_data = _to_device_graph(meta_data, self.device)
        target = target.to(self.device).float()
        return processed_features, dis_data, meta_data, target

    def _cache_backbone(self, processed_features, dis_data, meta_data):
        residual = self.model.encode_backbone(processed_features, dis_data, meta_data)
        policy_input = self.model.build_policy_input(residual)
        return residual, policy_input

    def _clip_scalar(self, value: Tensor, clip_value: Optional[float]) -> Tensor:
        if clip_value is None:
            return value
        return value.clamp(min=-clip_value, max=clip_value)

    @torch.no_grad()
    def collect_rollouts(self, processed_features, dis_data, meta_data, target):
        self.model.eval()

        processed_features, dis_data, meta_data, target = self._prepare_inputs(
            processed_features, dis_data, meta_data, target
        )
        residual, policy_input = self._cache_backbone(processed_features, dis_data, meta_data)

        storage: List[RolloutItem] = []
        reward_rows = []
        std_means = []
        std_mins = []
        std_maxs = []

        for _ in range(self.n_rollouts):
            pred, _, _, rl_info = self.model.forward_from_encoded(
                residual=residual,
                policy_input=policy_input,
                return_rl=True,
                rl_std_scale=self.rl_std_scale,
            )

            reward_vector = compute_reward_vector(pred.detach(), target.detach(), self.objectives)
            reward_rows.append(reward_vector)
            std_tensor = rl_info["latent_std"].detach()
            std_means.append(std_tensor.mean())
            std_mins.append(std_tensor.min())
            std_maxs.append(std_tensor.max())

            storage.append(
                RolloutItem(
                    latent_action=rl_info["latent_action"].detach(),
                    old_log_prob=rl_info["latent_log_prob"].detach(),
                    old_mean=rl_info["latent_mean"].detach(),
                    old_std=rl_info["latent_std"].detach(),
                    reward_vector=reward_vector.detach(),
                )
            )

        reward_matrix = torch.stack(reward_rows, dim=0)
        reward_matrix = torch.nan_to_num(reward_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        if self.reward_clip is not None:
            reward_matrix = reward_matrix.clamp(-self.reward_clip, self.reward_clip)

        weights = self.weight_controller.get_weights(reward_matrix)

        reward_for_advantage = reward_matrix
        if self.normalize_objective_rewards:
            reward_for_advantage = (
                reward_matrix - reward_matrix.mean(dim=0, keepdim=True)
            ) / (reward_matrix.std(dim=0, unbiased=False, keepdim=True) + self.eps)

        scalar_rewards = (reward_for_advantage * weights.unsqueeze(0)).sum(dim=-1)
        scalar_rewards = torch.nan_to_num(scalar_rewards, nan=0.0, posinf=0.0, neginf=0.0)

        advantages = (scalar_rewards - scalar_rewards.mean()) / (scalar_rewards.std(unbiased=False) + self.eps)
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        advantages = self._clip_scalar(advantages, self.advantage_clip)

        for idx, item in enumerate(storage):
            item.scalar_reward = scalar_rewards[idx].detach()
            item.advantage = advantages[idx].detach()

        rollout_stats = {
            "std_mean": float(torch.stack(std_means).mean().item()),
            "std_min": float(torch.stack(std_mins).min().item()),
            "std_max": float(torch.stack(std_maxs).max().item()),
        }
        return storage, reward_matrix.detach(), weights.detach(), rollout_stats

    def train_step(self, processed_features, dis_data, meta_data, target):
        processed_features, dis_data, meta_data, target = self._prepare_inputs(
            processed_features, dis_data, meta_data, target
        )
        rollouts, reward_matrix, weights, rollout_stats = self.collect_rollouts(
            processed_features, dis_data, meta_data, target
        )

        last_loss = None
        last_policy = None
        last_sup = None
        last_entropy = None
        last_kl = None
        ppo_epochs_ran = 0
        early_stopped = False

        self.model.eval()

        for _ in range(self.ppo_epochs):
            residual, policy_input = self._cache_backbone(processed_features, dis_data, meta_data)

            total_losses = []
            policy_losses = []
            supervised_losses = []
            entropy_terms = []
            kl_terms = []

            for item in rollouts:
                pred, _, _, rl_info = self.model.forward_from_encoded(
                    residual=residual,
                    policy_input=policy_input,
                    latent_action=item.latent_action,
                    return_rl=True,
                    rl_std_scale=self.rl_std_scale,
                )

                new_log_prob = rl_info["latent_log_prob"]
                log_ratio = torch.clamp(
                    new_log_prob - item.old_log_prob,
                    min=-self.ratio_log_clip,
                    max=self.ratio_log_clip,
                )
                ratio = torch.exp(log_ratio)

                advantage = torch.as_tensor(item.advantage, device=ratio.device, dtype=ratio.dtype)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()

                supervised_loss = self.criterion(pred, target)
                entropy_bonus = rl_info["latent_entropy"].mean()
                kl_loss = gaussian_kl(
                    old_mean=item.old_mean,
                    old_std=item.old_std,
                    new_mean=rl_info["latent_mean"],
                    new_std=rl_info["latent_std"],
                    eps=self.eps,
                )

                total_loss = (
                    self.supervised_coef * supervised_loss
                    + self.rl_coef * policy_loss
                    + self.kl_coef * kl_loss
                    - self.entropy_coef * entropy_bonus
                )

                total_losses.append(total_loss)
                policy_losses.append(policy_loss.detach())
                supervised_losses.append(supervised_loss.detach())
                entropy_terms.append(entropy_bonus.detach())
                kl_terms.append(kl_loss.detach())

            if len(total_losses) == 0:
                break

            loss = torch.stack(total_losses).mean()
            if not torch.isfinite(loss):
                early_stopped = True
                break

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            ppo_epochs_ran += 1
            last_loss = loss.detach()
            last_policy = torch.stack(policy_losses).mean()
            last_sup = torch.stack(supervised_losses).mean()
            last_entropy = torch.stack(entropy_terms).mean()
            last_kl = torch.stack(kl_terms).mean()

            if float(last_kl.item()) > self.target_kl:
                early_stopped = True
                break

        with torch.no_grad():
            self.model.eval()
            residual, policy_input = self._cache_backbone(processed_features, dis_data, meta_data)
            pred_det, _, _ = self.model.predict_from_encoded(
                residual=residual,
                policy_input=policy_input,
                rl_std_scale=self.rl_std_scale,
            )
            det_reward = compute_reward_vector(pred_det, target, self.objectives)

        return {
            "loss": float(last_loss.item()) if last_loss is not None else 0.0,
            "policy_loss": float(last_policy.item()) if last_policy is not None else 0.0,
            "supervised_loss": float(last_sup.item()) if last_sup is not None else 0.0,
            "entropy": float(last_entropy.item()) if last_entropy is not None else 0.0,
            "kl": float(last_kl.item()) if last_kl is not None else 0.0,
            "reward_mean_per_objective": reward_matrix.mean(dim=0).cpu().tolist(),
            "reward_std_per_objective": reward_matrix.std(dim=0, unbiased=False).cpu().tolist(),
            "objective_weights": weights.cpu().tolist(),
            "deterministic_reward": det_reward.cpu().tolist(),
            "ppo_epochs_ran": ppo_epochs_ran,
            "early_stopped": early_stopped,
            "policy_std_mean": rollout_stats["std_mean"],
            "policy_std_min": rollout_stats["std_min"],
            "policy_std_max": rollout_stats["std_max"],
        }
