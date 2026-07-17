from __future__ import division
from __future__ import print_function

import argparse
import copy
import os
import random
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch_geometric.data import Data
from torch.nn.utils import clip_grad_norm_

from utils import (
    load_data,
    constructNet,
    laplacian_positional_encoding,
    re_features,
    PolynomialDecayLR,
    get_link_labels,
    plot_auc_curves,
    plot_prc_curves,
)
from model import NGR_SDMRL
from morl_rl_clean import MORLPPOTrainer, ObjectiveSpec
from metric import cv_model_evaluate


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def sample_negative_edges(candidate_edges: List[List[int]], num_samples: int, rng: random.Random,
                          device) -> torch.Tensor:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if len(candidate_edges) < num_samples:
        raise ValueError(f"Not enough negative candidates: need {num_samples}, got {len(candidate_edges)}")

    edges = list(candidate_edges)
    rng.shuffle(edges)
    sampled = np.asarray(edges[:num_samples]).T
    return torch.tensor(sampled, dtype=torch.long, device=device)


class EdgeSupervisionState:
    """
    Stores the currently sampled training edges for one epoch, and exposes
    masked supervised loss / MORL reward functions that operate on the full
    prediction matrix produced by NGR_SDMRL.
    """

    def __init__(self, device, reward_threshold=0.5, reward_k=0, eps=1e-8):
        self.device = device
        self.reward_threshold = reward_threshold
        self.default_reward_k = reward_k
        self.eps = eps

        self.edge_index = None
        self.labels = None
        self.target_matrix = None
        self.reward_k = None

    def update(self, train_pos_edge_index, train_neg_edge_index, num_meta, num_dis):
        self.edge_index = torch.cat([train_pos_edge_index, train_neg_edge_index], dim=1).to(self.device)
        self.labels = get_link_labels(train_pos_edge_index, train_neg_edge_index).to(self.device).float()

        target_matrix = torch.zeros((num_meta, num_dis), dtype=torch.float32, device=self.device)
        target_matrix[train_pos_edge_index[0], train_pos_edge_index[1]] = 1.0
        self.target_matrix = target_matrix

        if self.default_reward_k is None or self.default_reward_k <= 0:
            self.reward_k = int(train_pos_edge_index.size(1))
        else:
            self.reward_k = int(min(self.default_reward_k, self.edge_index.size(1)))
        self.reward_k = max(1, self.reward_k)
        return self.target_matrix

    def _scores(self, pred_matrix):
        if self.edge_index is None:
            raise RuntimeError("EdgeSupervisionState.update() must be called before loss/reward computation.")
        return pred_matrix[self.edge_index[0], self.edge_index[1]]

    def masked_bce_loss(self, pred_matrix, _target_matrix):
        scores = self._scores(pred_matrix)
        return F.binary_cross_entropy(scores, self.labels)

    @torch.no_grad()
    def binary_f1_reward(self, pred_matrix, _target_matrix):
        scores = self._scores(pred_matrix)
        labels = self.labels

        pred_bin = (scores >= self.reward_threshold).float()
        tp = (pred_bin * labels).sum()
        fp = (pred_bin * (1.0 - labels)).sum()
        fn = ((1.0 - pred_bin) * labels).sum()
        return (2.0 * tp) / (2.0 * tp + fp + fn + self.eps)

    @torch.no_grad()
    def recall_at_k_reward(self, pred_matrix, _target_matrix):
        scores = self._scores(pred_matrix).reshape(-1)
        labels = self.labels.reshape(-1)

        k = max(1, min(self.reward_k, scores.numel()))
        topk_idx = torch.topk(scores, k=k).indices
        hits = labels[topk_idx].sum()
        total_pos = labels.sum()
        return hits / (total_pos + self.eps)

    @torch.no_grad()
    def negative_bce_reward(self, pred_matrix, _target_matrix):
        scores = self._scores(pred_matrix).clamp(min=self.eps, max=1.0 - self.eps)
        labels = self.labels
        return -F.binary_cross_entropy(scores, labels)

    @torch.no_grad()
    def train_auc(self, pred_matrix):
        scores = self._scores(pred_matrix).detach().cpu().numpy()
        labels = self.labels.detach().cpu().numpy()
        try:
            return metrics.roc_auc_score(labels, scores)
        except ValueError:
            return 0.5


def build_morl_trainer(model, optimizer, scheduler, edge_state, args, device):
    objectives = [
        ObjectiveSpec(
            name="f1",
            fn=edge_state.binary_f1_reward,
            weight=args.reward_w_f1,
            direction="max",
        ),
        ObjectiveSpec(
            name="recall_at_k",
            fn=edge_state.recall_at_k_reward,
            weight=args.reward_w_recall,
            direction="max",
        ),
        ObjectiveSpec(
            name="negative_bce",
            fn=edge_state.negative_bce_reward,
            weight=args.reward_w_bce,
            direction="max",
        ),
    ]

    return MORLPPOTrainer(
        model=model,
        optimizer=optimizer,
        objectives=objectives,
        criterion=edge_state.masked_bce_loss,
        scheduler=scheduler,
        clip_epsilon=args.ppo_clip,
        ppo_epochs=args.ppo_epochs,
        n_rollouts=args.ppo_rollouts,
        supervised_coef=args.supervised_coef,
        rl_coef=args.rl_coef,
        entropy_coef=args.entropy_coef,
        kl_coef=args.kl_coef,
        max_grad_norm=args.max_grad_norm,
        dynamic_weight=(not args.disable_dynamic_weight),
        normalize_objective_rewards=(not args.disable_reward_norm),
        rl_std_scale=args.rl_std_scale,
        device=device,
    )


def distillation_step(
        model,
        distill_optimizer,
        processed_features,
        dis_data,
        meta_data,
        alpha_tsd: float = 1.0,
        beta_ssd: float = 1.0,
        max_grad_norm: float = 1.0,
        T_student: int = None,
        T_teacher: int = None,
) -> dict:
   
    model.train()

    distill_optimizer.zero_grad(set_to_none=True)

    loss, stats = model.compute_distillation_loss(
        processed_features=processed_features,
        dis_data=dis_data,
        meta_data=meta_data,
        alpha_tsd=alpha_tsd,
        beta_ssd=beta_ssd,
        T_student=T_student,
        T_teacher=T_teacher,
    )

    if torch.isfinite(loss) and loss.item() > 0:
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        distill_optimizer.step()
    else:
        stats["loss_distill_total"] = 0.0

    return stats


def build_distill_optimizer(model, lr: float = 5e-4, weight_decay: float = 1e-4):
   
    distill_params = []

    distill_params += list(model.convs.parameters())
    if hasattr(model.graph_input_proj, 'parameters'):
        distill_params += list(model.graph_input_proj.parameters())

    distill_params += list(model.att_embeddings_nope.parameters())
    distill_params += list(model.layers.parameters())
    distill_params += list(model.final_ln.parameters())
    distill_params += list(model.attn_layer.parameters())


    distill_params += list(model.spiking_encoder.parameters())


    if model.weak_decoder is not None:
        distill_params += list(model.weak_decoder.parameters())

    return torch.optim.Adam(distill_params, lr=lr, weight_decay=weight_decay)


def evaluate_fold(model, processed_features, dis_data, meta_data, Adj, val_pos_edge_index, val_neg_edge_index,
                  edge_state):
    model.eval()
    with torch.no_grad():
        predict_y_proba, _, _ = model.predict_deterministic(processed_features, dis_data, meta_data)
        predict_y_proba = predict_y_proba.reshape(Adj.shape[0], Adj.shape[1])

        train_auc = edge_state.train_auc(predict_y_proba)

        score_val, label_val, metric_tmp = cv_model_evaluate(
            predict_y_proba,
            val_pos_edge_index,
            val_neg_edge_index,
        )

        fpr, tpr, _ = metrics.roc_curve(label_val, score_val)
        precision, recall, _ = metrics.precision_recall_curve(label_val, score_val)
        val_auc = metrics.auc(fpr, tpr)
        val_prc = metrics.auc(recall, precision)

    return {
        "predict_y_proba": predict_y_proba,
        "train_auc": train_auc,
        "metric_tmp": metric_tmp,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "val_auc": val_auc,
        "val_prc": val_prc,
    }




start_time = time.time()
start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

# Original training parameters
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--epochs", type=int, default=700, help="Number of epochs to train.")
parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay.")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")                                 # 0.1
parser.add_argument("--tot_updates", type=int, default=0,
                    help="Total optimizer steps for LR scheduling. 0 means auto-compute.")
parser.add_argument("--warmup_updates", type=int, default=400, help="Warmup steps for LR scheduling.")
parser.add_argument("--peak_lr", type=float, default=0.002, help="Peak learning rate.")
parser.add_argument("--end_lr", type=float, default=0.0001, help="Final learning rate.")

# Original model parameters
parser.add_argument("--pe_dim", type=int, default=15, help="Position embedding size.")
parser.add_argument("--hops", type=int, default=7, help="Hop of neighbors to be calculated.")
parser.add_argument("--graphformer_layers", type=int, default=1, help="Number of Graphormer layers.")         
parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads.")
parser.add_argument("--node_input", type=int, default=64, help="Input dimensions of node features / PCA.")
parser.add_argument("--node_hidden", type=int, default=128, help="Hidden dimensions of node features.")
parser.add_argument("--node_output", type=int, default=64, help="Output dimensions of node features.")
parser.add_argument("--ffn_dim", type=int, default=256, help="FFN layer size.")
parser.add_argument("--GCNII_layers", type=int, default=20, help="Number of GCNII layers.")                   

# MORL / PPO parameters
parser.add_argument("--ppo_epochs", type=int, default=4, help="Number of PPO update passes per epoch.")
parser.add_argument("--ppo_rollouts", type=int, default=8, help="Number of stochastic rollouts collected per epoch.")
parser.add_argument("--ppo_clip", type=float, default=0.2, help="PPO clipping epsilon.")
parser.add_argument("--supervised_coef", type=float, default=1.0, help="Weight of masked BCE loss.")
parser.add_argument("--rl_coef", type=float, default=0.2 , help="Weight of PPO policy loss.")                  
parser.add_argument("--entropy_coef", type=float, default=1e-3, help="Entropy bonus weight.")
parser.add_argument("--kl_coef", type=float, default=1e-3, help="KL penalty weight.")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm.")
parser.add_argument("--rl_std_scale", type=float, default=1.25,                                                
                    help="Exploration scale for policy std during rollouts.")
parser.add_argument("--disable_dynamic_weight", action="store_true", help="Disable dynamic objective weighting.")
parser.add_argument("--disable_reward_norm", action="store_true", help="Disable objective-wise reward normalization.")

# MORL reward parameters
parser.add_argument("--reward_threshold", type=float, default=0.5, help="Threshold used by F1 reward.")
parser.add_argument("--reward_k", type=int, default=0,
                    help="Recall@K reward. 0 means use the number of positive training edges.")
parser.add_argument("--reward_w_f1", type=float, default=0.40, help="Weight prior for F1 reward.")
parser.add_argument("--reward_w_recall", type=float, default=0.35, help="Weight prior for Recall@K reward.")
parser.add_argument("--reward_w_bce", type=float, default=0.25, help="Weight prior for negative BCE reward.")


parser.add_argument("--snn_T_s", type=int, default=4,                               
                    help="SNN T_s")
parser.add_argument("--snn_T_t", type=int, default=8,
                    help="SNN T_t")
parser.add_argument("--snn_tau", type=float, default=2.0,
                    help="LIF_τ")
parser.add_argument("--snn_threshold", type=float, default=1.0,
                    help="LIF_ϑ")


parser.add_argument("--alpha_tsd", type=float, default=1.0,                                    
                    help="TSD  α")
parser.add_argument("--beta_ssd", type=float, default=1.0,                                     
                    help="SSD_β")
parser.add_argument("--distill_lr", type=float, default=5e-4,
                    help="Self distillation optimizer learning rate")
parser.add_argument("--distill_weight_decay", type=float, default=1e-4,
                    help="Self distillation optimizer weight attenuation")
parser.add_argument("--disable_distillation", action="store_true",
                    help="Completely disable self distillation")
parser.add_argument("--distill_warmup", type=int, default=50,
                    help="Number of self distillation preheating rounds")
parser.add_argument("--stochastic_latency", action="store_true",
                    help="Enable random delay training")

args, unknown = parser.parse_known_args()
print("args", args)

seed_everything(args.seed)
os.makedirs("../result", exist_ok=True)

Adj, Dis_adj, Meta_adj, feature, random_index, k_folds, Dis_feature, Meta_feature, Dis_simi, Meta_simi = load_data(
    args.seed,
    args.node_input,
)

feature = feature.to(device)
all_negative_candidates = np.argwhere(np.asarray(Adj) < 1).tolist()

auc_result = []
acc_result = []
pre_result = []
recall_result = []
f1_result = []
prc_result = []
fprs = []
tprs = []
precisions = []
recalls = []

print("seed=%d, evaluating metabolite-disease with NGR_SDMRL + MORL + SNN + TSSD..." % args.seed)

for k in range(k_folds):
    print("------this is %dth cross validation------" % (k + 1))

    fold_rng = random.Random(args.seed + k)

    or_train = np.matrix(Adj, copy=True)
    val_pos_edge_index = np.array(random_index[k]).T
    val_pos_edge_index = torch.tensor(val_pos_edge_index, dtype=torch.long, device=device)

    val_neg_edge_index = sample_negative_edges(
        candidate_edges=all_negative_candidates,
        num_samples=val_pos_edge_index.shape[1],
        rng=fold_rng,
        device=device,
    )

    or_train[tuple(np.array(random_index[k]).T)] = 0
    train_pos_edge_index = np.mat(np.where(or_train > 0))
    train_pos_edge_index = torch.tensor(train_pos_edge_index, dtype=torch.long, device=device)

    or_train_matrix = np.matrix(Adj, copy=True)
    or_train_matrix[tuple(np.array(random_index[k]).T)] = 0
    or_adj = constructNet(torch.tensor(or_train_matrix, dtype=torch.float32)).to(device)

    lpe = laplacian_positional_encoding(or_adj, args.pe_dim).to(device)
    features = torch.cat((feature, lpe), dim=1)

    Dis_network = torch.nonzero(Dis_adj, as_tuple=True)
    Dis_network = torch.stack(Dis_network)
    dis_data = Data(x=feature[:Adj.shape[1], :], edge_index=Dis_network)

    Meta_network = torch.nonzero(Meta_adj, as_tuple=True)
    Meta_network = torch.stack(Meta_network)
    meta_data = Data(x=feature[Adj.shape[1]:, :], edge_index=Meta_network)

    processed_features = re_features(or_adj, features, args.hops).to(device)


    model = NGR_SDMRL(
        hops=args.hops,
        output_dim=args.node_output,
        input_dim=features.shape[1],
        pe_dim=args.pe_dim,
        num_dis=Adj.shape[1],
        num_meta=Adj.shape[0],
        graphformer_layers=args.graphformer_layers,
        num_heads=args.n_heads,
        hidden_dim=args.node_hidden,
        ffn_dim=args.ffn_dim,
        dropout_rate=args.dropout,
        GCNII_layers=args.GCNII_layers,
        graph_input_dim=feature.shape[1],
        # 新增参数
        snn_T_s=args.snn_T_s,
        snn_T_t=args.snn_T_t,
        snn_tau=args.snn_tau,
        snn_threshold=args.snn_threshold,
        enable_distillation=(not args.disable_distillation),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    spiking_params = sum(p.numel() for p in model.spiking_encoder.parameters())
    weak_dec_params = sum(p.numel() for p in model.weak_decoder.parameters()) if model.weak_decoder else 0
    print(f"total params: {total_params} (spiking_encoder: {spiking_params}, weak_decoder: {weak_dec_params})")


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)

    effective_tot_updates = args.tot_updates
    if effective_tot_updates <= 0:
        effective_tot_updates = max(1, args.epochs * args.ppo_epochs)

    effective_warmup = min(args.warmup_updates, max(0, effective_tot_updates - 1))
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=effective_warmup,
        tot_updates=effective_tot_updates,
        lr=args.peak_lr,
        end_lr=args.end_lr,
        power=1.0,
    )


    distill_optimizer = build_distill_optimizer(
        model, lr=args.distill_lr, weight_decay=args.distill_weight_decay
    )

    edge_state = EdgeSupervisionState(
        device=device,
        reward_threshold=args.reward_threshold,
        reward_k=args.reward_k,
    )
    trainer = build_morl_trainer(
        model=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        edge_state=edge_state,
        args=args,
        device=device,
    )

    best_epoch = 0
    best_auc = -1.0
    best_acc = -1.0
    best_prc = -1.0
    best_tpr = 0
    best_fpr = 0
    best_recall_curve = 0
    best_precision_curve = 0
    best_model_state = None
    metric_tmp_best = [0.0, 0.0, 0.0, 0.0]

    for epoch in range(args.epochs):
        start = time.time()
        epoch_rng = random.Random(args.seed + 1000 * (k + 1) + epoch)

        train_neg_edge_index = sample_negative_edges(
            candidate_edges=all_negative_candidates,
            num_samples=train_pos_edge_index.shape[1],
            rng=epoch_rng,
            device=device,
        )

        target_matrix = edge_state.update(
            train_pos_edge_index=train_pos_edge_index,
            train_neg_edge_index=train_neg_edge_index,
            num_meta=Adj.shape[0],
            num_dis=Adj.shape[1],
        )

      
        stats = trainer.train_step(
            processed_features=processed_features,
            dis_data=dis_data,
            meta_data=meta_data,
            target=target_matrix,
        )

  
        distill_stats = {"loss_tsd": 0.0, "loss_ssd": 0.0, "loss_distill_total": 0.0}

        do_distill = (
                not args.disable_distillation
                and epoch >= args.distill_warmup  
        )

        if do_distill:
        
            if args.stochastic_latency:
                T_student = random.randint(1, args.snn_T_s)
                T_teacher = 2 * T_student
            else:
                T_student = args.snn_T_s
                T_teacher = args.snn_T_t

            distill_stats = distillation_step(
                model=model,
                distill_optimizer=distill_optimizer,
                processed_features=processed_features,
                dis_data=dis_data,
                meta_data=meta_data,
                alpha_tsd=args.alpha_tsd,
                beta_ssd=args.beta_ssd,
                max_grad_norm=args.max_grad_norm,
                T_student=T_student,
                T_teacher=T_teacher,
            )

    
        eval_stats = evaluate_fold(
            model=model,
            processed_features=processed_features,
            dis_data=dis_data,
            meta_data=meta_data,
            Adj=Adj,
            val_pos_edge_index=val_pos_edge_index,
            val_neg_edge_index=val_neg_edge_index,
            edge_state=edge_state,
        )

        metric_tmp = eval_stats["metric_tmp"]
        val_auc = eval_stats["val_auc"]
        val_prc = eval_stats["val_prc"]

        end = time.time()
        print(
            "Epoch:", epoch + 1,
                      "Loss: %.4f" % stats["loss"],
                      "SupLoss: %.4f" % stats["supervised_loss"],
                      "PolicyLoss: %.4f" % stats["policy_loss"],
                      "Entropy: %.4f" % stats["entropy"],
                      "KL: %.4f" % stats["kl"],
     
                      "TSD: %.4f" % distill_stats["loss_tsd"],
                      "SSD: %.4f" % distill_stats["loss_ssd"],
                      "Acc: %.4f" % metric_tmp[0],
                      "Pre: %.4f" % metric_tmp[1],
                      "Recall: %.4f" % metric_tmp[2],
                      "F1: %.4f" % metric_tmp[3],
                      "Train AUC: %.4f" % eval_stats["train_auc"],
                      "Val AUC: %.4f" % val_auc,
                      "Val PRC: %.4f" % val_prc,
            "RewardMean:", [round(v, 4) for v in stats["reward_mean_per_objective"]],
            "ObjW:", [round(v, 4) for v in stats["objective_weights"]],
                      "Time: %.2f" % (end - start),
        )

        if (metric_tmp[0] > best_acc and val_auc > best_auc and val_prc > best_prc) or best_model_state is None:
            metric_tmp_best = metric_tmp
            best_auc = val_auc
            best_prc = val_prc
            best_acc = metric_tmp[0]
            best_epoch = epoch + 1
            best_tpr = eval_stats["tpr"]
            best_fpr = eval_stats["fpr"]
            best_recall_curve = eval_stats["recall"]
            best_precision_curve = eval_stats["precision"]
            best_model_state = copy.deepcopy(model.state_dict())

    print(
        "Fold:", k + 1,
        "Best Epoch:", best_epoch,
                 "Val acc: %.4f" % metric_tmp_best[0],
                 "Val Pre: %.4f" % metric_tmp_best[1],
                 "Val Recall: %.4f" % metric_tmp_best[2],
                 "Val F1: %.4f" % metric_tmp_best[3],
                 "Val AUC: %.4f" % best_auc,
                 "Val PRC: %.4f" % best_prc,
    )

    model_save_path = f"../result/model_fold_{k + 1}.pth"
    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
        print(f"Best model for fold {k + 1} saved to {model_save_path}")

    acc_result.append(metric_tmp_best[0])
    pre_result.append(metric_tmp_best[1])
    recall_result.append(metric_tmp_best[2])
    f1_result.append(metric_tmp_best[3])
    auc_result.append(round(best_auc, 4))
    prc_result.append(round(best_prc, 4))

    fprs.append(best_fpr)
    tprs.append(best_tpr)
    recalls.append(best_recall_curve)
    precisions.append(best_precision_curve)

print("## Training Finished !")
print("-----------------------------------------------------------------------------------------------")
print("Acc", acc_result)
print("Pre", pre_result)
print("Recall", recall_result)
print("F1", f1_result)
print("Auc", auc_result)
print("Prc", prc_result)
print(
    "AUC mean: %.4f, variance: %.4f \n" % (np.mean(auc_result), np.std(auc_result)),
    "Accuracy mean: %.4f, variance: %.4f \n" % (np.mean(acc_result), np.std(acc_result)),
    "Precision mean: %.4f, variance: %.4f \n" % (np.mean(pre_result), np.std(pre_result)),
    "Recall mean: %.4f, variance: %.4f \n" % (np.mean(recall_result), np.std(recall_result)),
    "F1-score mean: %.4f, variance: %.4f \n" % (np.mean(f1_result), np.std(f1_result)),
    "PRC mean: %.4f, variance: %.4f \n" % (np.mean(prc_result), np.std(prc_result)),
)

plot_auc_curves(fprs, tprs, auc_result, directory="../result", name="test_auc")
plot_prc_curves(precisions, recalls, prc_result, directory="../result", name="test_prc")

end_time = time.time()
total_time = end_time - start_time
minutes = int(total_time / 60)
seconds = total_time - minutes * 60

with open("../result/NGR_SDMRL.txt", mode="a+", encoding="utf-8") as file:
    file.write(f"Start time: {start_str}\n")
    file.write(f"args:{args}\n")
    file.write(f"Acc:{acc_result}\n")
    file.write(f"Pre:{pre_result}\n")
    file.write(f"Recall:{recall_result}\n")
    file.write(f"F1:{f1_result}\n")
    file.write(f"Auc:{auc_result}\n")
    file.write(f"Prc:{prc_result}\n")
    file.write(f"AUC mean:{np.mean(auc_result)}, variance:{np.std(auc_result)}\n")
    file.write(f"Accuracy mean:{np.mean(acc_result)}, variance:{np.std(acc_result)}\n")
    file.write(f"Precision mean:{np.mean(pre_result)}, variance:{np.std(pre_result)}\n")
    file.write(f"Recall mean:{np.mean(recall_result)}, variance:{np.std(recall_result)}\n")
    file.write(f"F1-score mean:{np.mean(f1_result)}, variance:{np.std(f1_result)}\n")
    file.write(f"PRC mean:{np.mean(prc_result)}, variance:{np.std(prc_result)}\n")
    file.write(f"total running time is: {total_time}s\n")
    file.write(f"total running time is: {minutes}min {seconds:.2f}sec\n\n\n")

