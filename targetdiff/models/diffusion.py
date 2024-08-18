import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean


from models.molopt_score_model import (
    get_beta_schedule,
    cosine_beta_schedule,
    to_torch_const,
    center_pos,
    extract,
    log_1_min_a,
    log_add_exp,
    index_to_log_onehot,
    log_onehot_to_index,
    categorical_kl,
    log_categorical,
    normal_kl,
    log_normal,
    log_sample_categorical,
    ScorePosNet3D,
)
from models.classifier import TimeClassifier, MoleBundle


# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# Criterion
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return F.mse_loss(pred, target).sqrt()

# Model
class Diffusion(nn.Module):
    """
    diffusion variables
    1. sample_time
    2. perturb_ligand
    3. get_classifier_loss
    """

    def __init__(self, config, epsilon: TimeClassifier,):
        super().__init__()
        self.config = config

        self.sample_time_method = (
            config.sample_time_method
        )  # ['importance', 'symmetric']

        # region diffusion variables
        if config.beta_schedule == "cosine":
            alphas = (
                cosine_beta_schedule(config.num_diffusion_timesteps, config.pos_beta_s)
                ** 2
            )
            # print('cosine pos alpha schedule applied!')
            betas = 1.0 - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(
            np.sqrt(1.0 - alphas_cumprod)
        )
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1.0 / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(
            np.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_c0_coef = to_torch_const(
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_ct_coef = to_torch_const(
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_var = to_torch_const(posterior_variance)
        self.posterior_logvar = to_torch_const(
            np.log(np.append(self.posterior_var[1], self.posterior_var[1:]))
        )

        # atom type diffusion schedule in log space
        if config.v_beta_schedule == "cosine":
            alphas_v = cosine_beta_schedule(self.num_timesteps, config.v_beta_s)
            # print('cosine v alpha schedule applied!')
        else:
            raise NotImplementedError
        log_alphas_v = np.log(alphas_v)
        log_alphas_cumprod_v = np.cumsum(log_alphas_v)
        self.log_alphas_v = to_torch_const(log_alphas_v)
        self.log_one_minus_alphas_v = to_torch_const(log_1_min_a(log_alphas_v))
        self.log_alphas_cumprod_v = to_torch_const(log_alphas_cumprod_v)
        self.log_one_minus_alphas_cumprod_v = to_torch_const(
            log_1_min_a(log_alphas_cumprod_v)
        )
        # endregion diffusion variables

        self.register_buffer("Lt_history", torch.zeros(self.num_timesteps))
        self.register_buffer("Lt_count", torch.zeros(self.num_timesteps))

        # model definition
        self.model_mean_type = config.model_mean_type  # ['C0', 'noise']
        self.hidden_dim = config.hidden_dim
        ligand_atom_feature_dim = epsilon.num_classes
        self.num_classes = ligand_atom_feature_dim
        if self.config.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # center pos
        self.center_pos_mode = config.center_pos_mode  # ['none', 'protein']

        # time embedding
        self.time_emb_dim = config.time_emb_dim
        self.time_emb_mode = config.time_emb_mode  # ['simple', 'sin']
        if self.time_emb_dim > 0:
            if self.time_emb_mode == "simple":
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + 1, emb_dim)
            elif self.time_emb_mode == "sin":
                self.time_emb = nn.Sequential(
                    SinusoidalPosEmb(self.time_emb_dim),
                    nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.time_emb_dim * 4, self.time_emb_dim),
                )
                self.ligand_atom_emb = nn.Linear(
                    ligand_atom_feature_dim + self.time_emb_dim, emb_dim
                )
            else:
                raise NotImplementedError
        else:
            self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        self.epsilon = epsilon
        self.criterion = RMSELoss()

    # region diffusion operations
    # atom type diffusion process
    def q_v_pred_one_timestep(self, log_vt_1, t, batch):
        # q(vt | vt-1)
        log_alpha_t = extract(self.log_alphas_v, t, batch)
        log_1_min_alpha_t = extract(self.log_one_minus_alphas_v, t, batch)

        # alpha_t * vt + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_vt_1 + log_alpha_t, log_1_min_alpha_t - np.log(self.num_classes)
        )
        return log_probs

    def q_v_pred(self, log_v0, t, batch):
        # compute q(vt | v0)
        log_cumprod_alpha_t = extract(self.log_alphas_cumprod_v, t, batch)
        log_1_min_cumprod_alpha = extract(self.log_one_minus_alphas_cumprod_v, t, batch)

        log_probs = log_add_exp(
            log_v0 + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - np.log(self.num_classes),
        )
        return log_probs

    def q_v_sample(self, log_v0, t, batch):
        log_qvt_v0 = self.q_v_pred(log_v0, t, batch)
        sample_index = log_sample_categorical(log_qvt_v0)
        log_sample = index_to_log_onehot(sample_index, self.num_classes)
        return sample_index, log_sample

    # atom type generative process
    def q_pos_sample(self, ligand_pos, t, batch_ligand):
        # compute q(x_t|x_0)
        a = self.alphas_cumprod.index_select(0, t)

        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()  # pos_noise ~ N(0, I)
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        ligand_pos_perturbed = (
            a_pos.sqrt() * ligand_pos + (1.0 - a_pos).sqrt() * pos_noise
        )  # pos_noise * std
        return ligand_pos_perturbed, pos_noise

    def q_v_posterior(self, log_v0, log_vt, t, batch):
        # q(vt-1 | vt, v0) = q(vt | vt-1, x0) * q(vt-1 | x0) / q(vt | x0)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_qvt1_v0 = self.q_v_pred(log_v0, t_minus_1, batch)
        unnormed_logprobs = log_qvt1_v0 + self.q_v_pred_one_timestep(log_vt, t, batch)
        log_vt1_given_vt_v0 = unnormed_logprobs - torch.logsumexp(
            unnormed_logprobs, dim=-1, keepdim=True
        )
        return log_vt1_given_vt_v0

    def kl_v_prior(self, log_x_start, batch):
        num_graphs = batch.max().item() + 1
        log_qxT_prob = self.q_v_pred(
            log_x_start, t=[self.num_timesteps - 1] * num_graphs, batch=batch
        )
        log_half_prob = -torch.log(self.num_classes * torch.ones_like(log_qxT_prob))
        kl_prior = categorical_kl(log_qxT_prob, log_half_prob)
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_e = (
            extract(self.sqrt_recip_alphas_cumprod, t, batch) * xt
            - extract(self.sqrt_recipm1_alphas_cumprod, t, batch) * eps
        )
        return pos0_from_e

    def q_pos_posterior(self, x0, xt, t, batch):
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        pos_model_mean = (
            extract(self.posterior_mean_c0_coef, t, batch) * x0
            + extract(self.posterior_mean_ct_coef, t, batch) * xt
        )
        return pos_model_mean

    def kl_pos_prior(self, pos0, batch):
        num_graphs = batch.max().item() + 1
        a_pos = extract(
            self.alphas_cumprod, [self.num_timesteps - 1] * num_graphs, batch
        )  # (num_ligand_atoms, 1)
        pos_model_mean = a_pos.sqrt() * pos0
        pos_log_variance = torch.log((1.0 - a_pos).sqrt())
        kl_prior = normal_kl(
            torch.zeros_like(pos_model_mean),
            torch.zeros_like(pos_log_variance),
            pos_model_mean,
            pos_log_variance,
        )
        kl_prior = scatter_mean(kl_prior, batch, dim=0)
        return kl_prior

    def compute_pos_Lt(self, pos_model_mean, x0, xt, t, batch):
        # fixed pos variance
        pos_log_variance = extract(self.posterior_logvar, t, batch)
        pos_true_mean = self.q_pos_posterior(x0=x0, xt=xt, t=t, batch=batch)
        kl_pos = normal_kl(
            pos_true_mean, pos_log_variance, pos_model_mean, pos_log_variance
        )
        kl_pos = kl_pos / np.log(2.0)

        decoder_nll_pos = -log_normal(
            x0, means=pos_model_mean, log_scales=0.5 * pos_log_variance
        )
        assert kl_pos.shape == decoder_nll_pos.shape
        mask = (t == 0).float()[batch]
        loss_pos = scatter_mean(
            mask * decoder_nll_pos + (1.0 - mask) * kl_pos, batch, dim=0
        )
        return loss_pos

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float()[batch]
        loss_v = scatter_mean(mask * decoder_nll_v + (1.0 - mask) * kl_v, batch, dim=0)
        return loss_v
    # endregion diffusion operations


    def sample_time(self, num_graphs, device, method):
        if method == "importance":
            if not (self.Lt_count > 10).all():
                return self.sample_time(num_graphs, device, method="symmetric")

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            time_step = torch.multinomial(
                pt_all, num_samples=num_graphs, replacement=True
            )
            pt = pt_all.gather(dim=0, index=time_step)
            return time_step, pt

        elif method == "symmetric":
            time_step = torch.randint(
                0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device
            )
            time_step = torch.cat(
                [time_step, self.num_timesteps - time_step - 1], dim=0
            )[:num_graphs]
            pt = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, pt

        else:
            raise ValueError


    def get_classifier_sample(
        self,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_protein: torch.Tensor,
        ligand_pos: torch.Tensor,
        ligand_v: torch.Tensor,
        batch_ligand: torch.Tensor,
    ):  # Tuple[Tensor, Tensor, Tensor]
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos,
            ligand_pos,
            batch_protein,
            batch_ligand,
            mode=self.center_pos_mode,
        )
        time_step, pt = self.sample_time(
            num_graphs, protein_pos.device, self.sample_time_method
        )  # (num_graphs, )
        (
            ligand_pos_perturbed,
            log_ligand_v0,
            ligand_v_perturbed,
            log_ligand_vt,
            pos_noise,
        ) = self.perturb_ligand(ligand_pos, ligand_v, batch_ligand, time_step)
        return ligand_pos_perturbed, ligand_v_perturbed, time_step

    def perturb_ligand(self, ligand_pos, ligand_v, batch_ligand, time_step):
        ligand_pos_perturbed, pos_noise = self.q_pos_sample(
            ligand_pos, time_step, batch_ligand
        )
        # Vt = a * V0 + (1-a) / K
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.q_v_sample(
            log_ligand_v0, time_step, batch_ligand
        )
        return (
            ligand_pos_perturbed,
            log_ligand_v0,
            ligand_v_perturbed,
            log_ligand_vt,
            pos_noise,
        )

    def get_classifier_loss(
        self,
        protein_pos: torch.Tensor,
        protein_v: torch.Tensor,
        batch_protein: torch.Tensor,
        ligand_pos: torch.Tensor,
        ligand_v: torch.Tensor,
        batch_ligand: torch.Tensor,
        return_acc: bool = True,
    ):
        # num_graphs = batch_protein.max().item() + 1 # batch size
        if len(batch_ligand.shape) != 1:
            batch_ligand.squeeze_(-1)

        ligand_pos_perturbed, ligand_v_perturbed, time_step = (
            self.get_classifier_sample(
                protein_pos,
                protein_v,
                batch_protein,
                ligand_pos,
                ligand_v,
                batch_ligand,
            )
        )

        tau:torch.Tensor = self.epsilon(
            MoleBundle(protein_pos, protein_v, batch_protein),
            MoleBundle(ligand_pos_perturbed, ligand_v_perturbed, batch_ligand),
        ) # (batch_size, 1)

        preds = tau.clone().detach().int().view(-1)
        abs_dist = (preds-time_step).abs()
        pred_len = preds.shape[0]
        acc_0 = (abs_dist == 0).sum()/pred_len
        acc_5 = (abs_dist < 5).sum()/pred_len
        acc_10 = (abs_dist < 10).sum()/pred_len
        acc_25 = (abs_dist < 25).sum()/pred_len
        acc_100 = (abs_dist < 100).sum()/pred_len
        acc = {
            "acc_0": acc_0.item(),
            "acc_5": acc_5.item(),
            "acc_10": acc_10.item(),
            "acc_25": acc_25.item(),
            "acc_100": acc_100.item(),
        }

        if isinstance(self.criterion, (nn.MSELoss, RMSELoss)):
            tau.squeeze_(-1)
            time_step = time_step.to(dtype=tau.dtype)

        if return_acc:
            return self.criterion(tau, time_step), acc
        else:
            return self.criterion(tau, time_step)

    forward = get_classifier_loss

    def get_sample_timestep(
            self, protein: MoleBundle, ligand: MoleBundle, time_range:tuple,
            ):
        # pipeline sample process
        batch_size:int = protein.batch.max().int().item() + 1
        commence = reversed(range(self.num_timesteps - time_range[0], self.num_timesteps)) # launch time
        # three parts of time sequence
        # 1. smoothly start from T
        for i in commence:
            t = torch.full(size=(batch_size,), fill_value=i, dtype=torch.long, device=protein.pos.device)
            yield t

        # 2. clf function area
        last_pred = t
        while True:
            rt_mask = last_pred <= time_range[1] # tensor
            if rt_mask.all(): break

            tau:torch.Tensor = self.epsilon(protein, ligand,).round() # (batch_size, 1)
            if not self.epsilon.fast_forward:
                tau = tau.where(tau >= last_pred, last_pred)
            timestep = tau.where(rt_mask, last_pred - 1) # ignore negtive case (dealed by outsider)
            yield timestep
            last_pred = timestep

        # 3. smoothly end to 0
        complete = reversed(range(last_pred.max().item())) # remaining time for ending smoothly
        for i in complete:
            last_pred -= 1
            yield last_pred

    @torch.inference_mode()
    def sample_diffusion(
        self,
        model:ScorePosNet3D,
        protein_pos, protein_v, batch_protein,
        ligand_pos, ligand_v, batch_ligand,
        time_range:tuple,
        num_steps:int, center_pos_mode="protein",
        ) -> dict:
        # code ported from ScorePosNet3D#sample_diffusion
        num_graphs = batch_protein.max().item() + 1

        protein_pos, ligand_pos, offset = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode)

        pos_traj, v_traj = [], []
        v0_pred_traj, vt_pred_traj = [], []
        steps_counter = torch.zeros((num_graphs,), dtype=torch.long, device=protein_pos.device)
        fake_step = torch.zeros((num_graphs,), dtype=torch.long, device=protein_pos.device)
        real_step = torch.ones((num_graphs,), dtype=torch.long, device=protein_pos.device)
        # time sequence
        if num_steps == None:
            num_steps = self.num_timesteps
        last_time = torch.ones((num_graphs,), dtype=torch.long, device=protein_pos.device)
        for t in self.get_sample_timestep(protein=MoleBundle(protein_pos, protein_v, batch_protein), ligand=MoleBundle(ligand_pos, ligand_v, batch_ligand), time_range=time_range):
            tfilter = last_time <= 0
            if tfilter.all() or steps_counter.max() >= time_range[2]:
                break
            step_delta = torch.where(tfilter, real_step, fake_step)
            steps_counter += step_delta

            preds = model( # predict from ScorePosNet3D
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,

                init_ligand_pos=ligand_pos,
                init_ligand_v=ligand_v,
                batch_ligand=batch_ligand,
                time_step=t
            )
            # Compute posterior mean and variance
            if self.model_mean_type == 'noise':
                pred_pos_noise = preds['pred_ligand_pos'] - ligand_pos
                pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos, eps=pred_pos_noise, t=t, batch=batch_ligand)
                v0_from_e = preds['pred_ligand_v']
            elif self.model_mean_type == 'C0':
                pos0_from_e = preds['pred_ligand_pos']
                v0_from_e = preds['pred_ligand_v']
            else:
                raise ValueError

            # region ligand position
            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
            # no noise when t == 0
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                ligand_pos)
            nonzero_mask_bool = nonzero_mask.bool()
            ligand_pos = torch.where(nonzero_mask_bool, ligand_pos, ligand_pos_next) # [N_ligand, 3]
            # endregion ligand position

            # region ligand type
            log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
            log_ligand_v = index_to_log_onehot(ligand_v, self.num_classes)
            log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
            ligand_v_next = log_sample_categorical(log_model_prob)

            v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
            vt_pred_traj.append(log_model_prob.clone().cpu())
            ligand_v = torch.where(nonzero_mask_bool.squeeze(-1), ligand_v, ligand_v_next) # [N_ligand, ]
            # endregion ligand type

            ori_ligand_pos = ligand_pos + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())

        ligand_pos = ligand_pos + offset[batch_ligand]
        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj,
            'steps': steps_counter.clone().cpu(), # TODO: manual check for steps
        }