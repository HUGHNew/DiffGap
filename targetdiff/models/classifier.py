import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import ShiftedSoftplus, MoleBundle
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral
import utils.misc as misc

# FIXME: dataset evaluation error [14, 21, 54, 57, 87, 92]


def compose_context(
    h_protein, pos_protein, batch_protein, h_ligand, pos_ligand, batch_ligand
):
    """Concatenate protein and ligand context."""

    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)
    # sort_idx = batch_ctx.argsort()
    sort_idx = torch.sort(batch_ctx, stable=True).indices

    mask_ligand = torch.cat(
        [
            torch.zeros([batch_protein.size(0)], device=batch_protein.device).bool(),
            torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool(),
        ],
        dim=0,
    )[sort_idx]

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[
        sort_idx
    ]  # (N_protein+N_ligand, H=128)
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[
        sort_idx
    ]  # (N_protein+N_ligand, 3)

    return h_ctx, pos_ctx, batch_ctx, mask_ligand


class MultiHeadAttentionLayer(nn.Module):
    src_pad_idx = 0

    def __init__(self, hid_dim: int, n_heads: int, dropout: float):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ):
        # query=key=value = [len, hid]

        batch_size = query.shape[0]

        Q = self.fc_q(query)  # [len, hid]
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        Q = Q.view(batch_size, self.n_heads, -1, self.head_dim)
        # [batch_size, n_heads, len, head_dim]

        Q = Q.view(-1, self.n_heads, self.head_dim)  # [len, num_head, head_dim]
        K = K.view(-1, self.n_heads, self.head_dim)
        V = V.view(-1, self.n_heads, self.head_dim)

        x = F.scaled_dot_product_attention(Q, K, V, mask, self.dropout)
        return self.fc_o(x)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x

    def make_src_mask(self, src):

        # src = [len]

        src_mask = (
            (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        )  # src_mask = [1, 1, len]

        return src_mask


class ContextEncoder(nn.Module):
    POS_DIM = 3
    PRO_OUT_DIM = 2
    LIG_OUT_DIM = 2

    def __init__(
        self,
        protein_atom_feature_dim: int,
        protein_compress_dim=1,
        hidden=64,
        attn_hid=0,
    ):
        super().__init__()
        self.protein_compressor = nn.Linear(
            protein_atom_feature_dim, protein_compress_dim
        )
        # TODO: hidden modification and attention test
        self.enable_attn = attn_hid > 0
        if self.enable_attn:
            self.protein_attn = MultiHeadAttentionLayer(attn_hid, 4, 0.5)
            self.ligand_attn = MultiHeadAttentionLayer(attn_hid, 4, 0.5)
        self.protein_mlp = nn.Linear(
            self.POS_DIM + protein_compress_dim, self.PRO_OUT_DIM
        )
        self.ligand_mlp = nn.Linear(self.POS_DIM + 1, self.LIG_OUT_DIM)
        self.out_net = nn.Sequential(
            nn.Linear(self.PRO_OUT_DIM + self.LIG_OUT_DIM, hidden),  # cat([x, h])
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, 1),
        )

    def forward(self, protein: MoleBundle, ligand: MoleBundle) -> torch.Tensor:
        batch_size = int(protein.batch.max().item() + 1)  # make type checker happy

        # ligand process
        ligand_v = ligand.v.argmax(1).unsqueeze(-1).float()
        ligand_cat = torch.cat([ligand.pos, ligand_v], dim=-1)  # (N_ligand, 4)
        ligand_compose: torch.Tensor = self.ligand_mlp(
            ligand_cat
        )  # (N_ligand, LIG_OUT_DIM)
        ligand_compose /= ligand_compose.shape[0]  # normal
        ligand_batch: torch.Tensor = torch.zeros(
            (batch_size, self.LIG_OUT_DIM),
            dtype=ligand_cat.dtype,
            device=ligand_cat.device,
        )
        # TODO: check if attn enable
        ligand_index = ligand.batch.unsqueeze(-1).expand(-1, self.LIG_OUT_DIM)
        ligand_batch.scatter_add_(
            0, ligand_index, ligand_compose
        )  # (batch_size, LIG_OUT_DIM)

        # protein process
        pro_v = self.protein_compressor(protein.v)  # (N_protein, protein_compress_dim)
        protein_cat = torch.cat([protein.pos, pro_v], dim=-1)  # (N_protein, 4)
        protein_compose: torch.Tensor = self.protein_mlp(
            protein_cat
        )  # (N_protein, PRO_OUT_DIM)
        protein_compose /= protein_compose.shape[0]
        protein_batch = torch.zeros(
            (batch_size, self.PRO_OUT_DIM),
            dtype=protein_cat.dtype,
            device=protein_cat.device,
        )
        protein_index = protein.batch.unsqueeze(-1).expand(-1, self.PRO_OUT_DIM)
        protein_batch.scatter_add_(
            0, protein_index, protein_compose
        )  # (batch_size, PRO_OUT_DIM)

        # out
        out = self.out_net(
            torch.cat([protein_batch, ligand_batch], dim=-1)
        )  # (batch_size, time_classes)
        return out


# Model
class TimeClassifier(nn.Module):

    def __init__(
        self, config: dict, protein_atom_feature_dim=27, ligand_atom_feature_dim=13
    ):
        super().__init__()
        self.config = config

        # sample definition
        self.fast_forward = config["fast_forward"]
        self.time_classes = config["num_diffusion_timesteps"]

        # model definition
        self.hidden_dim = config["hidden_dim"]
        self.num_classes = ligand_atom_feature_dim
        if self.config["node_indicator"]:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)
        self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        # center pos
        self.center_pos_mode = config["center_pos_mode"]  # ['none', 'protein']

        self.refine_net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=config["num_blocks"],
            num_layers=config["num_layers"],
            hidden_dim=config["hidden_dim"],
            n_heads=config["n_heads"],
            edge_feat_dim=config["edge_feat_dim"],
            num_r_gaussian=config["num_r_gaussian"],
            cutoff_mode=config["cutoff_mode"],
            ew_net_type=config["ew_net_type"],
            x2h_out_fc=config["x2h_out_fc"],
        )
        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )
        self.net_cls = ContextEncoder(
            protein_atom_feature_dim,
            1,
        )  # regression: (B, 1)

    # \tau = \mathcal{C}_\theta(x_t,\mathcal{P})
    def forward(
        self,
        protein: MoleBundle,  # pos: (N_protein, 3) N_protein=cat[proteins]
        # v: (N_protein, protein_atom_feature_dim=27) (float32)
        # batch: (N_protein)
        ligand: MoleBundle,  # pos: (N_ligand, 3)
        # v: (N_ligand) index of one-hot vector (int64)
        # batch: (N_ligand)
    ) -> torch.Tensor: # dtype=float32
        """pos:[cat(elements),3]
        v:[cat(elements),27]
        batch:[len(elements)] for element notation"""

        # or it can be batch_protein[-1] for simplicity
        batch_size = protein.batch.max().item() + 1
        # tau: (batch_size, time_classes)
        init_ligand_v = F.one_hot(ligand.v, self.num_classes).float()

        # from utils.misc import bp;bp()
        h_protein = self.protein_atom_emb(protein.v)
        h_ligand = self.ligand_atom_emb(init_ligand_v)

        if self.config["node_indicator"]:
            h_protein = torch.cat(
                [h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1
            )
            assert len(h_ligand.shape) == 2, f"{h_ligand.shape}"
            h_ligand = torch.cat(
                [h_ligand, torch.ones(len(h_ligand), 1).to(h_protein)], -1
            )

        h_all, pos_all, batch_all, mask_ligand = compose_context(
            h_protein=h_protein,
            pos_protein=protein.pos,
            h_ligand=h_ligand,
            pos_ligand=ligand.pos,
            batch_protein=protein.batch,
            batch_ligand=ligand.batch,
        )

        outputs = self.refine_net(
            h_all, pos_all, mask_ligand, batch_all, return_all=False, fix_x=False
        )
        final_pos, final_h = outputs["x"], outputs["h"]
        final_ligand_pos, final_ligand_h = (
            final_pos[mask_ligand],
            final_h[mask_ligand],
        )  # (N_ligand, 3), (N_ligand, hidden_dim=128)
        final_ligand_v = self.v_inference(final_ligand_h)
        # ligand_v: (N_ligand, ligand_atom_feature_dim=13)
        tau = self.net_cls(
            protein, MoleBundle(final_ligand_pos, final_ligand_v, ligand.batch)
        )  # (batch_size, 1)
        # protein:
        # pos: (N_protein, 3)
        # v: (N_protein, protein_atom_feature_dim=27)
        # batch: (N_protein,)
        # ligand:
        # pos: (N_ligand, 3)
        # v: (N_ligand, ligand_atom_feature_dim=13)
        # batch: (N_ligand,)
        return tau

    def parse_time(
        self, protein: MoleBundle, ligand: MoleBundle, timestep: torch.Tensor
    ) -> torch.Tensor:
        tau = self.forward(protein, ligand)
        if self.fast_forward:
            return tau
        else:
            mtau = torch.where(tau > timestep, tau, timestep)
            return mtau


if __name__ == "__main__":
    config = misc.load_config("./configs/class/temp.yml")
    tc = TimeClassifier(config.model, 27, 13)
    print(misc.count_parameters(tc))  # 440936
    # protein_limit = 600
    # ligand_limit = 40
