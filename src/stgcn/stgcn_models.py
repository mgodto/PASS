import torch
import torch.nn as nn
from src.stgcn.stgcn_layer import STGCNBlock

# MediaPipe 33個關節點的標準連接關係
EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], 
    [9, 10], [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], 
    [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [11, 23], [12, 24], 
    [23, 24], [23, 25], [25, 27], [27, 29], [27, 31], [24, 26], [26, 28], 
    [28, 30], [28, 32]
]

PART_NODE_GROUPS = {
    "Full Body": list(range(33)),
    "Head": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Trunk": [11, 12, 23, 24],
    "Left Arm": [11, 13, 15, 17, 19, 21],
    "Right Arm": [12, 14, 16, 18, 20, 22],
    "Left Leg": [23, 25, 27, 29, 31],
    "Right Leg": [24, 26, 28, 30, 32],
}

class STGCN_Baseline(nn.Module):
    # (保持不變)
    def __init__(self, num_nodes=33, num_classes=4):
        super().__init__()
        edge_index_tensor = torch.tensor(EDGES, dtype=torch.long).t().contiguous()
        self.register_buffer('edge_index', edge_index_tensor)
        self.st_blocks = nn.ModuleList([
            STGCNBlock(in_channels=3, out_channels=64, kernel_size=9),
            STGCNBlock(in_channels=64, out_channels=128, kernel_size=9),
            STGCNBlock(in_channels=128, out_channels=256, kernel_size=9)
        ])
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        for block in self.st_blocks:
            x = block(x, self.edge_index)
        x = x.mean(dim=[2, 3])
        return self.fc(x)

class STGCN_LateFusion(nn.Module):
    # (保持不變)
    def __init__(self, num_nodes=33, num_subspace_features=0, num_classes=4):
        super().__init__()
        edge_index_tensor = torch.tensor(EDGES, dtype=torch.long).t().contiguous()
        self.register_buffer('edge_index', edge_index_tensor)
        self.st_feature_extractor = nn.ModuleList([
            STGCNBlock(in_channels=3, out_channels=64, kernel_size=9),
            STGCNBlock(in_channels=64, out_channels=128, kernel_size=9),
            STGCNBlock(in_channels=128, out_channels=256, kernel_size=9)
        ])
        st_gcn_output_dim = 256
        fused_dim = st_gcn_output_dim + num_subspace_features
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, skeleton_data, subspace_features):
        x = skeleton_data
        for block in self.st_feature_extractor:
            x = block(x, self.edge_index)
        skeleton_features = x.mean(dim=[2, 3])
        fused_vector = torch.cat([skeleton_features, subspace_features], dim=1)
        return self.fusion_head(fused_vector)

# --- ★★★ 核心修改的 Class ★★★ ---
class STGCN_PartitionFusion(nn.Module):
    """
    針對新版 Partition 特徵的融合模型
    包含 Batch Normalization 層以解決特徵數量級不匹配問題
    """
    def __init__(self, num_nodes=33, num_classes=4, subspace_dim=42): # 這裡預設為 42 (14*3)
        super().__init__()
        
        edge_index_tensor = torch.tensor(EDGES, dtype=torch.long).t().contiguous()
        self.register_buffer('edge_index', edge_index_tensor)
        
        # ST-GCN 主幹
        self.st_feature_extractor = nn.ModuleList([
            STGCNBlock(in_channels=3, out_channels=64, kernel_size=9),
            STGCNBlock(in_channels=64, out_channels=128, kernel_size=9),
            STGCNBlock(in_channels=128, out_channels=256, kernel_size=9)
        ])
        
        st_gcn_output_dim = 256
        
        # ★★★ 新增: Batch Normalization 層 ★★★
        # 用於將輸入的物理特徵 (Max/Mean/Std) 歸一化，使其與 ST-GCN 特徵分佈更接近
        self.subspace_bn = nn.BatchNorm1d(subspace_dim)
        
        # 融合層：256 + 42 = 298
        fused_dim = st_gcn_output_dim + subspace_dim
        
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, skeleton_data, subspace_features):
        # 1. 骨架特徵提取
        x = skeleton_data
        for block in self.st_feature_extractor:
            x = block(x, self.edge_index)

        # Global Average Pooling -> (N, 256)
        skeleton_features = x.mean(dim=[2, 3])
        
        # 2. ★★★ 特徵歸一化 ★★★
        # subspace_features shape: (N, 42)
        subspace_features = self.subspace_bn(subspace_features)
        
        # 3. 融合與分類
        fused_vector = torch.cat([skeleton_features, subspace_features], dim=1)
        
        return self.fusion_head(fused_vector)


class STGCN_PartitionFusionAttention(nn.Module):
    """
    Attention-based late fusion using statistical pooling features (max/mean/std).
    """
    def __init__(self, num_nodes=33, num_classes=4, subspace_dim=42):
        super().__init__()

        edge_index_tensor = torch.tensor(EDGES, dtype=torch.long).t().contiguous()
        self.register_buffer('edge_index', edge_index_tensor)

        self.st_feature_extractor = nn.ModuleList([
            STGCNBlock(in_channels=3, out_channels=64, kernel_size=9),
            STGCNBlock(in_channels=64, out_channels=128, kernel_size=9),
            STGCNBlock(in_channels=128, out_channels=256, kernel_size=9)
        ])

        st_gcn_output_dim = 256

        self.subspace_proj = nn.Sequential(
            nn.Linear(subspace_dim, st_gcn_output_dim, bias=False),
            nn.BatchNorm1d(st_gcn_output_dim),
            nn.ReLU(),
        )

        self.attn_gate = nn.Sequential(
            nn.Linear(st_gcn_output_dim * 2, 1),
            nn.Sigmoid(),
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(st_gcn_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, skeleton_data, subspace_features):
        x = skeleton_data
        for block in self.st_feature_extractor:
            x = block(x, self.edge_index)

        skeleton_features = x.mean(dim=[2, 3])
        stats_proj = self.subspace_proj(subspace_features)

        gate_input = torch.cat([skeleton_features, stats_proj], dim=1)
        alpha = self.attn_gate(gate_input)

        fused_vector = alpha * skeleton_features + (1 - alpha) * stats_proj
        return self.fusion_head(fused_vector)


class STGCN_PartitionFusionConv(nn.Module):
    """
    Partition fusion with a shared 1D Conv encoder per body part.
    Expects subspace_features shape: (N, T, subspace_dim).
    """
    def __init__(
        self,
        num_nodes=33,
        num_classes=4,
        subspace_dim=14,
        part_feat_dim=2,
        part_embed_dim=8,
        conv_channels=8,
        kernel_size=5,
        dropout=0.5,
    ):
        super().__init__()

        if subspace_dim % part_feat_dim != 0:
            raise ValueError("subspace_dim must be divisible by part_feat_dim")

        edge_index_tensor = torch.tensor(EDGES, dtype=torch.long).t().contiguous()
        self.register_buffer('edge_index', edge_index_tensor)

        # ST-GCN backbone
        self.st_feature_extractor = nn.ModuleList([
            STGCNBlock(in_channels=3, out_channels=64, kernel_size=9),
            STGCNBlock(in_channels=64, out_channels=128, kernel_size=9),
            STGCNBlock(in_channels=128, out_channels=256, kernel_size=9)
        ])


        ## ""new add"" batch norm
        self.input_bn = nn.BatchNorm1d(subspace_dim)
        ## ""new add"" batch norm

        self.part_feat_dim = part_feat_dim
        self.num_parts = subspace_dim // part_feat_dim
        self.part_embed_dim = part_embed_dim

        # Shared temporal encoder for each part
        self.part_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=part_feat_dim,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.part_fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(conv_channels, part_embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        st_gcn_output_dim = 256
        fused_dim = st_gcn_output_dim + self.num_parts * part_embed_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, skeleton_data, subspace_features):
        # 1. Skeleton features
        x = skeleton_data
        for block in self.st_feature_extractor:
            x = block(x, self.edge_index)
        skeleton_features = x.mean(dim=[2, 3])

        # 2. Part features
        if subspace_features.dim() != 3:
            raise ValueError("subspace_features must have shape (N, T, D)")
        n, t, d = subspace_features.shape
        expected_dim = self.num_parts * self.part_feat_dim
        if d != expected_dim:
            raise ValueError(f"Expected subspace dim {expected_dim}, got {d}")
        
        # 2. Partition Encoder
        # ★★★ 這裡加入保護機制 ★★★
        # 輸入形狀: (N, T, D) -> (N, D, T)
        subspace_features = subspace_features.permute(0, 2, 1)
        
        # 強制標準化：把幾百幾千的數值壓到 0~1 左右
        subspace_features = self.input_bn(subspace_features)
        
        # 轉回來: (N, D, T) -> (N, T, D)
        subspace_features = subspace_features.permute(0, 2, 1)


        subspace_features = subspace_features.contiguous()
        parts = subspace_features.view(n, t, self.num_parts, self.part_feat_dim)
        parts = parts.permute(0, 2, 3, 1).contiguous()
        parts = parts.view(n * self.num_parts, self.part_feat_dim, t)
        part_emb = self.part_encoder(parts)
        part_emb = self.part_fc(part_emb)
        part_emb = part_emb.view(n, self.num_parts * self.part_embed_dim)

        # 3. Fusion
        fused_vector = torch.cat([skeleton_features, part_emb], dim=1)
        return self.fusion_head(fused_vector)


class STGCN_PartitionFusionHierarchical(nn.Module):
    """
    Hierarchical partition fusion:
    1) ST-GCN output keeps part identity via group pooling over nodes.
    2) Subspace signals stay per-part temporal sequence.
    3) Each part is fused independently, then concatenated for classification.
    Expects subspace_features shape: (N, T, subspace_dim).
    """

    def __init__(
        self,
        num_nodes=33,
        num_classes=4,
        subspace_dim=14,
        part_feat_dim=2,
        selected_part_names=None,
        st_part_dim=64,
        part_dyn_dim=64,
        part_fused_dim=64,
        conv_channels=16,
        kernel_size=5,
        dropout=0.5,
        use_part_gate=True,
    ):
        super().__init__()

        if subspace_dim % part_feat_dim != 0:
            raise ValueError("subspace_dim must be divisible by part_feat_dim")

        self.part_feat_dim = part_feat_dim
        self.num_parts = subspace_dim // part_feat_dim
        self.use_part_gate = bool(use_part_gate)

        if not selected_part_names:
            raise ValueError("selected_part_names is required for hierarchical partition fusion")
        if len(selected_part_names) != self.num_parts:
            raise ValueError(
                f"selected_part_names length ({len(selected_part_names)}) "
                f"does not match num_parts ({self.num_parts})"
            )

        self.selected_part_names = list(selected_part_names)
        self._part_index_buffer_names = []
        for idx, part_name in enumerate(self.selected_part_names):
            if part_name not in PART_NODE_GROUPS:
                raise ValueError(f"Unknown partition part name: {part_name}")
            buffer_name = f"part_indices_{idx}"
            self.register_buffer(
                buffer_name,
                torch.tensor(PART_NODE_GROUPS[part_name], dtype=torch.long),
            )
            self._part_index_buffer_names.append(buffer_name)

        edge_index_tensor = torch.tensor(EDGES, dtype=torch.long).t().contiguous()
        self.register_buffer("edge_index", edge_index_tensor)

        self.st_feature_extractor = nn.ModuleList([
            STGCNBlock(in_channels=3, out_channels=64, kernel_size=9),
            STGCNBlock(in_channels=64, out_channels=128, kernel_size=9),
            STGCNBlock(in_channels=128, out_channels=256, kernel_size=9),
        ])

        self.input_bn = nn.BatchNorm1d(subspace_dim)

        self.part_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=part_feat_dim,
                out_channels=conv_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.part_fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(conv_channels, part_dyn_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.st_part_proj = nn.Sequential(
            nn.Linear(256, st_part_dim),
            nn.ReLU(),
        )
        self.sub_part_proj = nn.Sequential(
            nn.Linear(part_dyn_dim, st_part_dim),
            nn.ReLU(),
        )
        # Optional per-part scalar gate: alpha_i in [0, 1]
        if self.use_part_gate:
            self.part_gate = nn.Sequential(
                nn.Linear(st_part_dim * 2, st_part_dim),
                nn.ReLU(),
                nn.Linear(st_part_dim, 1),
                nn.Sigmoid(),
            )
        else:
            self.part_gate = None
        self.part_fusion = nn.Sequential(
            nn.Linear(st_part_dim, part_fused_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        fused_dim = self.num_parts * part_fused_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def _group_pool(self, x):
        # x: (N, C, T, V)
        part_features = []
        for buffer_name in self._part_index_buffer_names:
            node_indices = getattr(self, buffer_name)
            part_x = x.index_select(dim=3, index=node_indices)
            pooled = part_x.mean(dim=[2, 3])  # (N, C)
            part_features.append(pooled)
        return torch.stack(part_features, dim=1)  # (N, P, C)

    def forward(self, skeleton_data, subspace_features):
        # 1) ST-GCN + group pooling by part
        x = skeleton_data
        for block in self.st_feature_extractor:
            x = block(x, self.edge_index)
        st_part = self._group_pool(x)  # (N, P, 256)

        # 2) Temporal encoding per part from raw subspace sequence
        if subspace_features.dim() != 3:
            raise ValueError("subspace_features must have shape (N, T, D)")
        n, t, d = subspace_features.shape
        expected_dim = self.num_parts * self.part_feat_dim
        if d != expected_dim:
            raise ValueError(f"Expected subspace dim {expected_dim}, got {d}")

        subspace_features = subspace_features.permute(0, 2, 1)  # (N, D, T)
        subspace_features = self.input_bn(subspace_features)
        part_seq = subspace_features.contiguous().view(n, self.num_parts, self.part_feat_dim, t)
        part_seq = part_seq.view(n * self.num_parts, self.part_feat_dim, t)
        part_dyn = self.part_encoder(part_seq)
        part_dyn = self.part_fc(part_dyn).view(n, self.num_parts, -1)  # (N, P, part_dyn_dim)

        # 3) Per-part fusion:
        # Gate mode: F_i = alpha_i * F_sub,i + (1 - alpha_i) * F_gcn,i
        # No-gate mode: F_i = 0.5 * F_sub,i + 0.5 * F_gcn,i
        st_part = self.st_part_proj(st_part)       # (N, P, st_part_dim)
        sub_part = self.sub_part_proj(part_dyn)    # (N, P, st_part_dim)
        if self.part_gate is not None:
            gate_input = torch.cat([sub_part, st_part], dim=2)
            alpha = self.part_gate(gate_input)     # (N, P, 1)
            fused_part = alpha * sub_part + (1.0 - alpha) * st_part
        else:
            fused_part = 0.5 * (sub_part + st_part)
        fused_part = self.part_fusion(fused_part)  # (N, P, part_fused_dim)
        fused_vector = fused_part.reshape(n, -1)
        return self.fusion_head(fused_vector)
