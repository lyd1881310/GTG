import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch.autograd import Function


def get_total_costs(model, network, device, t, grad):
    dur_cost, spd_cost, hidden_cost = get_costs(model, network, device, t, grad=grad)
    # obs cost 在 pref train 阶段不更新
    dur_cost, spd_cost = dur_cost.detach(), spd_cost.detach()
    pref = model.combine_pref(dur_cost, spd_cost, hidden_cost)
    return dur_cost, spd_cost, hidden_cost, pref


def get_costs(model, network, device, t, grad):
    t = torch.tensor(t, dtype=torch.float32, device=device)
    model.to(device)
    network.to(device)
    if not grad:
        with torch.no_grad():
            dur_cost, spd_cost, hidden_cost = model.pred_total_cost(
                torch.cat([t.repeat(network.num_nodes, 1), network.x], dim=-1),
                network.edge_index,
                network.edge_attr
            )
    else:
        with torch.enable_grad():
            dur_cost, spd_cost, hidden_cost = model.pred_total_cost(
                torch.cat([t.repeat(network.num_nodes, 1), network.x], dim=-1),
                network.edge_index,
                network.edge_attr
            )
    return dur_cost, spd_cost, hidden_cost


class GTGModel(nn.Module):
    def __init__(self, cfg):
        super(GTGModel, self).__init__()
        self.device = cfg['device']

        self.aggregator = TopoAggregator(cfg)
        self.disent_encoder = DisentEncoder(cfg)
        self.sem_pred = ObsCostPredictor(cfg, use_grl=False)
        self.dom_pred = ObsCostPredictor(cfg, use_grl=True)
        self.sem_disc = DomClassifier(cfg, use_grl=True)
        self.dom_disc = DomClassifier(cfg, use_grl=False)

        # preference pred
        self.hidden_pred = PredictBlock(cfg)
        self.obs_wt = nn.Parameter(torch.randn(2), requires_grad=True)

    def disentangle(self, nodes, edges, relation):
        reps = self.aggregator(nodes, edges, relation)
        sem_reps, dom_reps = self.disent_encoder(reps)
        sem_dur, sem_spd = self.sem_pred(sem_reps)
        dom_dur, dom_spd = self.dom_pred(dom_reps)
        sem_clf, dom_clf = self.sem_disc(sem_reps), self.dom_disc(dom_reps)
        return (sem_reps, dom_reps, sem_dur, sem_spd, sem_clf,
                dom_dur, dom_spd, dom_clf)

    def pred_obs_cost(self, nodes, edges, relation):
        reps = self.aggregator(nodes, edges, relation)
        sem_reps, _ = self.disent_encoder(reps)
        return self.sem_pred(sem_reps)

    def pred_total_cost(self, nodes, edges, relation):
        reps = self.aggregator(nodes, edges, relation)
        sem_reps, _ = self.disent_encoder(reps)
        dur_cost, spd_cost = self.sem_pred(sem_reps)
        hidden_cost = self.hidden_pred(sem_reps)
        return dur_cost, spd_cost, hidden_cost

    def combine_pref(self, dur_cost, spd_cost, hidden_cost):
        dur_cost = dur_cost / dur_cost.sum()
        spd_cost = spd_cost / spd_cost.sum()
        hidden_cost = hidden_cost / hidden_cost.sum()
        weight = torch.softmax(self.obs_wt, dim=-1)
        pref = weight[0] * dur_cost + weight[1] * spd_cost + hidden_cost
        return pref


class TopoAggregator(nn.Module):
    def __init__(self, cfg):
        super(TopoAggregator, self).__init__()
        self.device = cfg['device']

        # 离散属性 embedding
        self.time_emb = nn.Embedding(cfg['num_time'], cfg['dim_time'])
        self.type_emb = nn.Embedding(cfg['num_type'], cfg['dim_type'])
        self.biway_emb = nn.Embedding(2, cfg['dim_biway'])
        self.islink_emb = nn.Embedding(2, cfg['dim_islink'])

        in_channel = cfg['dim_con_feature'] + cfg['dim_time'] + cfg['dim_type'] + cfg['dim_biway'] + cfg['dim_islink']
        self.proj = nn.Linear(in_channel, cfg['dim_gat_hidden'])

        self.gnn_layers = nn.ModuleList()
        for _ in range(cfg['num_layers']):
            self.gnn_layers.append(GATv2Conv(in_channels=cfg['dim_gat_hidden'], out_channels=cfg['dim_gat_hidden'],
                                             heads=cfg['dim_gat_heads'], concat=False, dropout=cfg['dropout'],
                                             edge_dim=cfg['dim_edge']))

    def forward(self, nodes, edges, relation):
        nodes, edges = nodes.to(self.device), edges.to(self.device)
        time_index = nodes[:, 0].long()
        types, biway, islink = nodes[:, 1].long(), nodes[:, 2].long(), nodes[:, 3].long()
        con_feat = nodes[:, 4:]

        # note: 离散特征做 embedding
        time_rep = self.time_emb(time_index)
        type_rep = self.type_emb(types)
        biway_rep = self.biway_emb(biway)
        islink_rep = self.islink_emb(islink)

        nodes = torch.concat([time_rep, type_rep, biway_rep, islink_rep, con_feat], dim=-1)
        nodes = self.proj(nodes)
        for layer in self.gnn_layers:
            # 残差连接
            nodes = layer(nodes, edges, relation) + nodes
            nodes = torch.relu(nodes)
        return nodes


class PredictBlock(nn.Module):
    def __init__(self, cfg):
        super(PredictBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg['dim_gat_hidden'], cfg['dim_predict_hidden']),
            nn.BatchNorm1d(cfg['dim_predict_hidden']),
            nn.ReLU(),
            nn.Linear(cfg['dim_predict_hidden'], 1),
            # nn.Sigmoid()  # 把 cost 约束在 0~1 之间
            nn.Softplus()
        )

    def forward(self, reps):
        return self.mlp(reps).squeeze()


class DisentBlock(nn.Module):
    def __init__(self, cfg):
        super(DisentBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg['dim_gat_hidden'], cfg['dim_gat_hidden']),
            nn.BatchNorm1d(cfg['dim_gat_hidden']),
            nn.ReLU(),
            nn.Linear(cfg['dim_gat_hidden'], cfg['dim_gat_hidden']),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)


class DisentEncoder(nn.Module):
    def __init__(self, cfg):
        """
        把 GAT 提取的表征解耦成语义信息和域信息
        """
        super(DisentEncoder, self).__init__()
        self.sem_encoder = DisentBlock(cfg)
        self.dom_encoder = DisentBlock(cfg)

    def forward(self, x):
        sem_rep = self.sem_encoder(x)
        dom_rep = self.dom_encoder(x)
        return sem_rep, dom_rep


class GradReverse(Function):
    @ staticmethod
    def forward(ctx, x, **kwargs: None):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -1.0 * grad_output, None


class ObsCostPredictor(nn.Module):
    def __init__(self, cfg, use_grl: bool):
        super(ObsCostPredictor, self).__init__()
        self.time_pred = PredictBlock(cfg)
        self.spd_pred = PredictBlock(cfg)
        self.use_grl = use_grl

    def forward(self, nodes):
        if self.use_grl:
            nodes = GradReverse.apply(nodes)
        dur_cost = self.time_pred(nodes)
        spd_cost = self.spd_pred(nodes)
        return dur_cost, spd_cost


class DomClassifier(nn.Module):
    def __init__(self, cfg, use_grl: bool):
        super(DomClassifier, self).__init__()
        self.dom_clf = nn.Sequential(
            nn.Linear(cfg['dim_gat_hidden'], cfg['dim_predict_hidden']),
            nn.BatchNorm1d(cfg['dim_predict_hidden']),
            nn.ReLU(),
            nn.Linear(cfg['dim_predict_hidden'], 1),
            nn.Sigmoid()
        )
        self.use_grl = use_grl

    def forward(self, x):
        if self.use_grl:
            x = GradReverse.apply(x)
        return self.dom_clf(x).squeeze()
