import torch as th
import torch.nn as nn

import dgl.function as fn
import dgl.mp as mp
from dgl.nn.functional import edge_softmax

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)

    def apply_edges(self, edges):
        h_e = edges.data["h"]
        h_u = edges.src["h"]
        h_v = edges.dst["h"]
        score = self.W(th.cat([h_e, h_u, h_v], -1))
        return {"score": score}

    def forward(self, g, e_feat, u_feat, v_feat):
        with g.local_scope():
            g.edges["forward"].data["h"] = e_feat
            g.nodes["u"].data["h"] = u_feat
            g.nodes["v"].data["h"] = v_feat
            g.apply_edges(self.apply_edges, etype="forward")
            return g.edges["forward"].data["score"]

class GASConv(nn.Module):
    """One layer of GAS."""

    def __init__(
        self,
        e_in_dim,
        u_in_dim,
        v_in_dim,
        e_out_dim,
        u_out_dim,
        v_out_dim,
        activation=None,
        dropout=0,
    ):
        super(GASConv, self).__init__()

        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.e_linear = nn.Linear(e_in_dim, e_out_dim)
        self.u_linear = nn.Linear(u_in_dim, e_out_dim)
        self.v_linear = nn.Linear(v_in_dim, e_out_dim)

        self.W_ATTN_u = nn.Linear(u_in_dim, v_in_dim + e_in_dim)
        self.W_ATTN_v = nn.Linear(v_in_dim, u_in_dim + e_in_dim)

        # the proportion of h_u and h_Nu are specified as 1/2 in formula 8
        nu_dim = int(u_out_dim / 2)
        nv_dim = int(v_out_dim / 2)

        self.W_u = nn.Linear(v_in_dim + e_in_dim, nu_dim)
        self.W_v = nn.Linear(u_in_dim + e_in_dim, nv_dim)

        self.Vu = nn.Linear(u_in_dim, u_out_dim - nu_dim)
        self.Vv = nn.Linear(v_in_dim, v_out_dim - nv_dim)

    def forward(self, g, e_feat, u_feat, v_feat):
        # formula 3 and 4 (optimized implementation to save memory)
        he_u = self.u_linear(u_feat)
        he_v = self.v_linear(v_feat)
        fw_he_e = self.e_linear(e_feat)
        he = fw_he_e + mp.copy_u(g, he_u, etype='forward') + mp.copy_v(he_v)
        if self.activation is not None:
            he = self.activation(he)

        # formula 6
        h_ve = th.cat([mp.copy_u(g, he_v, etpye='backward'), e_feat], -1)
        h_ue = th.cat([mp.copy_u(g, he_u, etpye='forward'), e_feat], -1)

        # formula 7, self-attention
        h_att_u = self.W_ATTN_u(u_feat)
        h_att_v = self.W_ATTN_v(v_feat)

        # Step 1: dot product
        bw_edotv = h_ve * mp.copy_u(g, h_att_u, etype='backward')
        fw_edotv = h_ue * mp.copy_u(g, h_att_v, etype='forward')

        # Step 2. softmax
        bw_sfm = mp.edge_softmax(g, bw_edotv, etype='backward')
        fw_sfm = mp.edge_softmax(g, fw_edotv, etype='forward')

        # Step 3. Broadcast softmax value to each edge, and then attention is done
        bw_attn = h_ve * bw_sfm
        fw_attn = h_ue * fw_sfm

        # Step 4. Aggregate attention to dst,user nodes, so formula 7 is done
        agg_u = mp.sum(g, bw_attn, etype='backward')
        agg_v = mp.sum(g, fw_attn, etype='forward')

        # formula 5
        h_nu = self.W_u(agg_u)
        h_nv = self.W_v(agg_v)

        if self.activation is not None:
            h_nu = self.activation(h_nu)
            h_nv = self.activation(h_nv)

        # Dropout
        he = self.dropout(he)
        h_nu = self.dropout(h_nu)
        h_nv = self.dropout(h_nv)

        # formula 8
        hu = th.cat([self.Vu(u_feat), h_nu], -1)
        hv = th.cat([self.Vv(v_feat), h_nv], -1)

        return he, hu, hv

class GAS(nn.Module):
    def __init__(
        self,
        e_in_dim,
        u_in_dim,
        v_in_dim,
        e_hid_dim,
        u_hid_dim,
        v_hid_dim,
        out_dim,
        num_layers=2,
        dropout=0.0,
        activation=None,
    ):
        super(GAS, self).__init__()
        self.e_in_dim = e_in_dim
        self.u_in_dim = u_in_dim
        self.v_in_dim = v_in_dim
        self.e_hid_dim = e_hid_dim
        self.u_hid_dim = u_hid_dim
        self.v_hid_dim = v_hid_dim
        self.out_dim = out_dim
        self.num_layer = num_layers
        self.dropout = dropout
        self.activation = activation
        self.predictor = MLP(e_hid_dim + u_hid_dim + v_hid_dim, out_dim)
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(
            GASConv(
                self.e_in_dim,
                self.u_in_dim,
                self.v_in_dim,
                self.e_hid_dim,
                self.u_hid_dim,
                self.v_hid_dim,
                activation=self.activation,
                dropout=self.dropout,
            )
        )

        # Hidden layers with n - 1 CompGraphConv layers
        for i in range(self.num_layer - 1):
            self.layers.append(
                GASConv(
                    self.e_hid_dim,
                    self.u_hid_dim,
                    self.v_hid_dim,
                    self.e_hid_dim,
                    self.u_hid_dim,
                    self.v_hid_dim,
                    activation=self.activation,
                    dropout=self.dropout,
                )
            )

    def forward(self, graph, e_feat, u_feat, v_feat):
        # For full graph training, directly use the graph
        # Forward of n layers of GAS
        for layer in self.layers:
            e_feat, u_feat, v_feat = layer(graph, e_feat, u_feat, v_feat)

        # return the result of final prediction layer
        return self.predictor(graph, e_feat, u_feat, v_feat)
