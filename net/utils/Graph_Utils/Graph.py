import numpy as np
import torch
import net.utils.Graph_Utils.halpe_graph as h_g


ntu_links = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)]

ntu_links = [(i-1, j-1) for (i, j) in ntu_links]

azure_links = [(0, 1), (1, 2), (2, 3), (3, 26), (4, 3), (5, 4), (6, 5),
             (7, 6), (8, 7), (9, 8), (10, 7), (11, 3), (12, 11), (13, 12), (14, 13),
             (15, 14), (16, 15), (17, 14),  (18, 0), (19, 18), (20, 19), (21, 20),
             (22, 0), (23, 22), (24, 23), (25, 24), (26, 3), (27, 26), (28, 27),
            (29, 28), (30, 26), (31, 30)]

coco_links = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
                (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
                (1, 0), (3, 1), (2, 0), (4, 2)]

openpose_links = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9),
                (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0),
                (14, 0), (17, 15), (16, 14)]


def k_adjacency(A, k, with_self=False, self_factor=1):
    # A is a 2D square array
    if isinstance(A, torch.Tensor):
        A = A.data.cpu().numpy()
    assert isinstance(A, np.ndarray)
    Iden = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return Iden
    Ak = np.minimum(np.linalg.matrix_power(A + Iden, k), 1) - np.minimum(np.linalg.matrix_power(A + Iden, k - 1), 1)
    if with_self:
        Ak += (self_factor * Iden)
    return Ak


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A, dim=0):
    # A is a 2D square array
    # computes A D^{-1}
    Dl = np.sum(A, dim)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = 1/Dl[i] if Dl[i] != 0 else 0
    AD = np.dot(A, Dn)
    return AD


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.eye(num_node)
    for i, j in edge:
        A[i, j] = 1
        A[j, i] = 1
    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [
        np.linalg.matrix_power(A, d) for d in range(max_hop + 1)
    ]
    arrive_mat = (np.stack(transfer_mat) > 0) # tensor of shape (max_hop+1, num_nodes, num_nodes)
    # if any value in arrive_mat is true, for each hop distance starting from
    # the largest going to the smallest, then set that value to be the hop value
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


class Graph:
    def __init__(self,
                 layout='nturgb+d',
                 mode='spatial',
                 max_hop=1):
        self.max_hop = max_hop
        self.layout = layout
        self.mode = mode
        self.get_layout(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.inward, max_hop)
        assert hasattr(self, mode), f'Do Not Exist This Mode: {mode}'
        self.A = getattr(self, mode)()

    def __str__(self):
        return self.A

    def get_layout(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self.inward = openpose_links
            self.center = 1
        elif layout == 'nturgb+d':
            self.num_node = 25
            self.inward = ntu_links
            #self.inward = [(i - 1, j - 1) for (i, j) in neighbor_base]
            self.center = 21 - 1 # 21 in 1-indexing
        elif layout == 'coco':
            self.num_node = 17
            self.inward = coco_links
            self.center = 0
        elif layout == 'azure':
            self.num_node = 32
            self.inward = azure_links
        elif layout == 'halpe_all':
            self.num_node = 136
            self.inward = h_g.all_halpe_links
        elif layout == "halpe_medium":
            self.inward = h_g.medium_halpe_graph_links
            self.num_node = 68
        elif layout == "halpe_simple":
            self.inward = h_g.simple_halpe_graph_links
            self.num_node = 26
        elif layout == "halpe_simple_wh":
            self.inward = h_g.simple_halpe_graph_links_wh
            self.num_node = 32
        else:
            raise ValueError(f'This layout: {layout} does not exist')
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward

    def stgcn_spatial(self):
        adj = np.zeros((self.num_node, self.num_node))
        adj[self.hop_dis <= self.max_hop] = 1
        normalize_adj = normalize_digraph(adj)
        hop_dis = self.hop_dis
        center = self.center
        A = []
        # for each node and hop distance, if a hop distance (i ,j)
        # equals the hop, set either close to be the normalized adjacency value of (j, i)
        # if j is closer than equal to the center (in terms of hop distance)
        # otherwise set the further matrix to have the normalized adjacency value of (j, i)
        for hop in range(self.max_hop + 1):
            a_close = np.zeros((self.num_node, self.num_node))
            a_further = np.zeros((self.num_node, self.num_node))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if hop_dis[j, i] == hop:
                        if hop_dis[j, center] >= hop_dis[i, center]:
                            a_close[j, i] = normalize_adj[j, i]
                        else:
                            a_further[j, i] = normalize_adj[j, i]
            A.append(a_close)
            if hop > 0:
                A.append(a_further)
        return np.stack(A)

    def spatial(self):
        Iden = edge2mat(self.self_link, self.num_node)
        In = normalize_digraph(edge2mat(self.inward, self.num_node))
        Out = normalize_digraph(edge2mat(self.outward, self.num_node))
        A = np.stack((Iden, In, Out))
        return A

    def binary_adj(self):
        A = edge2mat(self.inward + self.outward, self.num_node)
        return A[None]
