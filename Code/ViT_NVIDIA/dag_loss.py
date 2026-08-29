"""
Asymmetric extinction-group DAG hop-loss (the cost-function piece).

The EG adjacency (artifacts/topology_graphs/extinction_group_adjacency.json) is a
group-subgroup DAG: adjacency[child] = [parents] (higher symmetry). Its node indices ARE
the canonical EG indices (verified identity map), so no remapping.

We build:
  D[t,k] = undirected shortest-path hop distance (for the eval metric)
  C[t,k] = ASYMMETRIC training cost: vertical hops (k is an ancestor/descendant of t, i.e. same
           group-subgroup lineage) are cheap (gamma_v); lateral hops (different branch) are
           expensive (gamma_l > gamma_v). Rationale: the weak-reflection wall biases the net
           toward over-/under-symmetrization (vertical, cascade-recoverable); a wrong lineage
           (lateral) is a genuine misread.

Loss (blended so Top-1 isn't sacrificed):
  L = CE(logits, y) + lam * E_{k~softmax(logits)}[ C[y,k] ]
The second term pushes probability mass toward the DAG-neighborhood of the truth, shaping the
Top-5 into a coherent neighborhood without collapsing Top-1.
"""
import json, numpy as np
from collections import deque


def build_cost_matrix(adj_path, n=99, gamma_v=1.0, gamma_l=2.5):
    d = json.load(open(adj_path))
    adjacency = d["adjacency"]                      # child -> [parents]
    und = {i: set() for i in range(n)}
    parents = {i: set() for i in range(n)}
    children = {i: set() for i in range(n)}
    for c, ps in adjacency.items():
        c = int(c)
        for p in ps:
            p = int(p)
            und[c].add(p); und[p].add(c)
            parents[c].add(p); children[p].add(c)

    INF = float(n)                                   # large but finite (disconnected/missing node)
    D = np.full((n, n), INF, dtype=np.float32)
    for s in range(n):
        D[s, s] = 0.0
        q = deque([s]); seen = {s}
        while q:
            u = q.popleft()
            for v in und[u]:
                if v not in seen:
                    seen.add(v); D[s, v] = D[s, u] + 1.0; q.append(v)

    def reach(start, graph):
        seen = set(); q = deque([start])
        while q:
            u = q.popleft()
            for v in graph[u]:
                if v not in seen:
                    seen.add(v); q.append(v)
        return seen

    vertical = np.eye(n, dtype=bool)
    for t in range(n):
        for k in reach(t, parents) | reach(t, children):
            vertical[t, k] = True

    C = D.copy()
    C[~vertical] *= gamma_l
    C[vertical] *= gamma_v
    np.fill_diagonal(C, 0.0)
    # normalize so E_p[C] ~ O(1) at uniform p -> lam reads as "fraction of CE"
    finite = (D < INF) & ~np.eye(n, dtype=bool)
    scale = C[finite].mean() if finite.any() else 1.0
    C = C / max(scale, 1e-6)
    return C, D


class DagHopLoss:
    """L = CE + lam * sum_k softmax(logits)_k * C[y,k]; plus mean-hop eval via D."""
    def __init__(self, adj_path, device, lam=0.2, n=99):
        import torch
        self._torch = torch
        C, D = build_cost_matrix(adj_path, n=n)
        self.C = torch.tensor(C, device=device)
        self.D = torch.tensor(D, device=device)
        self.lam = lam
        self.ce = torch.nn.CrossEntropyLoss()

    def __call__(self, logits, y):
        ce = self.ce(logits, y)
        if self.lam <= 0:
            return ce
        p = self._torch.softmax(logits, dim=1)
        hop_pen = (p * self.C[y]).sum(dim=1).mean()
        return ce + self.lam * hop_pen

    def mean_hop(self, logits, y):
        pred = logits.argmax(dim=1)
        return self.D[y, pred].mean().item()
