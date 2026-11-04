import torch
import torch.nn as nn


def mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int = 2) -> nn.Sequential:
    """Linear -> LayerNorm -> ReLU blocks. LayerNorm works per-sample, so it is safe for
    (B, N, D) inputs and does not couple the worlds in a batch (unlike BatchNorm)."""
    layers, d = [], in_dim
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU()]
        d = hidden
    layers += [nn.Linear(d, out_dim), nn.LayerNorm(out_dim), nn.ReLU()]
    return nn.Sequential(*layers)


class DeepSetsBrain(nn.Module):
    """
    Conditioned Deep Sets policy.

        z_i = phi([obstacle_i, agent_emb, goal_emb])      per obstacle, shared weights
        s   = rho([masked_max(z), masked_mean(z)])        permutation-invariant pooling
        out = head([agent_emb, goal_emb, s])

    Conditioning phi on the agent and goal embeddings lets each obstacle be scored in
    relation to where the agent is and where it wants to go, instead of being squeezed
    into a summary vector before the agent/goal are seen.

    Inputs:
        agent_state:     (B, 3)   [x, y, uom]
        goal_state:      (B, 2)
        obstacles_state: (B, N, obstacle_dim)   zero-padded rows are masked out
        obstacle_mask:   (B, N) bool, optional. True = real obstacle.
                         If omitted, rows that are entirely zero are treated as padding.
    Outputs:
        direction_logits (B, n_actions), uom_logits (B, n_uom)
    """

    def __init__(self, latent_size: int = 64, n_layers: int = 2,
                 obstacle_dim: int = 6, n_actions: int = 4, n_uom: int = 3):
        super().__init__()
        H = latent_size

        self.agent_encoder = mlp(3, H, H, n_layers=2)
        self.goal_encoder = mlp(2, H, H, n_layers=2)

        # phi: per-obstacle encoder, conditioned on agent + goal
        self.phi = mlp(obstacle_dim + 2 * H, H, H, n_layers=n_layers)

        # rho: after pooling (max ++ mean)
        self.rho = mlp(2 * H, H, H, n_layers=2)

        # head
        self.head = mlp(3 * H, H, H // 2, n_layers=2)
        self.direction_output = nn.Linear(H // 2, n_actions)
        self.unit_of_motion_output = nn.Linear(H // 2, n_uom)

    def forward(self, agent_state, goal_state, obstacles_state, obstacle_mask=None):
        B, N, _ = obstacles_state.shape

        a = self.agent_encoder(agent_state)                    # (B, H)
        g = self.goal_encoder(goal_state)                      # (B, H)

        if obstacle_mask is None:
            obstacle_mask = obstacles_state.abs().sum(dim=-1) > 0   # (B, N)
        m = obstacle_mask.unsqueeze(-1)                        # (B, N, 1)

        # per-obstacle features, conditioned on agent + goal
        ctx = torch.cat((a, g), dim=-1).unsqueeze(1).expand(-1, N, -1)   # (B, N, 2H)
        z = self.phi(torch.cat((obstacles_state, ctx), dim=-1))          # (B, N, H)

        # masked pooling
        z_max = z.masked_fill(~m, float("-inf")).max(dim=1).values       # (B, H)
        has_any = obstacle_mask.any(dim=1, keepdim=True)                 # (B, 1)
        z_max = torch.where(has_any, z_max, torch.zeros_like(z_max))     # no obstacles -> 0
        count = m.sum(dim=1).clamp_min(1)                                # (B, 1)
        z_mean = (z * m).sum(dim=1) / count                              # (B, H)

        s = self.rho(torch.cat((z_max, z_mean), dim=-1))                 # (B, H)

        h = self.head(torch.cat((a, g, s), dim=-1))                      # (B, H/2)
        return self.direction_output(h), self.unit_of_motion_output(h)


if __name__ == "__main__":
    # quick sanity checks
    torch.manual_seed(0)
    model = DeepSetsBrain(latent_size=64, n_layers=2)
    print("params:", sum(p.numel() for p in model.parameters()))

    B, N = 8, 30
    agent, goal = torch.randn(B, 3), torch.randn(B, 2)
    obs = torch.randn(B, N, 6)
    obs[:, 20:] = 0                                            # padding

    d, u = model(agent, goal, obs)
    print("shapes:", d.shape, u.shape)

    # permutation invariance (shuffle the real obstacles)
    perm = torch.randperm(20)
    obs_p = obs.clone()
    obs_p[:, :20] = obs[:, perm]
    d2, u2 = model(agent, goal, obs_p)
    print("permutation invariant:", torch.allclose(d, d2, atol=1e-5))

    # padding amount shouldn't matter
    d3, _ = model(agent, goal, obs[:, :20])
    print("padding invariant:", torch.allclose(d, d3, atol=1e-5))

    # batch independence (the BatchNorm problem)
    d4, _ = model(agent[:1], goal[:1], obs[:1])
    print("batch independent:", torch.allclose(d[:1], d4, atol=1e-5))