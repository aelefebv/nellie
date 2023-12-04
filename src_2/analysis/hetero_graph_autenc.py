pip install torch torchvision torchaudio
pip install torch-geometric

import torch
from torch_geometric.nn import GCNConv, HeteroConv
from torch_geometric.data import HeteroData

class HeteroGraphAutoEncoder(torch.nn.Module):
    def __init__(self, skeleton_channels, branch_channels, organelle_channels, hidden_channels):
        super(HeteroGraphAutoEncoder, self).__init__()

        # Intra-node message passing layers
        self.skeleton_conv = GCNConv(skeleton_channels, hidden_channels)
        self.branch_conv = GCNConv(branch_channels, hidden_channels)
        self.organelle_conv = GCNConv(organelle_channels, hidden_channels)

        # Inter-node message passing layer
        self.hetero_conv = HeteroConv({
            ('skeleton', 'to', 'branch'): GCNConv(hidden_channels, hidden_channels),
            ('branch', 'to', 'organelle'): GCNConv(hidden_channels, hidden_channels),
            ('organelle', 'to', 'skeleton'): GCNConv(hidden_channels, hidden_channels)
        }, aggr='mean')

        # Decoder layers
        self.decoder = torch.nn.ModuleDict({
            'skeleton': GCNConv(hidden_channels, skeleton_channels),
            'branch': GCNConv(hidden_channels, branch_channels),
            'organelle': GCNConv(hidden_channels, organelle_channels)
        })

    def encode(self, x_dict, edge_index_dict):
        z_dict = {}
        # Intra-node message passing
        z_dict['skeleton'] = self.skeleton_conv(x_dict['skeleton'], edge_index_dict[('skeleton', 'to', 'skeleton')])
        z_dict['branch'] = self.branch_conv(x_dict['branch'], edge_index_dict[('branch', 'to', 'branch')])
        z_dict['organelle'] = self.organelle_conv(x_dict['organelle'], edge_index_dict[('organelle', 'to', 'organelle')])

        # Inter-node message passing
        z_dict = self.hetero_conv(z_dict, edge_index_dict)
        return z_dict

    def decode(self, z_dict):
        x_recon_dict = {}
        x_recon_dict['skeleton'] = self.decoder['skeleton'](z_dict['skeleton'])
        x_recon_dict['branch'] = self.decoder['branch'](z_dict['branch'])
        x_recon_dict['organelle'] = self.decoder['organelle'](z_dict['organelle'])
        return x_recon_dict

    def forward(self, x_dict, edge_index_dict):
        z_dict = self.encode(x_dict, edge_index_dict)
        x_recon_dict = self.decode(z_dict)
        return x_recon_dict

model = HeteroGraphAutoEncoder(
    skeleton_channels=3,  # Replace with the actual number of skeleton features
    branch_channels=5,    # Replace with the actual number of branch features
    organelle_channels=4, # Replace with the actual number of organelle features
    hidden_channels=16    # Hidden channel size, you can adjust this
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

data = HeteroData({
    'skeleton': torch.randn(10, 3),  # Replace with the actual skeleton feature tensor
    'branch': torch.randn(10, 5),    # Replace with the actual branch feature tensor
    'organelle': torch.randn(10, 4), # Replace with the actual organelle feature tensor
    ('skeleton', 'to', 'skeleton'): torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]),
    ('branch', 'to', 'branch'): torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]),
    ('organelle', 'to', 'organelle'): torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]),
    ('skeleton', 'to', 'branch'): torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]),
    ('branch', 'to', 'organelle'): torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]),
    ('organelle', 'to', 'skeleton'): torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]])
})

# Training loop
for epoch in range(100):  # Number of epochs, adjust as needed
    model.train()
    optimizer.zero_grad()
    x_recon_dict = model(data.x_dict, data.edge_index_dict)
    loss = sum([criterion(x_recon_dict[node_type], data[node_type].x) for node_type in data.x_dict])
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch}, Loss: {loss.item()}')

