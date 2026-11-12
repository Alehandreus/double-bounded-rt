import torch
import torch.nn as nn

import tinycudann as tcnn


class RayModel(nn.Module):
    def __init__(self, model_config, mesh):
        super().__init__()

        self.model_config = model_config

        mesh_min, mesh_max = mesh.get_bounds()
        self.mesh_min = nn.Parameter(torch.tensor(mesh_min, dtype=torch.float32), requires_grad=False)
        self.mesh_max = nn.Parameter(torch.tensor(mesh_max, dtype=torch.float32), requires_grad=False)

        self.network_config = model_config.network_config
        self.point_encoding_config = model_config.point_encoding_config
        self.direction_encoding_config = model_config.direction_encoding_config

        # output dimentions of mlp head: presence of intersection (1), distance (1), normal (3) and color (3)
        self.n_output_dims = 8

        self.direction_encoding = tcnn.Encoding(3, self.direction_encoding_config)
        self.point_encoding = tcnn.Encoding(3, self.point_encoding_config)

        # "3d": encode entry and exit points; "3d+1": also their midpoint
        n_points = {"3d": 2, "3d+1": 3}[model_config.encoding_type]
        self.mlp_input_dims = self.point_encoding.n_output_dims * n_points + self.direction_encoding.n_output_dims

        self.network = tcnn.Network(self.mlp_input_dims, self.n_output_dims, self.network_config)

    def forward(self, points, points_inner, directions):
        directions_enc = self.direction_encoding((directions + 1) / 2).float()

        extent = self.mesh_max - self.mesh_min
        points = (points - self.mesh_min) / extent
        points_inner = (points_inner - self.mesh_min) / extent
        encoded = [self.point_encoding(points).float(), self.point_encoding(points_inner).float()]
        if self.model_config.encoding_type == "3d+1":
            encoded.append(self.point_encoding((points + points_inner) / 2).float())

        y = self.network(torch.cat(encoded + [directions_enc], dim=1)).float()
        has_intersection = y[:, 0]
        distance = y[:, 1]
        normal = y[:, 2:5]
        colors = y[:, 5:8]

        return has_intersection, distance, normal, colors
