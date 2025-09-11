""" Legacy models from PYG.

"""

# =============================================================================
# IMPORTS
# =============================================================================
import torch
import torch_geometric
import resff.nn.layers.torchmdnet as torchmdnet
# =============================================================================
# MODULE CLASSES
# =============================================================================
class GN(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        model_name="TorchMD_ET",
        kwargs={},
    ):
        super(GN, self).__init__()

        self.gn = getattr(torchmdnet, model_name)(
        hidden_channels=80,
        num_layers=3,
        num_rbf=64,
        rbf_type="expnorm",
        trainable_rbf=True,
        activation="silu",
        attn_activation="silu",
        neighbor_embedding=True,
        num_heads=8,
        distance_influence="both",
        cutoff_lower=0.0,
        cutoff_upper=10.0,
        max_atom_type=14,
        max_num_neighbors=128,
        derivative=False,**kwargs
        )

        # register these properties here for downstream handling
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, g, x, coord_feat, batch):
        return self.gn(g, x, coord_feat, batch)


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================


def gn(model_name="TorchMD_ET",kwargs={}):

    if model_name == "TorchMD_ET":

        return lambda in_features, out_features: GN(
            in_features=in_features,
            out_features=out_features,
            model_name=model_name,
            kwargs=kwargs,
    
        )
    else:
        print(f'{model_name} is not exist!')
