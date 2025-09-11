# This file is part of the espaloma package.
""" Legacy models from DGL.

"""

# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl
# =============================================================================
# CONSTANT
# =============================================================================
DEFAULT_MODEL_KWARGS = {
    "SAGEConv": {"aggregator_type": "mean"},
    "GATConv": {"num_heads": 4},
    "TAGConv": {"k": 2},
}


# =============================================================================
# MODULE CLASSES
# =============================================================================
class GN(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        model_name="GraphConv",
        kwargs={},
    ):
        super(GN, self).__init__()
        from dgl.nn import pytorch as dgl_pytorch

        if kwargs == {}:
            if model_name in DEFAULT_MODEL_KWARGS:
                kwargs = DEFAULT_MODEL_KWARGS[model_name]

        self.gn = getattr(dgl_pytorch.conv, model_name)(
            in_features, out_features, **kwargs
        )

        # register these properties here for downstream handling
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, g, x):
        return self.gn(g, x)


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================


def gn(model_name="GraphConv",kwargs={}):
    from dgl.nn import pytorch as dgl_pytorch

    if model_name == "GINConv":
        return lambda in_features, out_features: dgl_pytorch.conv.GINConv(
            apply_func=torch.nn.Linear(in_features, out_features),
            aggregator_type="sum",
        )
#测试
    elif model_name == "GATConv":
        return lambda in_features, out_features: dgl_pytorch.conv.GATConv(
            in_feats=in_features,
            out_feats=out_features,
            num_heads=kwargs.get("num_heads", 1),  # Get num_heads from kwargs or default to 1
            feat_drop=kwargs.get("feat_drop", 0.),
            attn_drop=kwargs.get("attn_drop", 0.),
            negative_slope=kwargs.get("negative_slope", 0.2),
            residual=kwargs.get("residual", False),
            activation=kwargs.get("activation", None),
            allow_zero_in_degree=kwargs.get("allow_zero_in_degree", False),
            bias=kwargs.get("bias", True),
            kwargs=kwargs,
            #share_weights=kwargs.get("share_weights", False),
        )
#
    else:
        return lambda in_features, out_features: GN(
            in_features=in_features,
            out_features=out_features,
            model_name=model_name,
            kwargs=kwargs,
    
        )
