""" Chain mutiple layers of GN together.
"""
import torch
import torch_geometric
import time
class _Sequential(torch.nn.Module):
    """Sequentially staggered neural networks."""

    def __init__(
        self,
        layer,
        config,
        in_features,
        model_kwargs={},
    ):
        super(_Sequential, self).__init__()

        self.exes = []

        # init dim
        dim = in_features

        # parse the config
        for idx, exe in enumerate(config):

            try:
                exe = float(exe)

                if exe >= 1:
                    exe = int(exe)
            except BaseException:
                pass

            # int -> feedfoward
            if isinstance(exe, int):
                setattr(self, "d" + str(idx), layer(dim, exe, **model_kwargs))
                dim = exe
                self.exes.append("d" + str(idx))

            # str -> torch or activation
            if isinstance(exe, str):
                if exe == "torch":
                    setattr(self, "torch" + str(idx), layer(dim, 128, **model_kwargs)) 
                    self.exes.append("torch" + str(idx))

                if exe == "bn":
                    setattr(self, "a" + str(idx), torch.nn.BatchNorm1d(dim))
                    self.exes.append("a" + str(idx))

                if exe == "relu":
                    activation = getattr(torch.nn.functional, exe)
                    setattr(self, "a" + str(idx), activation)
                    self.exes.append("a" + str(idx))

            # float -> dropout
            if isinstance(exe, float):
                dropout = torch.nn.Dropout(exe)
                setattr(self, "o" + str(idx), dropout)

                self.exes.append("o" + str(idx))

    def forward(self, _g, x, coord_feat, batch, g):
       
        for exe in self.exes:

            
            if exe.startswith("d"):
                if _g is not None:
                    x = getattr(self, exe)(_g, x)
                else:
                    x = getattr(self, exe)(x)
               
          
            if exe.startswith("torch"):
                x = getattr(self, exe)(g, x, coord_feat, batch)
       

            if exe.startswith("a"):
                x = getattr(self, exe)(x)

            if exe.startswith("o"):
                x = getattr(self, exe)(x)

        return x


class Sequential(torch.nn.Module):
    """Sequential neural network with input layers.

    Parameters
    ----------
    layer : torch.nn.Module
        DGL graph convolution layers.

    config : List
        A sequence of numbers (for units) and strings (for activation functions)
        denoting the configuration of the sequential model.

    feature_units : int(default=117)
        The number of input channels.

    Methods
    -------
    forward(g, x)
        Forward pass.
    """

    def __init__(
        self,
        layer,
        config,
        feature_units=114,
        input_units=128,
        model_kwargs={},
    ):
        super(Sequential, self).__init__()

        # initial featurization
        self.f_in = torch.nn.Sequential(
            torch.nn.Linear(feature_units, input_units), torch.nn.Tanh()
        )

        self._sequential = _Sequential(
            layer, config, in_features=input_units, model_kwargs=model_kwargs
        )
    
    def _forward(self, g, x):
        """Forward pass with graph and features."""
        for exe in self.exes:
            if exe.startswith("d"):
                x = getattr(self, exe)(g, x)
            else:
                x = getattr(self, exe)(x)

        return x

    def forward(self, g, x=None,coord_feat=None):
        """Forward pass.

        Parameters
        ----------
        g : `dgl.DGLHeteroGraph`,
            input graph

        Returns
        -------
        g : `dgl.DGLHeteroGraph`
            output graph
        """
        import dgl
        import logging
        # Calculate position vectors and edge weights
        # get homogeneous subgraph
        g_ = dgl.to_homo(g.edge_type_subgraph(["n1_neighbors_n1"]))
       
        
        if x is None:
            # get node attributes
            x = g.nodes["n1"].data["h0"]
            x = self.f_in(x)
            g.nodes["n1"].data["h"] = x
            
        coord_feat = g.nodes['n1'].data['xyz']
       
        a = g.nodes["n1"].data["idxs"]

        batch = torch.zeros(a.shape[0], dtype=torch.long, device=a.device)
        n = -1

        for i in range(a.shape[0]):
            if a[i] != 0:
                batch[i] = n
            else:
                n += 1
                batch[i] = n

 
        x = self._sequential(g_, x, coord_feat, batch, g)
        
        # put attribute back in the graph
        g.nodes["n1"].data["h"] = x

        return g



