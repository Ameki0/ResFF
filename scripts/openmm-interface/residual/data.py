# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from os.path import join
from tqdm import tqdm
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from lightning import LightningDataModule
from lightning_utilities.core.rank_zero import rank_zero_warn
from torchmdnet import datasets
from torchmdnet.utils import make_splits, MissingEnergyException
from torchmdnet.models.utils import scatter
from mendeleev import element
import warnings
import math

# get the pauling electronegativity from mendeleev
elems = {'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 35, 'I': 53}
en_vals = [getattr(element(atomic_num), 'en_pauling', None) for atomic_num in elems.values()]
std_en_vals = [x / en_vals[0] for x in en_vals]
std_vals_dict = {atomic_num: std_en_vals[i] for i, atomic_num in enumerate(elems.values())}


class DataModule(LightningDataModule):
    """A LightningDataModule for loading datasets from the torchmdnet.datasets module.

    Args:
        hparams (dict): A dictionary containing the hyperparameters of the
            dataset. See the documentation of the torchmdnet.datasets module
            for details.
        dataset (torch_geometric.data.Dataset): A dataset to use instead of
            loading a new one from the torchmdnet.datasets module.
    """

    def __init__(self, hparams, dataset=None):
        super(DataModule, self).__init__()
        self.save_hyperparameters(hparams)
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()
        self.dataset = dataset

    def adjust_noise_scale(self):
        """Adjusts noise scale using cosine decay based on the global step from the trainer."""
        total_steps = self.hparams['num_steps']  # Total number of steps for the decay
        cosine_decay = 0.5 * (1 + math.cos(math.pi * self.trainer.global_step / total_steps))
        noise_scale = self.hparams['min_noise_scale'] + (self.hparams['max_noise_scale'] - self.hparams['min_noise_scale']) * cosine_decay
        return noise_scale
    
    def filter_molecules(self, data):
        """Remove molecules containing elements not in the predefined list."""
        atom_ids = data.z.tolist()  # Assuming 'z' is your atomic number attribute
        # Check if all elements in the molecule are in the allowed elements list
        if all(atom_id in elems.values() for atom_id in atom_ids):
            return True
        return False
    
    def setup(self, stage):
        if self.dataset is None:
            def create_dataset_factory(transform, filter_fn=None):
                if self.hparams["dataset"] == 'Custom':
                    return datasets.Custom(
                        self.hparams["coord_files"],
                        self.hparams["embed_files"],
                        self.hparams["energy_files"],
                        self.hparams["force_files"],
                        self.hparams["dataset_preload_limit"],
                        transform=transform,
                        pre_filter=filter_fn if self.hparams['filter_elem'] else None
                    )
                else:
                    dataset_arg = {}
                    if self.hparams["dataset_arg"] is not None:
                        dataset_arg = self.hparams["dataset_arg"]
                    if self.hparams["dataset"] == "HDF5":
                        dataset_arg["dataset_preload_limit"] = self.hparams["dataset_preload_limit"]
                    return getattr(datasets, self.hparams["dataset"])(
                        self.hparams["dataset_root"],
                        **dataset_arg,
                        transform=transform,
                        pre_filter=filter_fn if self.hparams['filter_elem'] else None
                    )

            transform = None
            if (self.hparams['element_wise_noise']) and (self.hparams['position_noise_scale'] > 0.):
                def transform(data):
                    noise = torch.randn_like(data.pos) * self.hparams['position_noise_scale']
                    default_scale = 1.0
                    ele_scales = torch.tensor([std_vals_dict.get(z.item(), default_scale) for z in data.z], device=data.pos.device)
                    scaled_noise_by_ele = noise * ele_scales.unsqueeze(1)
                    data.pos_target = scaled_noise_by_ele
                    data.pos = data.pos + scaled_noise_by_ele
                    return data
            elif (self.hparams['element_wise_noise']) and (self.hparams['max_noise_scale'] > 0):
                def transform(data):
                    noise_scale = self.adjust_noise_scale()
                    noise = torch.randn_like(data.pos) * noise_scale
                    default_scale = 1.0
                    ele_scales = torch.tensor([std_vals_dict.get(z.item(), default_scale) for z in data.z], device=data.pos.device)
                    scaled_noise_by_ele = noise * ele_scales.unsqueeze(1)
                    data.pos_target = scaled_noise_by_ele
                    data.pos = data.pos + scaled_noise_by_ele
                    return data
            elif (not self.hparams['element_wise_noise']) and (self.hparams['max_noise_scale'] > 0.):
                def transform(data):
                    noise_scale = self.adjust_noise_scale()
                    noise = torch.randn_like(data.pos) * noise_scale
                    data.pos_target = noise
                    data.pos = data.pos + noise
                    return data
            else:
                def transform(data):
                    noise = torch.randn_like(data.pos) * self.hparams['position_noise_scale']
                    data.pos_target = noise
                    data.pos = data.pos + noise
                    return data
            # Create datasets with or without molecule filtering based on 'filter_elem' flag
            if self.hparams['filter_elem']:
                self.dataset_maybe_noisy = create_dataset_factory(transform, self.filter_molecules)
                self.dataset = create_dataset_factory(None, self.filter_molecules)
            else:
                self.dataset_maybe_noisy = create_dataset_factory(transform)
                self.dataset = create_dataset_factory(None)        
    
        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(self.dataset),
            self.hparams["train_size"],
            self.hparams["val_size"],
            self.hparams["test_size"],
            self.hparams["seed"],
            join(self.hparams["log_dir"], "splits.npz"),
            self.hparams["splits"],
        )
        print(
            f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}"
        )

        self.train_dataset = Subset(self.dataset_maybe_noisy, self.idx_train)
        if self.hparams['denoising_only']:
            self.val_dataset = Subset(self.dataset_maybe_noisy, self.idx_val)
            self.test_dataset = Subset(self.dataset_maybe_noisy, self.idx_test)            
        else:
            self.val_dataset = Subset(self.dataset, self.idx_val)
            self.test_dataset = Subset(self.dataset, self.idx_test)

        if self.hparams["standardize"]:
            # Mark as deprecated
            warnings.warn(
                "The standardize option is deprecated and will be removed in the future. ",
                DeprecationWarning,
            )
            self._standardize()

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        loaders = [self._get_dataloader(self.val_dataset, "val")]
        # To allow to report the performance on the testing dataset during training
        # we send the trainer two dataloaders every few steps and modify the
        # validation step to understand the second dataloader as test data.
        if self._is_test_during_training_epoch():
            loaders.append(self._get_dataloader(self.test_dataset, "test"))
        return loaders

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def atomref(self):
        """Returns the atomref of the dataset if it has one, otherwise None."""
        if hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self):
        """Returns the mean of the dataset if it has one, otherwise None."""
        return self._mean

    @property
    def std(self):
        """Returns the standard deviation of the dataset if it has one, otherwise None."""
        return self._std

    def _is_test_during_training_epoch(self):
        return (
            len(self.test_dataset) > 0
            and self.hparams["test_interval"] > 0
            and self.trainer.current_epoch > 0
            and self.trainer.current_epoch % self.hparams["test_interval"] == 0
        )

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        if stage in self._saved_dataloaders and store_dataloader:
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]

        shuffle = stage == "train"
        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.hparams["num_workers"],
            persistent_workers=True,
            pin_memory=True,
            shuffle=shuffle,
        )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl

    def _standardize(self):
        def get_energy(batch, atomref):
            if "y" not in batch or batch.y is None:
                raise MissingEnergyException()

            if atomref is None:
                return batch.y.clone()

            # remove atomref energies from the target energy
            atomref_energy = scatter(atomref[batch.z], batch.batch, dim=0)
            return (batch.y.squeeze() - atomref_energy.squeeze()).clone()

        data = tqdm(
            self._get_dataloader(self.train_dataset, "val", store_dataloader=False),
            desc="computing mean and std",
        )
        try:
            # only remove atomref energies if the atomref prior is used
            atomref = self.atomref if self.hparams["prior_model"] == "Atomref" else None
            # extract energies from the data
            ys = torch.cat([get_energy(batch, atomref) for batch in data])
        except MissingEnergyException:
            rank_zero_warn(
                "Standardize is true but failed to compute dataset mean and "
                "standard deviation. Maybe the dataset only contains forces."
            )
            return

        # compute mean and standard deviation
        self._mean = ys.mean(dim=0)
        self._std = ys.std(dim=0)
