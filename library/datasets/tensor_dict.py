import typing as t
import torch
from torch.utils.data import Dataset
from collections.abc import Mapping

class TensorDictDataset(Dataset):
    def __init__(self, data: t.Dict[str, torch.Tensor]) -> None:
        """
        A dataset that stores data in a dictionary of tensors.
        Each key-value pair corresponds to a feature or label name and a tensor.

        Args:
            data: A dictionary of tensors. All tensors must have the same length
                  in the first dimension (i.e., same batch size or number of samples).
        Raises:
            TypeError: If input data is not a dict or any value is not a Tensor.
            ValueError: If tensor lengths in the first dimension differ.
        """
        if not isinstance(data, Mapping):
            raise TypeError("Data must be a dictionary (dict) instance.")

        if not all(isinstance(tensor, torch.Tensor) for tensor in data.values()):
            raise TypeError("All values in the data must be torch.Tensor instances.")

        lengths: t.Set[int] = {tensor.shape[0] for tensor in data.values()}
        if len(lengths) != 1:
            raise ValueError("All tensors must have the same length in the first dimension.")

        self.data = data
        self._length = next(iter(data.values())).shape[0]

    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        
        Returns
        -------
        int
            The total number of samples.
        """
        
        return self._length

    def __getitem__(self, idx: int) -> t.Dict[str, torch.Tensor]:
        """Retrieve the sample corresponding to the index.
        
        Parameters
        ----------
        idx : int
            Index to access the sample. Must be in range [0, len(self)).
            
        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary containing the indexed tensor elements.
            
        Raises
        ------
        IndexError
            If index is out of bounds.
        """
        
        if not t.TYPE_CHECKING and not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} is out of bounds [0, {len(self) - 1}]")

        return {key: tensor[idx] for key, tensor in self.data.items()}

    def to(self, 
        device:str | torch.device
    ) -> 'TensorDictDataset':
        """Move all tensors in the dataset to a specific device.
        
        Parameters
        ----------
        device : str or torch.device
            The target device (e.g., 'cpu', 'cuda', or torch.device instance).
            
        Returns
        -------
        TensorDictDataset
            A new instance of TensorDictDataset with tensors moved to the device.
        """
        
        new_data: t.Dict[str, torch.Tensor] = {
            key: tensor.to(device) for key, tensor in self.data.items()
        }
        return TensorDictDataset(new_data)

    def get_dataloader(
        self,
        #? --- DataLoader Configuration ---
        batch_size: int = 32,
        shuffle: bool = False,
        #? --- Performance Settings ---
        num_workers: int = 0,
        pin_memory: bool = False,
        #? --- Batch Handling ---
        drop_last: bool = False,
    ) -> torch.utils.data.DataLoader:
        """Create a DataLoader instance for this dataset.
        
        Parameters
        ----------
        batch_size : int, default=32
            Number of samples per batch.
        shuffle : bool, default=False
            Whether to shuffle the data at every epoch.
        num_workers : int, default=0
            Number of subprocesses for data loading.
        pin_memory : bool, default=False
            Whether to use pinned memory for faster GPU transfers.
        drop_last : bool, default=False
            Whether to drop the last incomplete batch.
            
        Returns
        -------
        torch.utils.data.DataLoader
            A DataLoader instance configured with this dataset.
        """
        return torch.utils.data.DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    def __repr__(self) -> str:
        """Return a developer-focused string representation.
        
        Returns
        -------
        str
            A string representation that can recreate the object.
        """
        if not self.data:
            return f"{self.__class__.__name__}(empty)"

        sample_tensor = next(iter(self.data.values()))
        shape = sample_tensor.shape

        return (
            f"{self.__class__.__name__}(\n"
            f"    keys={list(self.data.keys())},\n"
            f"    shape={shape},\n"
            f"    length={len(self)}\n"
            f")"
        )

    @staticmethod
    def _validate_tensor(tensor: torch.Tensor) -> None:
        """Validate that the tensor has at least one dimension.
        
        Parameters
        ----------
        tensor : torch.Tensor
            A PyTorch tensor to validate.
            
        Raises
        ------
        ValueError
            If tensor has no dimensions.
        """
        if tensor.dim() < 1:
            raise ValueError("Tensor must have at least one dimension.")