# Code modified from: 
# https://github.com/DalhousieAI/benthicnet_probes/blob/master/utils/benthicnet_dataset.py
# Under GPL-3.0 License

import os

import PIL.Image
import torch.utils.data

class EchinodermsDataset(torch.utils.data.Dataset):
    """BenthicNet dataset."""

    def __init__(
        self,
        annotations=None,
        transform=None,
        local=None,
    ):
        """
        Dataset for BenthicNet data.

        Parameters
        ----------
        tar_dir : str
            Directory with all the images.
        annotations : str
            Dataframe with annotations.
        transform : callable, optional
            Optional transform to be applied on a sample.
        """
        self.dataframe = annotations.copy()
        self.dataframe.loc[:, "tarname"] = self.dataframe.loc[:, "dataset"] + ".tar"
        self.transform = transform
        self.local = local

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        img_name = row["image"] + ".jpg"

        path = row["dataset"] + "/" + row["site"] + "/" + img_name

        if self.local:
            node_file_path = self.local + "/" + path
        else:
            node_file_path = os.path.join(os.environ["SLURM_TMPDIR"], path)

        sample = PIL.Image.open(node_file_path)

        if self.transform:
            sample = self.transform(sample)

        return sample, row["catami_biota"]
