from torch.utils.data.dataloader import DataLoader
from ovo.data.NYU.nyu_dataset import NYUDataset
from ovo.data.NYU.collate import collate_fn_train
from ovo.data.NYU.collate import collate_fn_test
import pytorch_lightning as pl
from ovo.data.utils.torch_util import worker_init_fn


class NYUDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        preprocess_root,
        n_relations=4,
        batch_size=4,
        frustum_size=4,
        num_workers=6,
        all_prep_dir=None,
    ):
        super().__init__()
        self.n_relations = n_relations
        self.preprocess_root = preprocess_root
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frustum_size = frustum_size
        self.all_prep_dir = all_prep_dir

    def setup(self, stage=None):
        self.train_ds = NYUDataset(
            split="train",
            preprocess_root=self.preprocess_root,
            n_relations=self.n_relations,
            root=self.root,
            fliplr=0.5,
            frustum_size=self.frustum_size,
            color_jitter=(0.4, 0.4, 0.4),
            all_prep_dir=self.all_prep_dir
        )
        self.test_ds = NYUDataset(
            split="test",
            preprocess_root=self.preprocess_root,
            n_relations=self.n_relations,
            root=self.root,
            frustum_size=self.frustum_size,
            fliplr=0.0,
            color_jitter=None,
            all_prep_dir=self.all_prep_dir
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn_train,
        )

    def val_dataloader(self):
        return DataLoader(  # same as train_dataloader
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn_train,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn_test,
        )
