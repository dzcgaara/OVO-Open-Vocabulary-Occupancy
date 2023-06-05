from torch.utils.data.dataloader import DataLoader
from ovo.data.semantic_kitti.kitti_dataset import KittiDataset
import pytorch_lightning as pl
from ovo.data.semantic_kitti.collate import collate_fn_train
from ovo.data.semantic_kitti.collate import collate_fn_test
from ovo.data.utils.torch_util import worker_init_fn


class KittiDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        preprocess_root,
        project_scale=2,
        frustum_size=4,
        batch_size=4,
        num_workers=6,
        all_prep_dir=None,
    ):
        super().__init__()
        self.root = root
        self.preprocess_root = preprocess_root
        self.project_scale = project_scale
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frustum_size = frustum_size
        self.all_prep_dir = all_prep_dir

    def setup(self, stage=None):
        self.train_ds = KittiDataset(
            split="train",
            root=self.root,
            preprocess_root=self.preprocess_root,
            project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0.5,
            color_jitter=(0.4, 0.4, 0.4),
            all_prep_dir=self.all_prep_dir
        )

        self.val_ds = KittiDataset(
            split="val",
            root=self.root,
            preprocess_root=self.preprocess_root,
            project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0,
            color_jitter=None,
            all_prep_dir=self.all_prep_dir
        )

        self.test_ds = KittiDataset(
            split="test",
            root=self.root,
            preprocess_root=self.preprocess_root,
            project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0,
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
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn_train,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_ds,  # test on val (sequence 08)
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn_test,
        )
