import os
import hydra
from omegaconf import DictConfig
import numpy as np
import torch

from ovo.data.semantic_kitti.kitti_dm import KittiDataModule
from ovo.data.semantic_kitti.params import (
    semantic_kitti_class_frequencies,
    kitti_class_names,
)
from ovo.data.NYU.params import (
    class_weights as NYU_class_weights,
    NYU_class_names,
)
from ovo.data.NYU.nyu_dm import NYUDataModule
from ovo.models.ovo import OVO
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

hydra.output_subdir = None


@hydra.main(config_name="../config/ovo.yaml")
def main(config: DictConfig):
    exp_name = config.exp_prefix
    exp_name += "_{}_{}".format(config.dataset, config.run)
    exp_name += "_WD{}_lr{}".format(config.weight_decay, config.lr)

    if config.confidence_weight:
        exp_name += "_confidence"
    else:
        exp_name += "_woconfidence"
    if config.voxel_pixel_align:
        exp_name += "_VPAlign"
        exp_name += "_VPAweight_"+str(config.VPA_weight)
    else:
        exp_name += "_woVPAlign"
    if config.voxel_word_align:
        exp_name += "_VWAlign"
    else:
        exp_name += "_woVWAlign"
    if config.align_2d:
        exp_name += "_2dAlign_"
    else:
        exp_name += "_wo2dAlign_"

    # Setup dataloaders
    if config.dataset == "kitti":
        class_names = kitti_class_names
        max_epochs = 100
        logdir = config.logdir
        full_scene_size = (256, 256, 32)
        project_scale = 2
        feature = 64
        n_classes = 20
        class_weights = torch.from_numpy(
            1 / np.log(semantic_kitti_class_frequencies + 0.001)
        )
        data_module = KittiDataModule(
            root=config.kitti_root,
            preprocess_root=config.kitti_preprocess_root,
            frustum_size=config.frustum_size,
            project_scale=project_scale,
            batch_size=int(config.batch_size / config.n_gpus),
            num_workers=int(config.num_workers_per_gpu),
            all_prep_dir=config.kitti_prepare_total,
        )
        exp_name += config.kitti_prepare_total.split("/")[-1]

    elif config.dataset == "NYU":
        class_names = NYU_class_names
        max_epochs = 100
        logdir = config.logdir
        full_scene_size = (60, 36, 60)
        project_scale = 1
        feature = 200
        n_classes = 12
        class_weights = NYU_class_weights
        data_module = NYUDataModule(
            root=config.NYU_root,
            preprocess_root=config.NYU_preprocess_root,
            n_relations=config.n_relations,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size / config.n_gpus),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
            all_prep_dir=config.NYU_prepare_total,
        )
        exp_name += config.NYU_prepare_total.split("/")[-1]

    project_res = ["1"]
    if config.project_1_2:
        project_res.append("2")
    if config.project_1_4:
        project_res.append("4")
    if config.project_1_8:
        project_res.append("8")

    print(exp_name)

    # Initialize OVO model
    model = OVO(
        dataset=config.dataset,
        frustum_size=config.frustum_size,
        project_scale=project_scale,
        n_relations=config.n_relations,
        fp_loss=config.fp_loss,
        feature=feature,
        full_scene_size=full_scene_size,
        project_res=project_res,
        n_classes=n_classes,
        class_names=class_names,
        context_prior=config.context_prior,
        relation_loss=config.relation_loss,
        CE_ssc_loss=config.CE_ssc_loss,
        sem_scal_loss=config.sem_scal_loss,
        geo_scal_loss=config.geo_scal_loss,
        lr=config.lr,
        weight_decay=config.weight_decay,
        class_weights=class_weights,
        # for ablation study
        confidence_weight=config.confidence_weight,
        voxel_pixel_align=config.voxel_pixel_align,
        voxel_word_align=config.voxel_word_align,
        align_2d=config.align_2d,
        VPA_weight=config.VPA_weight,
        with_ov=config.with_ov
    )

    if config.enable_log:
        logger = TensorBoardLogger(save_dir=logdir, name=exp_name, version="")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_callbacks = [
            ModelCheckpoint(
                save_last=True,
                monitor="val/mIoU",
                save_top_k=1,
                mode="max",
                filename="{epoch:03d}-{val/mIoU:.5f}",
            ),
            lr_monitor,
        ]
    else:
        logger = False
        checkpoint_callbacks = False

    model_path = os.path.join(logdir, exp_name, "checkpoints/last.ckpt")
    if os.path.isfile(model_path):
        # Continue training from last.ckpt
        trainer = Trainer(
            callbacks=checkpoint_callbacks,
            resume_from_checkpoint=model_path,
            sync_batchnorm=True,
            deterministic=False,
            max_epochs=max_epochs,
            gpus=config.n_gpus,
            logger=logger,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            flush_logs_every_n_steps=100,
            accelerator="ddp",
        )
    else:
        # Train from scratch
        trainer = Trainer(
            callbacks=checkpoint_callbacks,
            sync_batchnorm=True,
            deterministic=False,
            max_epochs=max_epochs,
            gpus=config.n_gpus,
            logger=logger,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            flush_logs_every_n_steps=100,
            accelerator="ddp",
        )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    # compare with naive Monoscene training, we should modify as follow
    # 1. add lseg embedding for Unet2d supervision  NOTE that we should lift it to 3d for end2end inference
    # 2. add BEV feature align module
    # 3. add img align module, which should add a voxel-shape gt like (b,w,h,d,512) to align feature with step 2 result. However, a bool mask is also needed given by invalid pair info, a weight matrix is also needed for lseg confidence.
    # 4. add word align module, which will also be prepared before.
    # In sum, we should prepare 1. 2d lseg ret, 2. (b,w,h,d,512) img 23d align gt 3. (b,w,h,d,1) valid positions 4. (b,w,h,d,1) mimic weight
    main()
