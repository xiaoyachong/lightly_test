import os
os.environ["LIGHTLY_TRAIN_CACHE_DIR"] = "/pscratch/sd/x/xchong/lightly/.cache"
os.environ["LIGHTLY_TRAIN_MODEL_CACHE_DIR"] = "/pscratch/sd/x/xchong/lightly/.cache"

import lightly_train

if __name__ == "__main__":
    lightly_train.train_semantic_segmentation(
        out="out_slurm_vitl16/my_experiment_petiole",
        model="dinov3/vitl16-eomt",
        steps=5000,  # Total training steps
        devices=4,
        data={
            "train": {
                "images": "4folders_png/train/images",   # Path to training images
                "masks": "4folders_png/train/masks",     # Path to training masks
            },
            "val": {
                "images": "4folders_png/val/images",     # Path to validation images
                "masks": "4folders_png/val/masks",       # Path to validation masks
            },
            "classes": {                                # Classes in the dataset 
                0: "background",
                1: "cortex",
                2: "Phloem Fibers",
                3: "Phloem",
                4: "Xylem vessels",
                5: "Air-based Pith cells",
                6: "Water-based Pith cells",
                255: "unknown"
            },
            # Optional, classes that are in the dataset but should be ignored during
            # training.
            "ignore_classes": [255],  # Ignore background (0) and any 255 values 
        },
        logger_args={
            "log_every_num_steps": 100,           # Log training metrics every 100 steps
            "val_every_num_steps": 100,           # Validate every 200 steps
            "val_log_every_num_steps": 100,         # Log all validation steps
        },
        save_checkpoint_args={
            "save_every_num_steps": 100,          # Save checkpoint every 200 steps
            "save_last": True,                     # Save last checkpoint
            "save_best": True,                     # Save best checkpoint
        },
    )