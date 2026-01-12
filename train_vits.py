import os
os.environ["LIGHTLY_TRAIN_CACHE_DIR"] = "/pscratch/sd/x/xchong/lightly/.cache"
os.environ["LIGHTLY_TRAIN_MODEL_CACHE_DIR"] = "/pscratch/sd/x/xchong/lightly/.cache"

import lightly_train

if __name__ == "__main__":
    lightly_train.train_semantic_segmentation(
        out="out_slurm_vits16/my_experiment",
        model="dinov3/vits16-eomt",
        steps=2000,  # Total training steps
        devices=4,
        data={
            "train": {
                "images": "sand_data_seg_png/train/images",   # Path to training images
                "masks": "sand_data_seg_png/train/masks",     # Path to training masks
            },
            "val": {
                "images": "sand_data_seg_png/val/images",     # Path to validation images
                "masks": "sand_data_seg_png/val/masks",       # Path to validation masks
            },
            "classes": {                                # Classes in the dataset                    
                0: "outside field of view",
                1: "air",
                2: "sand",
                3: "capillary area",
                4: "inclusions",
                # ...
            },
            # Optional, classes that are in the dataset but should be ignored during
            # training.
            #"ignore_classes": [0], 
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