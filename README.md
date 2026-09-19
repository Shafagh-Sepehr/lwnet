![wnet](models/fig2.png?raw=true "wnet")

# The Little W-Net that Could
You have reached the official repository for our work on retinal vessel segmentation with minimalistic models.
The above picture represents a WNet architecture, which contains roughly around 70k parameters and closely matches (or outperforms) other more complicated techniques.
For more details about our work, you can check the related paper:

```
The Little W-Net That Could: State-of-the-Art Retinal Vessel Segmentation with Minimalistic Models
Adrian Galdran, André Anjos, Jose Dolz, Hadi Chakor, Hervé Lombaert, Ismail Ben Ayed
https://arxiv.org/abs/2009.01907, Sep. 2020
```

We would appreciate if you could cite our work if it is useful for you :)


> **Note**: If you are just looking for our results, you can directly download them [at this link](https://gitlab.com/agaldran/shared_results/-/raw/master/pre_generated_results.zip?inline=false).


Please find below a table of contents describing what you can find in this repository:

## Table of Contents
1. [Dependencies and getting the data ready](https://github.com/agaldran/lwnet#1-dependencies-and-getting-the-data-ready)
2. [Training a W-Net for vessel segmentation](https://github.com/agaldran/lwnet#2-training-a-w-net-for-vessel-segmentation)
3. [Generating segmentations](https://github.com/agaldran/lwnet#3-generating-segmentations)
4. [Computing Performance](https://github.com/agaldran/lwnet#4-computing-performance)
5. [Cross-Dataset Experiments](https://github.com/agaldran/lwnet#5-cross-dataset-experiments)
6. [Training with pseudo-labels and computing performance](https://github.com/agaldran/lwnet#6-training-with-pseudo-labels-and-computing-performance)
7. [Evaluating your own model](https://github.com/agaldran/lwnet#7-evaluating-your-own-model)
8. [Training a W-Net for Artery/Vein segmentation](https://github.com/agaldran/lwnet#8-training-a-w-net-for-arteryvein-segmentation)
9. [Generating Artery/Vein segmentations](https://github.com/agaldran/lwnet#9-generating-arteryvein-segmentations)
10. [Generating vessel and A/V segmentations on your own data](https://github.com/agaldran/lwnet#10-generating-vessel-and-av-segmentations-on-your-own-data)

## 1. Dependencies and getting the data ready
First things first, clone this repo somewhere in your computer:
```
git clone https://github.com/agaldran/lwnet.git .
```

For full reproducibility, you should use the configuration specified in the `requirements.txt` file.
If you are using conda, you can install dependencies in one line, just run on a terminal:
```
conda create --name lwnet --file environment.txt
conda activate lwnet
```

We have made an effort to automate the data download and preparation so that everything is as reproducible as possible.
Out of the ten datasets we use in the paper, seven of them are public, and you can get them just running:
 ```
python get_public_data.py
```
This will populate the `data` directory with the seven sub-folders.
If everything goes right, each sub-folder in `data` is named as the corresponding dataset, and contains at least:
* Three folders called `images`, `mask`, `manual`
* A csv file called `test_all.csv`

If the dataset is used in our work for training a vessel segmentation model (DRIVE, CHASE-DB, and HRF), you will also find:
* Three csv files called `train.csv`, `val.csv`, `test.csv`

If the dataset also has Artery/Vein annotations, you will also see:
* A folder called `manual_av`
* A csv file called `test_all_av.csv`

If the dataset is used in our work for training an A/V models (DRIVE and HRF), you will also find:
* Three csv files called `train_av.csv`, `val_av.csv`, `test_av.csv`

> **Note**: The DRIVE dataset will also contain a folder called `ZoneB_manual`, which is used to evaluate A/V performance around the optic disc.
The HRF dataset will also contain folders called `images_resized`, `manual_resized`, `mask_resized`.
These are used only for training.

> **Note**: The LES-AV dataset is still public but it now needs to be downloaded manually, please see the comments in `get_public_data.py` Line 400 forward for details.

## 2. Training a W-Net for vessel segmentation
Train a model on a given dataset. You also need to supply the path to save the model.
Note that the training defaults to using the CPU, which is feasible due to the small size of our models.
Checkpoint validation runs at cycle boundaries by default. Use `--epoch_checkpointing_from N` to
start epoch-level checks at cycle N (use `0` to keep cycle-level checks throughout). `--metric`
accepts an ordered list such as `auc,dice,loss`; multi-metric
policies use absolute near-tie tolerances (`auc=0.0005`, `dice=0.0001`, `loss=0.000001`) unless
overridden with `--metric_tolerances`. A single metric remains strict unless an explicit tolerance
is supplied. Comparisons are pairwise against the last selected checkpoint, so approximate equality
is not transitive; `max_validation_auc_seen` in checkpoint metadata records the diagnostic maximum.
To reproduce our results in table 2 of our paper, you need to run:
```
python train_cyclical.py --csv_train data/DRIVE/train.csv --cycle_lens 20/50
                         --model_name wnet --save_path wnet_drive --device cuda:0
python train_cyclical.py --csv_train data/CHASEDB/train.csv --cycle_lens 40/50
                         --model_name wnet --save_path wnet_chasedb --device cuda:0
python train_cyclical.py --csv_train data/HRF/train.csv --cycle_lens 30/50
                         --model_name wnet --save_path wnet_hrf_1024
                         --im_size 1024 --batch_size 2 --grad_acc_steps 1 --device cuda:0
```
This will store the model weights in `experiments/wnet_drive`, `experiments/wnet_chasedb`, `experiments/wnet_hrf` respectively.

The parameter `cycle_lens` specifies the length of the training, and it is adjusted depending on the amount of images in the training set.
For instance, in the DRIVE case, `--cycle_lens 20/50` implies that we train for 20 cycles, each cycle running for 50 epochs.
As CHASE-DB has less training images than DRIVE (8 vs 16), we double the number of cycles in that case.

Note that we use a `batch_size` of 4 by default, and that we train on HRF with an image size of `1024x1024`.
In order to train on a single GPU, we use gradient accumulation in that case.

## 2.1 Gated cross-stage decoder bridges (A1)

The W-Net can optionally give U-Net 2 direct access to U-Net 1's multi-scale decoder
representations through zero-initialized gated residual bridges. At each selected decoder
stage `i`, the U2 feature is updated as `u2_feature_i = u2_feature_i + alpha_i * u1_feature_i`,
where `alpha_i` is a learned gate initialized to exactly zero. Before any training update the
bridged model therefore reproduces the original W-Net exactly.

Three training options control the bridge:

| Option | Meaning |
|---|---|
| `--cross_stage_bridge none|scalar|channel` | `none` (default) is the legacy architecture; `scalar` adds one gate per selected scale; `channel` adds one gate per channel per scale. |
| `--cross_stage_bridge_scales all|quarter,half,full` | Which decoder scales to bridge. `all` canonicalizes to `half,full` for `wnet` and `quarter,half,full` for `big_wnet`. |
| `--cross_stage_bridge_init 0.0` | Initial raw gate value; the A1 default is exactly `0.0`. |

Primary experiment (pair it against an identical control with `--cross_stage_bridge none`):

```
python train_cyclical.py --csv_train data/DRIVE/train.csv --cycle_lens 20/50 --model_name wnet --cross_stage_bridge scalar --cross_stage_bridge_scales all --cross_stage_bridge_init 0.0 --checkpoint_interval epoch --metric auc,dice,loss --batch_size 4 --im_size 512 --device cuda:0 --save_path wnet_drive_a1_scalar_all_seed0 --seed 0
```

Inference requires no new bridge flags: when `--config_file` (or a model directory containing
`config.cfg`) is supplied, the bridge architecture is recovered automatically and the checkpoint
is loaded strictly.

## 3. Generating segmentations
Once the model is trained, you can produce the corresponding segmentations calling `generate_results.py` and specifying which dataset should be used:
```
python generate_results.py --config_file experiments/wnet_drive/config.cfg
                           --dataset DRIVE --device cuda:0
python generate_results.py --config_file experiments/wnet_chasedb/config.cfg
                           --dataset CHASEDB --device cuda:0
python generate_results.py --config_file experiments/wnet_hrf_1024/config.cfg
                           --dataset HRF --im_size 1024 --device cuda:0
```
The above stores the predictions for those datasets in `results/DRIVE/experiments/wnet_drive`, `results/CHASEDB/experiments/wnet_chasedb`, and `results/HRF/experiments/wnet_hrf_1024` respectively.

## 4. Computing Performance
We call `analyze_results.py` to compute performance.
It is important to specify what was the training and what is the test set here.
For that, you pass the path to the train/test predictions, and the name of the train/test datasets:
```
python analyze_results.py --path_train_preds results/DRIVE/experiments/wnet_drive
                          --path_test_preds results/DRIVE/experiments/wnet_drive
                          --train_dataset DRIVE --test_dataset DRIVE

python analyze_results.py --path_train_preds results/CHASEDB/experiments/wnet_chasedb
                          --path_test_preds results/CHASEDB/experiments/wnet_chasedb
                          --train_dataset CHASEDB --test_dataset CHASEDB

python analyze_results.py --path_train_preds results/HRF/experiments/wnet_hrf_1024
                          --path_test_preds results/HRF/experiments/wnet_hrf_1024
                          --train_dataset HRF --test_dataset HRF
```
The code uses the csv files in each dataset folder to check which images should be used for running an AUC analysis in the training set and finding an optimal binarizing threshold to be used in the test set images.
## 5. Cross-Dataset Experiments
When a model has been trained on dataset A (say, DRIVE) and we want to test it on dataset B (say, CHASE-DB), we first generate segmentations on both datasets:
```
python generate_results.py --config_file experiments/wnet_drive/config.cfg
                           --dataset DRIVE  --device cuda:0
python generate_results.py --config_file experiments/wnet_drive/config.cfg
                           --dataset CHASEDB  --device cuda:0
```
and then we compute performance:
```
python analyze_results.py --path_train_preds results/DRIVE/experiments/wnet_drive
                          --path_test_preds results/CHASEDB/experiments/wnet_drive
                          --train_dataset DRIVE --test_dataset CHASEDB
```

## 6. Training with pseudo-labels and computing performance
1) Train a model on a source dataset (DRIVE); this will store the model in `experiments/wnet_drive`
```
python train_cyclical.py --csv_train data/DRIVE/train.csv --cycle_lens 20/50
                         --model_name wnet --save_path wnet_drive
                         --device cuda:0
```
2) Generate predictions on target dataset (CHASEDB) with this model; this will store predictions at `results/CHASEDB/experiments/wnet_drive`
```
python generate_results.py --config_file experiments/wnet_drive/config.cfg
                           --dataset CHASEDB --device cuda:0
```

3) Train a model on DRIVE manual segmentations plus CHASEDB pseudo-segmentations for one cycle of 10 epochs with a lower learning rate, starting from the weights of the model trained on DRIVE.
Note that in this case we use the AUC on the training set as checkpointing criterion.
This training is slower because of the AUC computation on a large set of images at the end of each cycle.
In this case, we save the new model in a  folder called `wnet_drive_chasedb_pl`:
```
python train_cyclical.py --save_path wnet_drive_chasedb_pl
                         --checkpoint_folder experiments/wnet_drive
                         --csv_test data/CHASEDB/test_all.csv
                         --path_test_preds results/CHASEDB/experiments/wnet_drive
                         --max_lr 0.0001 --cycle_lens 10/1 --metric tr_auc
                         --device cuda:0
```

3) Generate predictions with this new model on source dataset DRIVE:
```
python generate_results.py --config_file experiments/wnet_drive_chasedb_pl/config.cfg
                           --dataset DRIVE --device cuda:0
```

4) Generate predictions on target dataset CHASEDB:
```
python generate_results.py --config_file experiments/wnet_drive_chasedb_pl/config.cfg
                           --dataset CHASEDB --device cuda:0
```

5) Analyze results: we use DRIVE predictions to find optimal thresholding value:
```
python analyze_results.py --path_train_preds results/DRIVE/experiments/wnet_drive
                          --path_test_preds results/CHASEDB/experiments/wnet_drive_chasedb_pl
                          --train_dataset DRIVE --test_dataset CHASEDB
```

## 7. Evaluating your own model
We have made also an effort in making our evaluation protocol easy to use.
You just need to build your own probabilistic segmentations with your segmentation system and store training/test predictions in folders called `train_preds` and `trest_preds`.

Be careful: you need to produce segmentations for the test dataset, and also for the training dataset, which we use to find an optimal threshold.
Then you can call our code to compute performance.
If you used dataset `dataset_A` for training and you want to test on `dataset_B`, you would run:
```
python analyze_results.py --path_train_preds train_preds --path_test_preds test_preds
                          --train_dataset dataset_A --test_dataset dataset_B
```

Be very careful to use the same train/test splits as we are using here (check the csvs in the corresponding dataset folder), or you might be testing on training data.
Also, predictions should have the same exact name as the corresponding retinal images, but with a `.png` extension (otherwise the code will not find them).

## 8. Training a W-Net for Artery/Vein segmentation
In our work we train models on DRIVE and HRF, and we use a larger W-Net in this task.
Again, HRF is trained at image size `1024x1024`:
```
python train_cyclical.py --csv_train data/DRIVE/train_av.csv --model_name big_wnet
                         --cycle_len 40/50 --do_not_save False --save_path big_wnet_drive_av
                         --device cuda:0
```

```
python train_cyclical.py --csv_train data/HRF/train_av.csv --model_name big_wnet
                         --cycle_len 40/50 --do_not_save False --save_path big_wnet_hrf_av_1024
                         --im_size 1024 --batch_size 2 --grad_acc_steps 1  --device cuda:0
```

## 9. Generating Artery/Vein segmentations
This is similar to the vessel segmentation case, but calling `generate_av_results.py` instead::
```
python generate_av_results.py --config_file experiments/big_wnet_drive_av/config.cfg
                              --dataset DRIVE --device cuda:0
python generate_av_results.py --config_file experiments/big_wnet_drive_av/config.cfg
                              --dataset LES_AV --device cuda:0
```
and remember to set the image size for generating HRF segmentations:
```
python generate_av_results.py --config_file experiments/big_wnet_hrf_av_1024/config.cfg
                              --dataset HRF --im_size 1024 --device cuda:0
```

## 10. Generating vessel and A/V segmentations on your own data
To make it easy to construct segmentations on new data, we have also made available pretrained weights in the `experiments/` folder, and a script you can call on your own images:
```
python predict_one_image.py --model_path experiments/wnet_drive/
                            --im_path folder/my_image.jpg
                            --result_path my_results/
                            --mask_path folder/my_mask.jpg
                            --device cuda:0
                            --bin_thresh 0.42
```
The script uses a model trained on DRIVE by default, you can change it to
use a model that you would have trained on HRF (larger resolution but slower, see below).
You can optionally pass the path to a FOV mask (if you do not, the code builds one for you),
the device used for the forward pass of the network (defaults to CPU), and the binarizing threshold
(by default set to the optimal one in the DRIVE training set, 0.42).
If for instance you want to use a model trained on HRF, you will want to change the image size and the threshold as follows:
```
python predict_one_image.py --model_path experiments/wnet_hrf_1024/
                            --im_path folder/my_image.jpg
                            --result_path my_results/
                            --device cuda:0
                            --im_size 1024
                            --bin_thresh 0.3725
```
If you are interested in generating A/V segmentations, you can use a second script called `predict_one_image_av.py`.
The usage is very similar (on the CPU in this case, automatic mask computing):
```
python predict_one_image_av.py --model_path experiments/big_wnet_drive/
                              --im_path folder/my_image.jpg
                              --result_path my_results/
```
Note that there is no need to supply a threshold in this case, since we take the argmax of the probabilities to generate hard segmentations.

If instead you want to use a model trained on HRF (also provided) at a larger resolution of 1024x1024, you would run:
```
python predict_one_image_av.py --model_path experiments/big_wnet_hrf_av_1024/ 
                               --im_path folder/my_image.jpg 
                               --result_path my_results/
                               --im_size 1024
```
Using this model should result in finer arteries and veins delineations (although not necessarily more accuracy), which may be desirable if your data is of higher resolution than DRIVE.

## Structural-Saliency Self-Supervision

LwNet supports an optional auxiliary **structural-saliency self-supervision** task,
inspired by the RaffeSDG structural-saliency pretext task. It is **not** the full
RaffeSDG coupling architecture: no attention coupling is implemented.

During training, the network receives the (frequency-)augmented image, while a
Gaussian high-frequency-content (HFC) target is constructed from the
geometrically aligned **original** (non-frequency-augmented) RGB image. An
auxiliary decoder branching from U-Net 1's encoder reconstructs that target, and
the reconstruction is supervised with a plain mean-squared error:

`
L = Lseg(U1) + Lseg(U2) + lambda_sal * Lmse
`

Key properties:

1. the auxiliary decoder branches from U-Net 1's encoder and shares its
   bottleneck and skip features;
2. the structural target is a three-channel Gaussian HFC representation of the
   aligned original image (range [-1, +1], exactly -1 outside the FOV);
3. reconstruction uses 	orch.nn.functional.mse_loss (reduction='mean');
4. the auxiliary decoder executes **only during training** (never during
   validation or inference);
5. the decoder's parameters remain in the saved checkpoint, so inference must
   reconstruct the architecture from the saved config (handled automatically by
   get_arch_options/get_arch);
6. prediction output is still U-Net 2's vessel logits only.

Enable it with --structural_saliency (weight --structural_saliency_weight,
default 1.0; Gaussian kernel --structural_saliency_kernel/--structural_saliency_sigma,
defaults 27/9.0; HFC ratio --structural_saliency_ratio, default 4.0). It works
with or without --freesdg, and with every supported augmentation mode, so the
ablation RAFFE vs RAFFE + structural saliency is directly comparable.
