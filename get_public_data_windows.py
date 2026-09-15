"""Download and prepare the public datasets on Windows.

This is the Windows equivalent of ``get_public_data.py``.  Dataset layout,
splits, generated CSVs, and image processing intentionally match the original
script.  The download and archive setup uses Python's standard library instead
of Unix-only commands such as curl, wget, tar, unzip, mv, and rm.
"""

import os
import os.path as osp
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile

import pandas as pd
from tqdm import tqdm
from PIL import Image
from skimage import io
import numpy as np
from torchvision.transforms.functional import resize


def download(url, destination, label):
    """Download *url* to *destination* while displaying byte-level progress."""
    print('Downloading {}:'.format(label))
    print('  URL: {}'.format(url))
    print('  Destination: {}'.format(destination))

    def report_progress(block_count, block_size, total_size):
        downloaded = block_count * block_size
        if total_size > 0:
            percent = min(downloaded / total_size, 1.0) * 100
            status = '{:6.2f}% ({:.1f}/{:.1f} MB)'.format(
                percent, downloaded / 1024 ** 2, total_size / 1024 ** 2
            )
        else:
            status = '{:.1f} MB'.format(downloaded / 1024 ** 2)
        print('  Progress: {}'.format(status), end='\r', flush=True)

    urllib.request.urlretrieve(url, destination, reporthook=report_progress)
    size_mb = osp.getsize(destination) / 1024 ** 2
    print('  Progress: 100.00% ({:.1f} MB)'.format(size_mb))
    print('  Download complete.')


def extract_tar_data(archive, destination):
    """Extract the archive's ``deepdyn-master/data`` contents into destination."""
    with tarfile.open(archive, "r:gz") as tar:
        members = []
        prefix = "deepdyn-master/data/"
        for member in tar.getmembers():
            if not member.name.startswith(prefix):
                continue
            relative_name = member.name[len(prefix):]
            if not relative_name:
                continue
            member.name = relative_name
            members.append(member)
        tar.extractall(destination, members=members, filter='data')
    print('  Extraction complete: {}'.format(destination))


def move_contents(source, destination):
    """Move every item directly inside source into destination."""
    os.makedirs(destination, exist_ok=True)
    for name in os.listdir(source):
        source_path = osp.join(source, name)
        destination_path = osp.join(destination, name)
        if osp.isdir(destination_path) and not osp.islink(destination_path):
            shutil.rmtree(destination_path)
        elif osp.exists(destination_path) or osp.islink(destination_path):
            os.remove(destination_path)
        shutil.move(source_path, destination_path)


def move_replace(source, destination):
    """Move source to exactly destination, replacing an existing path."""
    if osp.isdir(destination) and not osp.islink(destination):
        shutil.rmtree(destination)
    elif osp.exists(destination) or osp.islink(destination):
        os.remove(destination)
    shutil.move(source, destination)


def extract_zip(archive_path, destination, label):
    """Extract a ZIP archive and report its progress."""
    print('Extracting {}:'.format(label))
    print('  Archive: {}'.format(archive_path))
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        total_size = sum(member.file_size for member in members)
        extracted_size = 0
        for member in members:
            archive.extract(member, destination)
            extracted_size += member.file_size
            if total_size:
                percent = extracted_size / total_size * 100
                print('  Progress: {:6.2f}%'.format(percent), end='\r', flush=True)
    print('  Progress: 100.00%')
    print('  Extraction complete: {}'.format(destination))


def replace_component(path, old, new):
    """Replace one directory component without assuming '/' separators."""
    parts = osp.normpath(path).split(osp.sep)
    return osp.join(*(new if part == old else part for part in parts))


def remove_if_present(path):
    if osp.exists(path):
        os.remove(path)


with tempfile.TemporaryDirectory() as temp_dir:
    os.makedirs('experiments', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    print('downloading data')

    deepdyn_archive = osp.join(temp_dir, 'deepdyn.tar.gz')
    download('https://codeload.github.com/sraashis/deepdyn/tar.gz/master', deepdyn_archive, 'DeepDyn public datasets')
    extract_tar_data(deepdyn_archive, 'data')

    shutil.rmtree('data/VEVIO')
    shutil.rmtree('data/DRIVE/splits')
    shutil.rmtree('data/STARE/splits')
    shutil.rmtree('data/STARE/labels-vk')
    shutil.rmtree('data/CHASEDB/splits')
    shutil.rmtree('data/AV-WIDE/splits')

    move_replace('data/STARE/labels-ah', 'data/STARE/manual')
    move_replace('data/STARE/stare-images', 'data/STARE/images')

    av_archive = osp.join(temp_dir, 'AV_groundTruth.zip')
    download('http://webeye.ophth.uiowa.edu/abramoff/AV_groundTruth.zip', av_archive, 'DRIVE artery/vein ground truth')
    extract_zip(av_archive, 'data/DRIVE', 'DRIVE artery/vein ground truth')
    os.remove(av_archive)
    manual_av = 'data/DRIVE/manual_av'
    move_contents('data/DRIVE/AV_groundTruth/training/av', manual_av)
    move_contents('data/DRIVE/AV_groundTruth/test/av', manual_av)
    shutil.rmtree('data/DRIVE/AV_groundTruth')

    hrf_archive = osp.join(temp_dir, 'all.zip')
    download('https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip', hrf_archive, 'HRF dataset')
    extract_zip(hrf_archive, 'data/HRF', 'HRF dataset')
    move_replace('data/HRF/manual1', 'data/HRF/manual')

    drhagis_archive = osp.join(temp_dir, 'DRHAGIS.zip')
    download('http://personalpages.manchester.ac.uk/staff/niall.p.mcloughlin/DRHAGIS.zip', drhagis_archive, 'DR-HAGIS dataset')
    extract_zip(drhagis_archive, 'data/DRHAGIS', 'DR-HAGIS dataset')
    move_replace('data/DRHAGIS/DRHAGIS', 'data/DR_HAGIS')
    shutil.rmtree('data/DRHAGIS')
    move_replace('data/DR_HAGIS/Fundus_Images', 'data/DR_HAGIS/images')
    move_replace('data/DR_HAGIS/Mask_images', 'data/DR_HAGIS/mask')
    move_replace('data/DR_HAGIS/Manual_Segmentations', 'data/DR_HAGIS/manual')
    remove_if_present('data/DR_HAGIS/.DS_Store')
    remove_if_present('data/DR_HAGIS/images/.DS_Store')
    remove_if_present('data/DR_HAGIS/manual/.DS_Store')
    remove_if_present('data/DR_HAGIS/mask/.DS_Store')
    move_replace('data/DR_HAGIS', 'data/DR-HAGIS')


print('preparing data')


# Process DRIVE data and generate CSVs.
path_ims = 'data/DRIVE/images'
path_masks = 'data/DRIVE/mask'
path_gts = 'data/DRIVE/manual'

all_im_names = sorted(os.listdir(path_ims))
all_mask_names = sorted(os.listdir(path_masks))
all_gt_names = sorted(os.listdir(path_gts))

all_im_names = [osp.join(path_ims, n) for n in all_im_names]
all_mask_names = [osp.join(path_masks, n) for n in all_mask_names]
all_gt_names = [osp.join(path_gts, n) for n in all_gt_names]

num_ims = len(all_im_names)
test_im_names = all_im_names[:num_ims // 2]
train_im_names = all_im_names[num_ims // 2:]
test_mask_names = all_mask_names[:num_ims // 2]
train_mask_names = all_mask_names[num_ims // 2:]
test_gt_names = all_gt_names[:num_ims // 2]
train_gt_names = all_gt_names[num_ims // 2:]

df_drive_all = pd.DataFrame({'im_paths': all_im_names,
                             'gt_paths': all_gt_names,
                             'mask_paths': all_mask_names})
df_drive_train = pd.DataFrame({'im_paths': train_im_names,
                               'gt_paths': train_gt_names,
                               'mask_paths': train_mask_names})
df_drive_test = pd.DataFrame({'im_paths': test_im_names,
                              'gt_paths': test_gt_names,
                              'mask_paths': test_mask_names})
df_drive_train, df_drive_val = df_drive_train[:16], df_drive_train[16:]

df_drive_train.to_csv('data/DRIVE/train.csv', index=False)
df_drive_val.to_csv('data/DRIVE/val.csv', index=False)
df_drive_test.to_csv('data/DRIVE/test.csv', index=False)
df_drive_all.to_csv('data/DRIVE/test_all.csv', index=False)

# Derive A/V splits from the vessel split.
df_drive_train.gt_paths = [replace_component(n, 'manual', 'manual_av').replace('manual1.gif', 'training.png')
                           for n in df_drive_train.gt_paths]
df_drive_val.gt_paths = [replace_component(n, 'manual', 'manual_av').replace('manual1.gif', 'training.png')
                         for n in df_drive_val.gt_paths]
df_drive_test.gt_paths = [replace_component(n, 'manual', 'manual_av').replace('manual1.gif', 'test.png')
                          for n in df_drive_test.gt_paths]
df_drive_train.to_csv('data/DRIVE/train_av.csv', index=False)
df_drive_val.to_csv('data/DRIVE/val_av.csv', index=False)
df_drive_test.to_csv('data/DRIVE/test_av.csv', index=False)
print('DRIVE prepared')

df_drive_train_val = pd.concat([df_drive_train, df_drive_val], axis=0)
for n in df_drive_train_val.gt_paths:
    x = io.imread(n)
    arteries = np.zeros_like(x[:, :, 0])
    veins = np.zeros_like(x[:, :, 0])
    unk = np.zeros_like(x[:, :, 0])
    av = np.zeros_like(x[:, :, 0])

    arteries[x[:, :, 0] == 255] = 255
    unk[x[:, :, 1] == 255] = 255
    veins[x[:, :, 2] == 255] = 255
    av[unk == 255] = 85
    av[arteries == 255] = 170
    av[veins == 255] = 255
    io.imsave(n, av)
print('DRIVE A/V prepared')


# Process CHASE-DB data.
path_ims = 'data/CHASEDB/images'
path_masks = 'data/CHASEDB/masks'
path_gts = 'data/CHASEDB/manual'

all_im_names = sorted(os.listdir(path_ims))
all_mask_names = sorted(os.listdir(path_masks))
all_gt_names = sorted(os.listdir(path_gts))
all_im_names = [osp.join(path_ims, n) for n in all_im_names]
all_mask_names = [osp.join(path_masks, n) for n in all_mask_names]
all_gt_names = [osp.join(path_gts, n) for n in all_gt_names if '1st' in n]

num_ims = len(all_im_names)
train_im_names = all_im_names[:8]
test_im_names = all_im_names[8:]
train_mask_names = all_mask_names[:8]
test_mask_names = all_mask_names[8:]
train_gt_names = all_gt_names[:8]
test_gt_names = all_gt_names[8:]

df_chasedb_all = pd.DataFrame({'im_paths': all_im_names,
                               'gt_paths': all_gt_names,
                               'mask_paths': all_mask_names})
df_chasedb_train = pd.DataFrame({'im_paths': train_im_names,
                                 'gt_paths': train_gt_names,
                                 'mask_paths': train_mask_names})
df_chasedb_test = pd.DataFrame({'im_paths': test_im_names,
                                'gt_paths': test_gt_names,
                                'mask_paths': test_mask_names})

num_ims = len(df_chasedb_train)
tr_ims = int(0.8 * num_ims)
df_chasedb_train, df_chasedb_val = df_chasedb_train[:tr_ims], df_chasedb_train[tr_ims:]
df_chasedb_train.to_csv('data/CHASEDB/train.csv', index=False)
df_chasedb_val.to_csv('data/CHASEDB/val.csv', index=False)
df_chasedb_test.to_csv('data/CHASEDB/test.csv', index=False)
df_chasedb_all.to_csv('data/CHASEDB/test_all.csv', index=False)
print('CHASE-DB prepared')


# Process HRF data and generate CSVs.
path_ims = 'data/HRF/images'
path_masks = 'data/HRF/mask'
path_gts = 'data/HRF/manual'
path_ims_resized = 'data/HRF/images_resized'
path_masks_resized = 'data/HRF/mask_resized'
path_gts_resized = 'data/HRF/manual_resized'
os.makedirs(path_ims_resized, exist_ok=True)
os.makedirs(path_masks_resized, exist_ok=True)
os.makedirs(path_gts_resized, exist_ok=True)

all_im_names = sorted(os.listdir(path_ims))
all_mask_names = sorted(os.listdir(path_masks))
all_gt_names = sorted(os.listdir(path_gts))
num_ims = len(all_im_names)
all_im_names = [osp.join(path_ims, n) for n in all_im_names]
all_mask_names = [osp.join(path_masks, n) for n in all_mask_names]
all_gt_names = [osp.join(path_gts, n) for n in all_gt_names]

df_hrf_all = pd.DataFrame({'im_paths': all_im_names,
                           'gt_paths': all_gt_names,
                           'mask_paths': all_mask_names})
train_im_names = all_im_names[:3 * 5]
test_im_names = all_im_names[3 * 5:]
train_mask_names = all_mask_names[:3 * 5]
test_mask_names = all_mask_names[3 * 5:]
train_gt_names = all_gt_names[:3 * 5]
test_gt_names = all_gt_names[3 * 5:]

train_im_names_resized = [replace_component(n, 'images', 'images_resized') for n in train_im_names]
train_mask_names_resized = [replace_component(n, 'mask', 'mask_resized') for n in train_mask_names]
train_gt_names_resized = [replace_component(n, 'manual', 'manual_resized') for n in train_gt_names]
df_hrf_train = pd.DataFrame({'im_paths': train_im_names_resized,
                             'gt_paths': train_gt_names_resized,
                             'mask_paths': train_mask_names_resized})
df_hrf_test = pd.DataFrame({'im_paths': test_im_names,
                            'gt_paths': test_gt_names,
                            'mask_paths': test_mask_names})

num_ims = len(df_hrf_train)
tr_ims = int(0.8 * num_ims)
df_hrf_train, df_hrf_val = df_hrf_train[:tr_ims], df_hrf_train[tr_ims:]
df_hrf_train.to_csv('data/HRF/train.csv', index=False)
df_hrf_val.to_csv('data/HRF/val.csv', index=False)
df_hrf_test.to_csv('data/HRF/test.csv', index=False)
df_hrf_all.to_csv('data/HRF/test_all.csv', index=False)

# Needed for AUC analysis on the training set.
df_hrf_train_full_res = pd.DataFrame({'im_paths': train_im_names,
                                      'gt_paths': train_gt_names,
                                      'mask_paths': train_mask_names})
df_hrf_train_full_res, df_hrf_val_full_res = df_hrf_train_full_res[:tr_ims], df_hrf_train_full_res[tr_ims:]
df_hrf_train_full_res.to_csv('data/HRF/train_full_res.csv', index=False)
df_hrf_val_full_res.to_csv('data/HRF/val_full_res.csv', index=False)

print('Resizing HRF images (**only** for training, but we resize all because A/V training set is test set on Vessels)\n')
for i in tqdm(range(len(all_im_names))):
    im_name = all_im_names[i]
    im_name_out = replace_component(im_name, 'images', 'images_resized')
    im = Image.open(im_name)
    im_res = resize(im, size=(im.size[1] // 2, im.size[0] // 2), interpolation=Image.BICUBIC)
    im_res.save(im_name_out)

    mask_name = replace_component(im_name, 'images', 'mask').replace('.JPG', '_mask.tif').replace('.jpg', '_mask.tif')
    mask_name_out = replace_component(mask_name, 'mask', 'mask_resized')
    mask = Image.open(mask_name)
    mask_res = resize(mask, size=(mask.size[1] // 2, mask.size[0] // 2), interpolation=Image.NEAREST)
    mask = Image.fromarray(np.array(mask)[:, :, 0])
    mask_res = Image.fromarray(np.array(mask_res)[:, :, 0])
    mask.save(mask_name)
    mask_res.save(mask_name_out)

    gt_name = replace_component(im_name, 'images', 'manual').replace('.JPG', '.tif').replace('.jpg', '.tif')
    gt_name_out = replace_component(gt_name, 'manual', 'manual_resized')
    gt = Image.open(gt_name)
    gt_res = resize(gt, size=(gt.size[1] // 2, gt.size[0] // 2), interpolation=Image.NEAREST)
    gt_res.save(gt_name_out)
print('HRF prepared')


# Prepare A/V ground truth for HRF. Training images become the A/V test set.
path_gts_resized = 'data/HRF/manual_av_resized'
os.makedirs(path_gts_resized, exist_ok=True)
print('preparing HRF training set for A/V segmentation:')
for i in tqdm(range(len(test_im_names))):
    n = all_gt_names[i]
    n_av = replace_component(n, 'manual', 'manual_av').replace('.tif', '_AVmanual.png')
    x = io.imread(n_av)

    arteries = np.zeros_like(x[:, :, 0])
    veins = np.zeros_like(x[:, :, 0])
    unk = np.zeros_like(x[:, :, 0])
    av = np.zeros_like(x[:, :, 0])
    arteries[x[:, :, 0] == 255] = 255
    unk[x[:, :, 1] == 255] = 255
    veins[x[:, :, 2] == 255] = 255
    av[unk == 255] = 85
    av[arteries == 255] = 170
    av[veins == 255] = 255

    av = Image.fromarray(av)
    av = resize(av, size=(av.size[1] // 2, av.size[0] // 2), interpolation=Image.NEAREST)
    av.save(replace_component(n_av, 'manual_av', 'manual_av_resized'))

av_test = pd.concat([df_hrf_train, df_hrf_val], axis=0)
av_im_paths = [replace_component(n, 'images_resized', 'images') for n in av_test.im_paths]
av_gt_paths = [replace_component(n, 'manual_resized', 'manual_av') for n in av_test.gt_paths]
av_mask_paths = [replace_component(n, 'mask_resized', 'mask') for n in av_test.mask_paths]
av_test_df = pd.DataFrame(list(zip(av_im_paths, av_gt_paths, av_mask_paths)),
                          columns=['im_paths', 'gt_paths', 'mask_paths'])

av_train, av_val = df_hrf_test[:24], df_hrf_test[24:]
av_train_im_paths = [replace_component(n, 'images', 'images_resized') for n in av_train.im_paths]
av_train_gt_paths = [replace_component(n, 'manual_av', 'manual_av_resized') for n in av_train.gt_paths]
av_train_mask_paths = [replace_component(n, 'mask', 'mask_resized') for n in av_train.mask_paths]
av_train_df = pd.DataFrame(list(zip(av_train_im_paths, av_train_gt_paths, av_train_mask_paths)),
                           columns=['im_paths', 'gt_paths', 'mask_paths'])

av_val_im_paths = [replace_component(n, 'images', 'images_resized') for n in av_val.im_paths]
av_val_gt_paths = [replace_component(n, 'manual_av', 'manual_av_resized') for n in av_val.gt_paths]
av_val_mask_paths = [replace_component(n, 'mask', 'mask_resized') for n in av_val.mask_paths]
av_val_df = pd.DataFrame(list(zip(av_val_im_paths, av_val_gt_paths, av_val_mask_paths)),
                         columns=['im_paths', 'gt_paths', 'mask_paths'])

av_train_df.to_csv('data/HRF/train_av.csv', index=False)
av_val_df.to_csv('data/HRF/val_av.csv', index=False)
av_test_df.to_csv('data/HRF/test_av.csv')
print('HRF A/V prepared')


# Process STARE data.
path_ims = 'data/STARE/images'
path_masks = 'data/STARE/mask'
path_gts = 'data/STARE/manual'
all_im_names = sorted(os.listdir(path_ims))
all_mask_names = sorted(os.listdir(path_masks))
all_gt_names = sorted(os.listdir(path_gts))
all_im_names = [osp.join(path_ims, n) for n in all_im_names]
all_mask_names = [osp.join(path_masks, n) for n in all_mask_names]
all_gt_names = [osp.join(path_gts, n) for n in all_gt_names]
df_stare_all = pd.DataFrame({'im_paths': all_im_names,
                             'gt_paths': all_gt_names,
                             'mask_paths': all_mask_names})
df_stare_all.to_csv('data/STARE/test_all.csv', index=False)
print('STARE prepared')


# Process AV-WIDE data.
path_ims = 'data/AV-WIDE/images'
path_masks = 'data/AV-WIDE/masks'
os.makedirs(path_masks, exist_ok=True)
path_gts = 'data/AV-WIDE/manual'
test_im_names = sorted(os.listdir(path_ims))
test_gt_names = sorted(os.listdir(path_gts))
for n in test_im_names:
    im = Image.open(osp.join(path_ims, n))
    mask = 255 * np.ones((im.size[1], im.size[0]), dtype=np.uint8)
    Image.fromarray(mask).save(osp.join(path_masks, n))

test_mask_names = [osp.join(path_masks, n) for n in test_im_names]
test_im_names = [osp.join(path_ims, n) for n in test_im_names]
test_gt_names = [osp.join(path_gts, n) for n in test_gt_names]
df_wide_test = pd.DataFrame({'im_paths': test_im_names,
                             'gt_paths': test_gt_names,
                             'mask_paths': test_mask_names})
df_wide_test.to_csv('data/AV-WIDE/test_all.csv', index=False)
print('AV-WIDE prepared')


# Process DR-HAGIS data.
path_ims = 'data/DR-HAGIS/images'
path_masks = 'data/DR-HAGIS/mask'
path_gts = 'data/DR-HAGIS/manual'
all_im_names = sorted(os.listdir(path_ims), key=lambda s: s.split('_')[0])
all_mask_names = sorted(os.listdir(path_masks), key=lambda s: s.split('_')[0])
all_gt_names = sorted(os.listdir(path_gts), key=lambda s: s.split('_')[0])
all_im_names = [osp.join(path_ims, n) for n in all_im_names]
all_mask_names = [osp.join(path_masks, n) for n in all_mask_names]
all_gt_names = [osp.join(path_gts, n) for n in all_gt_names]
df_drhagis_all = pd.DataFrame({'im_paths': all_im_names,
                               'gt_paths': all_gt_names,
                               'mask_paths': all_mask_names})
df_drhagis_all.to_csv('data/DR-HAGIS/test_all.csv', index=False)
print('DR-HAGIS prepared')


print('All public data prepared, ready to go.')
print('NOTE: The Les-AV dataset is hosted at figshare now; download it manually before preparing it.')
print(104 * '-')

# LES-AV has been removed from the old public URL and is hosted at Figshare.
# Download LES-AV.zip from the following URL and place it beside this script:
# https://figshare.com/articles/dataset/LES-AV_dataset/11857698
#
# The original Unix commands are implemented below with Python's standard
# library so the same section can run on Windows.
def prepare_les_av(archive_path='LES-AV.zip'):
    """Prepare LES-AV from the manually downloaded Figshare ZIP archive."""
    if not osp.isfile(archive_path):
        print('NOTE: LES-AV was not prepared because LES-AV.zip was not found.')
        print('Download it from https://figshare.com/articles/dataset/LES-AV_dataset/11857698')
        print('Place LES-AV.zip beside this script and run it again to prepare LES-AV.')
        return False

    staging_path = 'data/LES_AV'
    dataset_path = osp.join(staging_path, 'LES-AV')
    output_path = 'data/LES-AV'

    if osp.exists(staging_path):
        shutil.rmtree(staging_path)
    os.makedirs(staging_path, exist_ok=True)

    print('preparing LES-AV')
    extract_zip(archive_path, staging_path, 'LES-AV dataset')
    os.remove(archive_path)

    # Figshare archives may include macOS metadata alongside the dataset.
    remove_tree = osp.join(staging_path, '__MACOSX')
    if osp.exists(remove_tree):
        shutil.rmtree(remove_tree)

    if not osp.isdir(dataset_path):
        raise FileNotFoundError(
            'Expected the ZIP archive to contain a top-level LES-AV directory: '
            + osp.abspath(archive_path)
        )

    if osp.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    move_replace(osp.join(dataset_path, 'images'), osp.join(output_path, 'images'))
    move_replace(osp.join(dataset_path, 'masks'), osp.join(output_path, 'mask'))
    move_replace(osp.join(dataset_path, 'vessel-segmentations'), osp.join(output_path, 'manual'))
    move_replace(osp.join(dataset_path, 'arteries-and-veins'), osp.join(output_path, 'manual_av'))
    shutil.rmtree(staging_path)

    path_ims = osp.join(output_path, 'images')
    path_masks = osp.join(output_path, 'mask')
    path_gts = osp.join(output_path, 'manual')
    all_im_names = sorted(os.listdir(path_ims))
    all_mask_names = sorted(os.listdir(path_masks))
    all_gt_names = sorted(os.listdir(path_gts))

    all_im_names = [osp.join(path_ims, n) for n in all_im_names]
    all_mask_names = [osp.join(path_masks, n) for n in all_mask_names]
    all_gt_names = [osp.join(path_gts, n) for n in all_gt_names]

    df_lesav_all = pd.DataFrame({'im_paths': all_im_names,
                                 'gt_paths': all_gt_names,
                                 'mask_paths': all_mask_names})
    df_lesav_all.to_csv(osp.join(output_path, 'test_all.csv'), index=False)

    # Create the A/V CSV using the same image and mask paths.
    df_lesav_all.gt_paths = [replace_component(n, 'manual', 'manual_av')
                             for n in df_lesav_all.gt_paths]
    df_lesav_all.to_csv(osp.join(output_path, 'test_all_av.csv'), index=None)
    print('LES-AV prepared')
    return True


prepare_les_av()
