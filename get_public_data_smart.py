"""Download and prepare the public datasets used by lwnet.

Each dataset is prepared independently.  A dataset is skipped when all of its
final artifacts are present, while archives are kept in ``.public_data_cache``
so a failed dataset can be retried without downloading its source again.
"""

import os
import os.path as osp
import shutil
import tarfile
import tempfile
import traceback
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
from torchvision.transforms.functional import resize
from tqdm import tqdm


DATA_DIR = 'data'
CACHE_DIR = '.public_data_cache'


def has_files(path):
    return osp.isdir(path) and any(osp.isfile(osp.join(path, name))
                                   for name in os.listdir(path))


def dataset_is_complete(dataset, directories, files):
    dataset_path = osp.join(DATA_DIR, dataset)
    if not (all(has_files(osp.join(dataset_path, directory))
                for directory in directories)
            and all(osp.isfile(osp.join(dataset_path, file_name))
                    for file_name in files)):
        return False

    # CSVs are the manifest for the processed files.  Validate their paths so
    # a deleted image or annotation causes only this dataset to be retried.
    for file_name in files:
        try:
            frame = pd.read_csv(osp.join(dataset_path, file_name))
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            return False
        for column in ('im_paths', 'gt_paths', 'mask_paths'):
            if column in frame:
                if not all(osp.isfile(path) for path in frame[column].dropna()):
                    return False
    return True


def remove_if_present(path):
    if osp.isdir(path) and not osp.islink(path):
        shutil.rmtree(path)
    elif osp.exists(path) or osp.islink(path):
        os.remove(path)


def replace_component(path, old, new):
    """Replace one directory component without assuming a path separator."""
    parts = osp.normpath(path).split(osp.sep)
    return osp.join(*(new if part == old else part for part in parts))


def replace_with_av_annotation(path, source_directory, directory):
    """Point a vessel annotation path at its downloaded A/V annotation."""
    path = replace_component(path, source_directory, directory)
    return path.replace('.tif', '_AVmanual.png')


def move_replace(source, destination):
    """Move source to destination, replacing an existing path."""
    remove_if_present(destination)
    os.makedirs(osp.dirname(destination), exist_ok=True)
    shutil.move(source, destination)


def move_contents(source, destination):
    """Move all direct children of source to destination."""
    os.makedirs(destination, exist_ok=True)
    for name in os.listdir(source):
        move_replace(osp.join(source, name), osp.join(destination, name))


def download(url, destination, label):
    """Download a URL atomically, keeping a partial download out of the cache."""
    print('Downloading {}:'.format(label))
    print('  URL: {}'.format(url))
    print('  Destination: {}'.format(destination))
    partial_destination = destination + '.part'

    def report_progress(block_count, block_size, total_size):
        downloaded = block_count * block_size
        if total_size > 0:
            percent = min(downloaded / total_size, 1.0) * 100
            status = '{:6.2f}% ({:.1f}/{:.1f} MB)'.format(
                percent, downloaded / 1024 ** 2, total_size / 1024 ** 2)
        else:
            status = '{:.1f} MB'.format(downloaded / 1024 ** 2)
        print('  Progress: {}'.format(status), end='\r', flush=True)

    try:
        urllib.request.urlretrieve(url, partial_destination,
                                   reporthook=report_progress)
        os.replace(partial_destination, destination)
    finally:
        remove_if_present(partial_destination)
    print('  Progress: 100.00%')
    print('  Download complete.')


def cached_download(url, filename, label):
    os.makedirs(CACHE_DIR, exist_ok=True)
    destination = osp.join(CACHE_DIR, filename)
    if osp.isfile(destination) and osp.getsize(destination) > 0:
        if (filename.endswith('.zip') and zipfile.is_zipfile(destination)) or \
                (filename.endswith(('.tar', '.tar.gz', '.tgz')) and
                 tarfile.is_tarfile(destination)):
            print('Using cached {}: {}'.format(label, destination))
            return destination
        print('Discarding invalid cached {}: {}'.format(label, destination))
        remove_if_present(destination)
    download(url, destination, label)
    if ((filename.endswith('.zip') and not zipfile.is_zipfile(destination)) or
            (filename.endswith(('.tar', '.tar.gz', '.tgz')) and
             not tarfile.is_tarfile(destination))):
        remove_if_present(destination)
        raise ValueError('downloaded file is not a valid archive: {}'.format(filename))
    return destination


def cached_download_any(urls, filename, label):
    """Try download sources in order and keep the first valid ZIP archive."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    destination = osp.join(CACHE_DIR, filename)
    if osp.isfile(destination) and zipfile.is_zipfile(destination):
        print('Using cached {}: {}'.format(label, destination))
        return destination

    remove_if_present(destination)
    errors = []
    for url in urls:
        try:
            download(url, destination, label)
            if not zipfile.is_zipfile(destination):
                raise ValueError('downloaded file is not a valid ZIP archive')
            return destination
        except Exception as error:
            errors.append('{} ({})'.format(url, error))
            remove_if_present(destination)
    raise RuntimeError('Could not download {}: {}'.format(label, '; '.join(errors)))


def extract_zip(archive_path, destination):
    """Extract a ZIP while rejecting paths that escape destination."""
    os.makedirs(destination, exist_ok=True)
    destination = osp.abspath(destination)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_name = member.filename.replace('/', osp.sep)
            target = osp.abspath(osp.join(destination, member_name))
            if osp.commonpath([destination, target]) != destination:
                raise ValueError('Unsafe path in ZIP archive: {}'.format(member.filename))
        archive.extractall(destination)


def extract_deepdyn_dataset(archive_path, dataset):
    """Extract only one dataset from the shared DeepDyn archive."""
    destination = osp.abspath(DATA_DIR)
    prefix = 'deepdyn-master/data/{}/'.format(dataset)
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive.getmembers():
            if not member.name.startswith(prefix):
                continue
            relative_name = member.name[len('deepdyn-master/data/'):]
            target = osp.abspath(osp.join(destination, relative_name))
            if osp.commonpath([destination, target]) != destination:
                raise ValueError('Unsafe path in TAR archive: {}'.format(member.name))
            member.name = relative_name
            archive.extract(member, destination, filter='data')


def find_named_path(root, name):
    """Find a file or directory by basename below root."""
    for current_root, directories, files in os.walk(root):
        if name in directories:
            return osp.join(current_root, name)
        if name in files:
            return osp.join(current_root, name)
    return None


def ensure_full_image_masks(images_path, masks_path):
    """Create full-image masks when a public dataset has no FOV masks."""
    os.makedirs(masks_path, exist_ok=True)
    for name in sorted(os.listdir(images_path)):
        output = osp.join(masks_path, name)
        if osp.isfile(output):
            continue
        with Image.open(osp.join(images_path, name)) as image:
            mask = 255 * np.ones((image.size[1], image.size[0]), dtype=np.uint8)
        Image.fromarray(mask).save(output)


def ensure_deepdyn_dataset(dataset, source_directories):
    dataset_path = osp.join(DATA_DIR, dataset)
    if all(has_files(osp.join(dataset_path, directory))
           for directory in source_directories):
        return

    archive = cached_download(
        'https://codeload.github.com/sraashis/deepdyn/tar.gz/master',
        'deepdyn.tar.gz', 'DeepDyn public datasets')
    os.makedirs(DATA_DIR, exist_ok=True)
    extract_deepdyn_dataset(archive, dataset)


def prepare_drive():
    dataset = 'DRIVE'
    if dataset_is_complete(
            dataset,
            ['images', 'mask', 'manual', 'manual_av'],
            ['train.csv', 'val.csv', 'test.csv', 'test_all.csv',
             'train_av.csv', 'val_av.csv', 'test_av.csv']):
        print('DRIVE already prepared; skipping download and processing.')
        return

    ensure_deepdyn_dataset(dataset, ['images', 'mask', 'manual'])
    remove_if_present(osp.join(DATA_DIR, dataset, 'splits'))
    manual_av = osp.join(DATA_DIR, dataset, 'manual_av')
    manual_count = len(os.listdir(osp.join(DATA_DIR, dataset, 'manual')))
    manual_av_count = len(os.listdir(manual_av)) if osp.isdir(manual_av) else 0
    if manual_av_count < manual_count:
        archive = cached_download(
            'http://webeye.ophth.uiowa.edu/abramoff/AV_groundTruth.zip',
            'AV_groundTruth.zip', 'DRIVE artery/vein ground truth')
        with tempfile.TemporaryDirectory() as staging:
            extract_zip(archive, staging)
            av_root = find_named_path(staging, 'AV_groundTruth')
            if not av_root:
                raise FileNotFoundError('AV_groundTruth directory not found in archive')
            os.makedirs(manual_av, exist_ok=True)
            move_contents(osp.join(av_root, 'training', 'av'), manual_av)
            move_contents(osp.join(av_root, 'test', 'av'), manual_av)

    process_drive()


def process_drive():
    path_ims = osp.join(DATA_DIR, 'DRIVE', 'images')
    path_masks = osp.join(DATA_DIR, 'DRIVE', 'mask')
    path_gts = osp.join(DATA_DIR, 'DRIVE', 'manual')
    all_im_names = sorted(os.listdir(path_ims))
    all_mask_names = sorted(os.listdir(path_masks))
    all_gt_names = sorted(os.listdir(path_gts))
    if not (len(all_im_names) == len(all_mask_names) == len(all_gt_names)):
        raise ValueError('DRIVE image, mask, and ground-truth counts differ')

    all_im_names = [osp.join(path_ims, name) for name in all_im_names]
    all_mask_names = [osp.join(path_masks, name) for name in all_mask_names]
    all_gt_names = [osp.join(path_gts, name) for name in all_gt_names]
    num_ims = len(all_im_names)
    test_im_names, train_im_names = all_im_names[:num_ims // 2], all_im_names[num_ims // 2:]
    test_mask_names, train_mask_names = all_mask_names[:num_ims // 2], all_mask_names[num_ims // 2:]
    test_gt_names, train_gt_names = all_gt_names[:num_ims // 2], all_gt_names[num_ims // 2:]

    df_all = pd.DataFrame({'im_paths': all_im_names, 'gt_paths': all_gt_names,
                           'mask_paths': all_mask_names})
    df_train = pd.DataFrame({'im_paths': train_im_names, 'gt_paths': train_gt_names,
                             'mask_paths': train_mask_names})
    df_test = pd.DataFrame({'im_paths': test_im_names, 'gt_paths': test_gt_names,
                            'mask_paths': test_mask_names})
    df_train, df_val = df_train[:16], df_train[16:]
    dataset_path = osp.join(DATA_DIR, 'DRIVE')
    df_train.to_csv(osp.join(dataset_path, 'train.csv'), index=False)
    df_val.to_csv(osp.join(dataset_path, 'val.csv'), index=False)
    df_test.to_csv(osp.join(dataset_path, 'test.csv'), index=False)
    df_all.to_csv(osp.join(dataset_path, 'test_all.csv'), index=False)

    for frame, test_name in ((df_train, 'training.png'),
                             (df_val, 'training.png'),
                             (df_test, 'test.png')):
        frame.gt_paths = [replace_component(name, 'manual', 'manual_av')
                          .replace('manual1.gif', test_name)
                          for name in frame.gt_paths]
    df_train.to_csv(osp.join(dataset_path, 'train_av.csv'), index=False)
    df_val.to_csv(osp.join(dataset_path, 'val_av.csv'), index=False)
    df_test.to_csv(osp.join(dataset_path, 'test_av.csv'), index=False)

    for name in pd.concat([df_train, df_val]).gt_paths:
        x = io.imread(name)
        if x.ndim < 3:
            continue
        arteries = np.zeros_like(x[:, :, 0])
        veins = np.zeros_like(x[:, :, 0])
        unknown = np.zeros_like(x[:, :, 0])
        av = np.zeros_like(x[:, :, 0])
        arteries[x[:, :, 0] == 255] = 255
        unknown[x[:, :, 1] == 255] = 255
        veins[x[:, :, 2] == 255] = 255
        av[unknown == 255] = 85
        av[arteries == 255] = 170
        av[veins == 255] = 255
        io.imsave(name, av)
    print('DRIVE prepared')


def prepare_chasedb():
    dataset = 'CHASEDB'
    if dataset_is_complete(dataset, ['images', 'masks', 'manual'],
                            ['train.csv', 'val.csv', 'test.csv', 'test_all.csv']):
        print('CHASE-DB already prepared; skipping download and processing.')
        return
    ensure_deepdyn_dataset(dataset, ['images', 'manual'])
    ensure_full_image_masks(
        osp.join(DATA_DIR, dataset, 'images'),
        osp.join(DATA_DIR, dataset, 'masks'))
    remove_if_present(osp.join(DATA_DIR, dataset, 'splits'))
    process_chasedb()


def process_chasedb():
    dataset_path = osp.join(DATA_DIR, 'CHASEDB')
    path_ims = osp.join(dataset_path, 'images')
    path_masks = osp.join(dataset_path, 'masks')
    path_gts = osp.join(dataset_path, 'manual')
    all_im_names = sorted(os.listdir(path_ims))
    all_mask_names = sorted(os.listdir(path_masks))
    all_gt_names = sorted(name for name in os.listdir(path_gts) if '1st' in name)
    if not (len(all_im_names) == len(all_mask_names) == len(all_gt_names)):
        raise ValueError('CHASE-DB image, mask, and ground-truth counts differ')
    all_im_names = [osp.join(path_ims, name) for name in all_im_names]
    all_mask_names = [osp.join(path_masks, name) for name in all_mask_names]
    all_gt_names = [osp.join(path_gts, name) for name in all_gt_names]
    train_im_names, test_im_names = all_im_names[:8], all_im_names[8:]
    train_mask_names, test_mask_names = all_mask_names[:8], all_mask_names[8:]
    train_gt_names, test_gt_names = all_gt_names[:8], all_gt_names[8:]
    df_all = pd.DataFrame({'im_paths': all_im_names, 'gt_paths': all_gt_names,
                           'mask_paths': all_mask_names})
    df_train = pd.DataFrame({'im_paths': train_im_names, 'gt_paths': train_gt_names,
                             'mask_paths': train_mask_names})
    df_test = pd.DataFrame({'im_paths': test_im_names, 'gt_paths': test_gt_names,
                            'mask_paths': test_mask_names})
    tr_ims = int(0.8 * len(df_train))
    df_train, df_val = df_train[:tr_ims], df_train[tr_ims:]
    df_train.to_csv(osp.join(dataset_path, 'train.csv'), index=False)
    df_val.to_csv(osp.join(dataset_path, 'val.csv'), index=False)
    df_test.to_csv(osp.join(dataset_path, 'test.csv'), index=False)
    df_all.to_csv(osp.join(dataset_path, 'test_all.csv'), index=False)
    print('CHASE-DB prepared')


def prepare_hrf():
    dataset = 'HRF'
    if dataset_is_complete(
            dataset,
            ['images', 'mask', 'manual', 'manual_av', 'images_resized',
             'mask_resized', 'manual_resized', 'manual_av_resized'],
            ['train.csv', 'val.csv', 'test.csv', 'test_all.csv',
             'train_full_res.csv', 'val_full_res.csv',
             'train_av.csv', 'val_av.csv', 'test_av.csv']):
        print('HRF already prepared; skipping download and processing.')
        return

    dataset_path = osp.join(DATA_DIR, dataset)
    ensure_deepdyn_dataset(dataset, ['images', 'mask', 'manual'])
    if not all(has_files(osp.join(dataset_path, directory))
               for directory in ['images', 'mask', 'manual']):
        archive = cached_download(
            'https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip',
            'hrf.zip', 'HRF dataset')
        with tempfile.TemporaryDirectory() as staging:
            extract_zip(archive, staging)
            for source_name, destination_name in (
                    ('images', 'images'), ('mask', 'mask'), ('manual1', 'manual')):
                source = find_named_path(staging, source_name)
                if not source:
                    raise FileNotFoundError(
                        '{} directory not found in HRF archive'.format(source_name))
                move_replace(source, osp.join(dataset_path, destination_name))

    manual_av = osp.join(dataset_path, 'manual_av')
    if not has_files(manual_av):
        archive = cached_download_any(
            [
                'https://github.com/rubenhx/av-segmentation/archive/refs/heads/master.zip',
                'http://iflexis.com/downloads/HRF_AV_GT.zip',
            ],
            'HRF_AV_GT.zip', 'HRF artery/vein ground truth')
        with tempfile.TemporaryDirectory() as staging:
            extract_zip(archive, staging)
            source = find_named_path(staging, 'HRF_AV_GT')
            if not source:
                source = find_named_path(staging, 'HFR_AV_GT')
            if not source:
                raise FileNotFoundError('HRF_AV_GT directory not found in archive')
            move_replace(source, manual_av)
    process_hrf()


def process_hrf():
    dataset_path = osp.join(DATA_DIR, 'HRF')
    path_ims = osp.join(dataset_path, 'images')
    path_masks = osp.join(dataset_path, 'mask')
    path_gts = osp.join(dataset_path, 'manual')
    path_ims_resized = osp.join(dataset_path, 'images_resized')
    path_masks_resized = osp.join(dataset_path, 'mask_resized')
    path_gts_resized = osp.join(dataset_path, 'manual_resized')
    path_gts_av_resized = osp.join(dataset_path, 'manual_av_resized')
    for path in [path_ims_resized, path_masks_resized, path_gts_resized,
                 path_gts_av_resized]:
        os.makedirs(path, exist_ok=True)

    all_im_names = sorted(os.listdir(path_ims))
    all_mask_names = sorted(os.listdir(path_masks))
    all_gt_names = sorted(os.listdir(path_gts))
    if not (len(all_im_names) == len(all_mask_names) == len(all_gt_names)):
        raise ValueError('HRF image, mask, and ground-truth counts differ')
    all_im_names = [osp.join(path_ims, name) for name in all_im_names]
    all_mask_names = [osp.join(path_masks, name) for name in all_mask_names]
    all_gt_names = [osp.join(path_gts, name) for name in all_gt_names]
    df_all = pd.DataFrame({'im_paths': all_im_names, 'gt_paths': all_gt_names,
                           'mask_paths': all_mask_names})

    train_im_names, test_im_names = all_im_names[:15], all_im_names[15:]
    train_mask_names, test_mask_names = all_mask_names[:15], all_mask_names[15:]
    train_gt_names, test_gt_names = all_gt_names[:15], all_gt_names[15:]
    train_im_resized = [replace_component(name, 'images', 'images_resized')
                        for name in train_im_names]
    train_mask_resized = [replace_component(name, 'mask', 'mask_resized')
                          for name in train_mask_names]
    train_gt_resized = [replace_component(name, 'manual', 'manual_resized')
                        for name in train_gt_names]
    df_train = pd.DataFrame({'im_paths': train_im_resized,
                             'gt_paths': train_gt_resized,
                             'mask_paths': train_mask_resized})
    df_test = pd.DataFrame({'im_paths': test_im_names, 'gt_paths': test_gt_names,
                            'mask_paths': test_mask_names})
    tr_ims = int(0.8 * len(df_train))
    df_train, df_val = df_train[:tr_ims], df_train[tr_ims:]
    df_train.to_csv(osp.join(dataset_path, 'train.csv'), index=False)
    df_val.to_csv(osp.join(dataset_path, 'val.csv'), index=False)
    df_test.to_csv(osp.join(dataset_path, 'test.csv'), index=False)
    df_all.to_csv(osp.join(dataset_path, 'test_all.csv'), index=False)

    df_train_full = pd.DataFrame({'im_paths': train_im_names,
                                  'gt_paths': train_gt_names,
                                  'mask_paths': train_mask_names})
    df_train_full, df_val_full = df_train_full[:tr_ims], df_train_full[tr_ims:]
    df_train_full.to_csv(osp.join(dataset_path, 'train_full_res.csv'), index=False)
    df_val_full.to_csv(osp.join(dataset_path, 'val_full_res.csv'), index=False)

    print('Resizing HRF images:')
    for im_name in tqdm(all_im_names):
        im_out = replace_component(im_name, 'images', 'images_resized')
        im = Image.open(im_name)
        resize(im, size=(im.size[1] // 2, im.size[0] // 2),
               interpolation=Image.BICUBIC).save(im_out)

        mask_name = (replace_component(im_name, 'images', 'mask')
                     .replace('.JPG', '_mask.tif').replace('.jpg', '_mask.tif'))
        mask_out = replace_component(mask_name, 'mask', 'mask_resized')
        with Image.open(mask_name) as mask:
            mask_resized = resize(
                mask, size=(mask.size[1] // 2, mask.size[0] // 2),
                interpolation=Image.NEAREST)
            mask_array = np.array(mask)
            mask_resized_array = np.array(mask_resized)
            mask_was_multichannel = mask_array.ndim == 3
        if mask_array.ndim == 3:
            mask_array = mask_array[:, :, 0]
        if mask_resized_array.ndim == 3:
            mask_resized_array = mask_resized_array[:, :, 0]
        if mask_was_multichannel:
            Image.fromarray(mask_array).save(mask_name)
        Image.fromarray(mask_resized_array).save(mask_out)

        gt_name = (replace_component(im_name, 'images', 'manual')
                   .replace('.JPG', '.tif').replace('.jpg', '.tif'))
        gt_out = replace_component(gt_name, 'manual', 'manual_resized')
        gt = Image.open(gt_name)
        resize(gt, size=(gt.size[1] // 2, gt.size[0] // 2),
               interpolation=Image.NEAREST).save(gt_out)

    print('Preparing HRF artery/vein data:')
    for gt_name in tqdm(test_gt_names):
        source = replace_component(gt_name, 'manual', 'manual_av')
        source = source.replace('.tif', '_AVmanual.png')
        output = replace_component(source, 'manual_av', 'manual_av_resized')
        x = io.imread(source)
        if x.ndim < 3:
            av = x
        else:
            av = np.zeros_like(x[:, :, 0])
            av[x[:, :, 1] == 255] = 85
            av[x[:, :, 0] == 255] = 170
            av[x[:, :, 2] == 255] = 255
        av = Image.fromarray(av)
        resize(av, size=(av.size[1] // 2, av.size[0] // 2),
               interpolation=Image.NEAREST).save(output)

    av_test = pd.concat([df_train, df_val], axis=0)
    av_test_df = pd.DataFrame({
        'im_paths': [replace_component(name, 'images_resized', 'images')
                     for name in av_test.im_paths],
        'gt_paths': [replace_with_av_annotation(name, 'manual_resized', 'manual_av')
                     for name in av_test.gt_paths],
        'mask_paths': [replace_component(name, 'mask_resized', 'mask')
                       for name in av_test.mask_paths]})
    av_train, av_val = df_test[:24], df_test[24:]
    av_train_df = pd.DataFrame({
        'im_paths': [replace_component(name, 'images', 'images_resized')
                     for name in av_train.im_paths],
        'gt_paths': [replace_with_av_annotation(name, 'manual',
                                                'manual_av_resized')
                     for name in av_train.gt_paths],
        'mask_paths': [replace_component(name, 'mask', 'mask_resized')
                       for name in av_train.mask_paths]})
    av_val_df = pd.DataFrame({
        'im_paths': [replace_component(name, 'images', 'images_resized')
                     for name in av_val.im_paths],
        'gt_paths': [replace_with_av_annotation(name, 'manual',
                                                'manual_av_resized')
                     for name in av_val.gt_paths],
        'mask_paths': [replace_component(name, 'mask', 'mask_resized')
                       for name in av_val.mask_paths]})
    av_train_df.to_csv(osp.join(dataset_path, 'train_av.csv'), index=False)
    av_val_df.to_csv(osp.join(dataset_path, 'val_av.csv'), index=False)
    av_test_df.to_csv(osp.join(dataset_path, 'test_av.csv'))
    print('HRF prepared')


def prepare_stare():
    dataset = 'STARE'
    if dataset_is_complete(dataset, ['images', 'mask', 'manual'], ['test_all.csv']):
        print('STARE already prepared; skipping download and processing.')
        return
    dataset_path = osp.join(DATA_DIR, dataset)
    if not all(has_files(osp.join(dataset_path, directory))
               for directory in ['images', 'manual']):
        ensure_deepdyn_dataset(dataset, ['labels-ah', 'stare-images'])
    remove_if_present(osp.join(dataset_path, 'splits'))
    remove_if_present(osp.join(dataset_path, 'labels-vk'))
    if not has_files(osp.join(dataset_path, 'manual')):
        move_replace(osp.join(dataset_path, 'labels-ah'),
                     osp.join(dataset_path, 'manual'))
    if not has_files(osp.join(dataset_path, 'images')):
        move_replace(osp.join(dataset_path, 'stare-images'),
                     osp.join(dataset_path, 'images'))
    ensure_full_image_masks(
        osp.join(dataset_path, 'images'), osp.join(dataset_path, 'mask'))
    path_ims = osp.join(dataset_path, 'images')
    path_masks = osp.join(dataset_path, 'mask')
    path_gts = osp.join(dataset_path, 'manual')
    all_im_names = [osp.join(path_ims, name) for name in sorted(os.listdir(path_ims))]
    all_mask_names = [osp.join(path_masks, name) for name in sorted(os.listdir(path_masks))]
    all_gt_names = [osp.join(path_gts, name) for name in sorted(os.listdir(path_gts))]
    if not (len(all_im_names) == len(all_mask_names) == len(all_gt_names)):
        raise ValueError('STARE image, mask, and ground-truth counts differ')
    pd.DataFrame({'im_paths': all_im_names, 'gt_paths': all_gt_names,
                  'mask_paths': all_mask_names}).to_csv(
                      osp.join(dataset_path, 'test_all.csv'), index=False)
    print('STARE prepared')


def prepare_av_wide():
    dataset = 'AV-WIDE'
    if dataset_is_complete(dataset, ['images', 'masks', 'manual'], ['test_all.csv']):
        print('AV-WIDE already prepared; skipping download and processing.')
        return
    ensure_deepdyn_dataset(dataset, ['images', 'manual'])
    dataset_path = osp.join(DATA_DIR, dataset)
    remove_if_present(osp.join(dataset_path, 'splits'))
    path_ims = osp.join(dataset_path, 'images')
    path_masks = osp.join(dataset_path, 'masks')
    path_gts = osp.join(dataset_path, 'manual')
    ensure_full_image_masks(path_ims, path_masks)
    im_names = sorted(os.listdir(path_ims))
    gt_names = sorted(os.listdir(path_gts))
    mask_names = [osp.join(path_masks, name) for name in im_names]
    if len(im_names) != len(gt_names):
        raise ValueError('AV-WIDE image and ground-truth counts differ')
    pd.DataFrame({'im_paths': [osp.join(path_ims, name) for name in im_names],
                  'gt_paths': [osp.join(path_gts, name) for name in gt_names],
                  'mask_paths': mask_names}).to_csv(
                      osp.join(dataset_path, 'test_all.csv'), index=False)
    print('AV-WIDE prepared')


def prepare_dr_hagis():
    dataset = 'DR-HAGIS'
    if dataset_is_complete(dataset, ['images', 'mask', 'manual'], ['test_all.csv']):
        print('DR-HAGIS already prepared; skipping download and processing.')
        return
    dataset_path = osp.join(DATA_DIR, dataset)
    if not all(has_files(osp.join(dataset_path, directory))
               for directory in ['images', 'mask', 'manual']):
        archive = cached_download_any(
            ['https://zenodo.org/api/records/10847157/files/DRHAGIS.zip/content'],
            'DRHAGIS.zip', 'DR-HAGIS dataset')
        with tempfile.TemporaryDirectory() as staging:
            extract_zip(archive, staging)
            source = find_named_path(staging, 'DRHAGIS')
            if not source:
                raise FileNotFoundError('DRHAGIS directory not found in archive')
            required = ['Fundus_Images', 'Mask_images', 'Manual_Segmentations']
            if not all(osp.isdir(osp.join(source, name)) for name in required):
                raise FileNotFoundError('DRHAGIS archive has an unexpected layout')
            staging_output = osp.join(DATA_DIR, '.DR-HAGIS.tmp')
            remove_if_present(staging_output)
            os.makedirs(staging_output, exist_ok=True)
            move_replace(osp.join(source, 'Fundus_Images'),
                         osp.join(staging_output, 'images'))
            move_replace(osp.join(source, 'Mask_images'),
                         osp.join(staging_output, 'mask'))
            move_replace(osp.join(source, 'Manual_Segmentations'),
                         osp.join(staging_output, 'manual'))
            for directory in ['images', 'mask', 'manual']:
                remove_if_present(osp.join(staging_output, directory, '.DS_Store'))
            remove_if_present(dataset_path)
            move_replace(staging_output, dataset_path)

    path_ims = osp.join(DATA_DIR, dataset, 'images')
    path_masks = osp.join(DATA_DIR, dataset, 'mask')
    path_gts = osp.join(DATA_DIR, dataset, 'manual')
    all_im_names = sorted(os.listdir(path_ims), key=lambda name: name.split('_')[0])
    all_mask_names = sorted(os.listdir(path_masks), key=lambda name: name.split('_')[0])
    all_gt_names = sorted(os.listdir(path_gts), key=lambda name: name.split('_')[0])
    if not (len(all_im_names) == len(all_mask_names) == len(all_gt_names)):
        raise ValueError('DR-HAGIS image, mask, and ground-truth counts differ')
    pd.DataFrame({
        'im_paths': [osp.join(path_ims, name) for name in all_im_names],
        'gt_paths': [osp.join(path_gts, name) for name in all_gt_names],
        'mask_paths': [osp.join(path_masks, name) for name in all_mask_names],
    }).to_csv(osp.join(DATA_DIR, dataset, 'test_all.csv'), index=False)
    print('DR-HAGIS prepared')


def prepare_les_av():
    dataset = 'LES-AV'
    if dataset_is_complete(dataset, ['images', 'mask', 'manual', 'manual_av'],
                            ['test_all.csv', 'test_all_av.csv']):
        print('LES-AV already prepared; skipping processing.')
        return
    archive_path = 'LES-AV.zip'
    dataset_path = osp.join(DATA_DIR, dataset)
    has_raw_data = all(has_files(osp.join(dataset_path, directory))
                       for directory in ['images', 'mask', 'manual', 'manual_av'])
    if not has_raw_data and not osp.isfile(archive_path):
        print('NOTE: LES-AV was not prepared because LES-AV.zip was not found.')
        print('Download it from https://figshare.com/articles/dataset/LES-AV_dataset/11857698')
        print('Place LES-AV.zip beside this script and run it again to prepare LES-AV.')
        return

    if not has_raw_data:
        print('Preparing LES-AV')
        with tempfile.TemporaryDirectory() as staging:
            extract_zip(archive_path, staging)
            source = find_named_path(staging, 'LES-AV')
            if not source:
                raise FileNotFoundError('LES-AV directory not found in archive')
            required = ['images', 'masks', 'vessel-segmentations', 'arteries-and-veins']
            if not all(osp.isdir(osp.join(source, name)) for name in required):
                raise FileNotFoundError('LES-AV archive has an unexpected layout')
            staging_output = osp.join(DATA_DIR, '.LES-AV.tmp')
            remove_if_present(staging_output)
            os.makedirs(staging_output, exist_ok=True)
            move_replace(osp.join(source, 'images'), osp.join(staging_output, 'images'))
            move_replace(osp.join(source, 'masks'), osp.join(staging_output, 'mask'))
            move_replace(osp.join(source, 'vessel-segmentations'),
                         osp.join(staging_output, 'manual'))
            move_replace(osp.join(source, 'arteries-and-veins'),
                         osp.join(staging_output, 'manual_av'))
            remove_if_present(dataset_path)
            move_replace(staging_output, dataset_path)

    path_ims = osp.join(dataset_path, 'images')
    path_masks = osp.join(dataset_path, 'mask')
    path_gts = osp.join(dataset_path, 'manual')
    all_im_names = sorted(os.listdir(path_ims))
    all_mask_names = sorted(os.listdir(path_masks))
    all_gt_names = sorted(os.listdir(path_gts))
    if not (len(all_im_names) == len(all_mask_names) == len(all_gt_names)):
        raise ValueError('LES-AV image, mask, and ground-truth counts differ')
    df_all = pd.DataFrame({
        'im_paths': [osp.join(path_ims, name) for name in all_im_names],
        'gt_paths': [osp.join(path_gts, name) for name in all_gt_names],
        'mask_paths': [osp.join(path_masks, name) for name in all_mask_names],
    })
    df_all.to_csv(osp.join(dataset_path, 'test_all.csv'), index=False)
    df_all.gt_paths = [replace_component(name, 'manual', 'manual_av')
                       for name in df_all.gt_paths]
    df_all.to_csv(osp.join(dataset_path, 'test_all_av.csv'), index=None)
    print('LES-AV prepared')


def run_dataset(name, prepare):
    try:
        prepare()
        return True
    except Exception as error:
        print('{} failed: {}: {}'.format(name, type(error).__name__, error))
        traceback.print_exc()
        return False


def main():
    os.makedirs('experiments', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    print('Preparing public data')

    failed = []
    for name, prepare in [
            ('DRIVE', prepare_drive),
            ('CHASE-DB', prepare_chasedb),
            ('HRF', prepare_hrf),
            ('STARE', prepare_stare),
            ('AV-WIDE', prepare_av_wide),
            ('DR-HAGIS', prepare_dr_hagis),
            ('LES-AV', prepare_les_av)]:
        if not run_dataset(name, prepare):
            failed.append(name)

    if failed:
        raise SystemExit('Datasets failed: {}'.format(', '.join(failed)))
    print('All public data prepared, ready to go.')


if __name__ == '__main__':
    main()
