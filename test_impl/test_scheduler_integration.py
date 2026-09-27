import os
import re
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = 'train_cyclical.py'


def _write_pair(path, size=64, fov_white=True):
    img = Image.new('L', (size, size), 0)
    gt = Image.new('L', (size, size), 0)
    mask = Image.new('L', (size, size), 255 if fov_white else 0)
    lo, hi = size // 3, size - size // 3
    for y in range(lo, hi):
        for x in range(lo, hi):
            img.putpixel((x, y), 255)
            gt.putpixel((x, y), 255)
    img.convert('RGB').save(path + '_img.png')
    gt.save(path + '_gt.png')
    mask.save(path + '_mask.png')


def _build_fixture(root, n_train=4, n_val=2, n_test=2):
    for i in range(n_train):
        _write_pair(os.path.join(root, 'train{}'.format(i)))
    for i in range(n_val):
        _write_pair(os.path.join(root, 'val{}'.format(i)))
    test_preds = os.path.join(root, 'test_preds')
    os.makedirs(test_preds, exist_ok=True)
    for i in range(n_test):
        base = os.path.join(root, 'test{}'.format(i))
        _write_pair(base)
        Image.open(base + '_gt.png').save(os.path.join(test_preds, 'test{}.png'.format(i)))

    def write_csv(path, rows):
        with open(path, 'w') as f:
            f.write('im_paths,gt_paths,mask_paths\n')
            for name in rows:
                im = os.path.join(root, name + '_img.png').replace('\\', '/')
                gt = os.path.join(root, name + '_gt.png').replace('\\', '/')
                mask = os.path.join(root, name + '_mask.png').replace('\\', '/')
                f.write('{},{},{}\n'.format(im, gt, mask))

    write_csv(os.path.join(root, 'train.csv'), ['train{}'.format(i) for i in range(n_train)])
    write_csv(os.path.join(root, 'val.csv'), ['val{}'.format(i) for i in range(n_val)])
    write_csv(os.path.join(root, 'test_all.csv'), ['test{}'.format(i) for i in range(n_test)])
    return test_preds


def _run(args, timeout=300):
    return subprocess.run(
        [sys.executable, TRAIN] + args, cwd=REPO_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, timeout=timeout)


def _val_metrics(output):
    values = {}
    for key in ('val_auc', 'val_dice', 'val_loss'):
        match = re.search(r'{}\s*:\s*([0-9.eE+-]+)'.format(key), output)
        values[key] = float(match.group(1)) if match else None
    return values


class SchedulerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='lwnet_sched_')
        cls.fixture = cls.tmp.name
        cls.test_preds = _build_fixture(cls.fixture)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _common(self):
        return [
            '--csv_train', os.path.join(self.fixture, 'train.csv'),
            '--cycle_lens', '1/1', '--model_name', 'wnet', '--batch_size', '2',
            '--im_size', '64', '--device', 'cpu', '--do_not_save', 'True', '--seed', '0',
        ]

    def test_default_and_explicit_cosine_match(self):
        default = _run(self._common())
        self.assertEqual(default.returncode, 0, default.stdout[-2000:])
        explicit = _run(self._common() + ['--scheduler', 'cosine'])
        self.assertEqual(explicit.returncode, 0, explicit.stdout[-2000:])
        # The default path and an explicit --scheduler cosine must be identical
        # (same seed, data, LR schedule, losses, selection).
        self.assertEqual(_val_metrics(default.stdout), _val_metrics(explicit.stdout))

    def test_damped_cosine_trains_and_reports_schedule(self):
        run = _run(self._common() + ['--scheduler', 'damped_cosine'])
        self.assertEqual(run.returncode, 0, run.stdout[-2000:])
        self.assertIn('Scheduler: damped_cosine', run.stdout)
        self.assertIn('damped_cosine: alpha=', run.stdout)
        self.assertIsNotNone(_val_metrics(run.stdout)['val_loss'])

    def test_damped_cosine_with_grad_accumulation(self):
        run = _run(self._common() + ['--scheduler', 'damped_cosine', '--grad_acc_steps', '1'])
        self.assertEqual(run.returncode, 0, run.stdout[-2000:])

    def test_query_mode_prints_and_creates_no_experiment_dir(self):
        before = set(os.listdir(os.path.join(REPO_ROOT, 'experiments')))
        query = _run([
            '--csv_train', os.path.join(self.fixture, 'train.csv'),
            '--cycle_lens', '1/1', '--batch_size', '2', '--im_size', '64',
            '--device', 'cpu', '--scheduler', 'damped_cosine', '--dc_show_max_lr',
        ])
        self.assertEqual(query.returncode, 0, query.stdout[-2000:])
        self.assertIn('highest lr ever hit', query.stdout)
        self.assertIn('optimizer updates', query.stdout)
        after = set(os.listdir(os.path.join(REPO_ROOT, 'experiments')))
        self.assertEqual(before, after)

    def test_query_mode_rejects_cosine(self):
        query = _run(self._common() + ['--dc_show_max_lr'])
        self.assertNotEqual(query.returncode, 0)
        self.assertIn('--scheduler damped_cosine', query.stdout)

    def test_query_uses_pseudo_label_extended_loader(self):
        base = _run([
            '--csv_train', os.path.join(self.fixture, 'train.csv'),
            '--cycle_lens', '1/1', '--batch_size', '2', '--im_size', '64',
            '--device', 'cpu', '--scheduler', 'damped_cosine', '--dc_show_max_lr',
        ])
        extended = _run([
            '--csv_train', os.path.join(self.fixture, 'train.csv'),
            '--cycle_lens', '1/1', '--batch_size', '2', '--im_size', '64',
            '--device', 'cpu', '--scheduler', 'damped_cosine', '--dc_show_max_lr',
            '--csv_test', os.path.join(self.fixture, 'test_all.csv'),
            '--path_test_preds', self.test_preds,
        ])
        self.assertEqual(base.returncode, 0, base.stdout[-2000:])
        self.assertEqual(extended.returncode, 0, extended.stdout[-2000:])
        base_updates = int(re.search(r'\((\d+) optimizer updates', base.stdout).group(1))
        ext_updates = int(re.search(r'\((\d+) optimizer updates', extended.stdout).group(1))
        self.assertGreater(ext_updates, base_updates)
        self.assertIn('pseudo-label-extended', extended.stdout)


if __name__ == '__main__':
    unittest.main()
