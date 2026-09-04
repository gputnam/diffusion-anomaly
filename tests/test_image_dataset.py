import numpy as np

from guided_diffusion.image_datasets import ImageDataset


def test_image_dataset_yields_npz_array_and_dict(tmp_npz):
    npz_path, n, h, w = tmp_npz

    ds = ImageDataset(
        resolution=h,
        image_paths=str(npz_path.parent),
        classes=None,
        charge_scale=1.0,
        require_charge=False,
        importance_sampling=False,
    )
    it = iter(ds)
    arr, out_dict = next(it)

    # ImageDataset.__iter__ yields a single (1, H, W) slice from the cached array
    assert arr.shape == (1, h, w)
    assert arr.dtype == np.float32
    assert "path" in out_dict
    assert "weight" in out_dict
    assert "pixel_weight" in out_dict
    assert out_dict["path"] == "sample"


def test_load_image_file_npz(tmp_npz):
    from guided_diffusion.image_datasets import load_image_file

    npz_path, n, h, w = tmp_npz
    images, truth = load_image_file(npz_path, h)
    assert images.shape == (n, 1, h, w) and truth.shape == (n, 1, h, w)
    assert images.dtype == np.float32
    assert images.min() >= -1.0 and images.max() <= 1.0


def test_load_image_file_sbnd_h5(tmp_path):
    import h5py
    from guided_diffusion.image_datasets import load_image_file

    # SBND raw format: per-event "raw" of shape (5638 wires, T ticks). With
    # T=1024 each plane (1984/1984/1670 wires) yields 3/3/3 rows x 2 cols of
    # 512x512 tiles -> 18 tiles per event.
    rng = np.random.default_rng(0)
    path = tmp_path / "sbnd.h5"
    with h5py.File(path, "w") as f:
        for ev in range(2):
            f.create_group(str(ev)).create_dataset(
                "raw", data=(50 * rng.standard_normal((5638, 1024))).astype(np.float32)
            )
    images, truth = load_image_file(path, 512)
    assert images.shape == (36, 1, 512, 512)
    assert truth.shape == images.shape and np.all(truth == 1)
    assert images.min() >= -1.0 and images.max() <= 1.0


def test_single_file_image_paths(tmp_npz):
    npz_path, n, h, w = tmp_npz
    ds = ImageDataset(resolution=h, image_paths=str(npz_path))
    assert ds.local_images == [str(npz_path)]
    arr, out_dict = next(iter(ds))
    assert arr.shape == (1, h, w)
    assert out_dict["path"] == "sample"
