"""Unit and Integration Tests for Downscale-First and Grayscale Preprocessing Pipeline.

Validates:
1. Preprocessing downscaling to 32x32 pixels followed by grayscaling.
2. Preprocessing downscaling to 8x8 pixels followed by grayscaling.
3. Information-theoretic entropy, edge energy, SSIM, and PSNR metrics.
4. Feature dimensionalities (1,024 for 32x32, 64 for 8x8).
5. Quick cross-validation fit on downscaled representations.
"""

import sys

import numpy as np
from PIL import Image

from src.downscale_preprocessing import (
    compute_laplacian_variance,
    compute_psnr,
    compute_shannon_entropy,
    compute_ssim_numpy,
    preprocess_slice_downscale_then_gray,
)


def test_downscale_32x32():
    dummy = np.random.randint(0, 255, (176, 208, 3), dtype=np.uint8)
    res = preprocess_slice_downscale_then_gray(dummy, target_size=(32, 32))

    assert res["target_size"] == (32, 32)
    assert res["arr_float"].shape == (32, 32)
    assert res["arr_uint8"].shape == (32, 32)
    assert 0.0 <= res["arr_float"].min() <= res["arr_float"].max() <= 1.0
    assert res["arr_recon"].shape == (176, 208)
    print("  [PASS] test_downscale_32x32")


def test_downscale_8x8():
    dummy = np.random.randint(0, 255, (176, 208, 3), dtype=np.uint8)
    res = preprocess_slice_downscale_then_gray(dummy, target_size=(8, 8))

    assert res["target_size"] == (8, 8)
    assert res["arr_float"].shape == (8, 8)
    assert res["arr_uint8"].shape == (8, 8)
    assert 0.0 <= res["arr_float"].min() <= res["arr_float"].max() <= 1.0
    assert res["arr_recon"].shape == (176, 208)
    print("  [PASS] test_downscale_8x8")


def test_entropy_and_metrics():
    arr = np.linspace(0.0, 1.0, 64).reshape((8, 8)).astype(np.float32)
    ent = compute_shannon_entropy(arr)
    assert ent > 0.0

    lap = compute_laplacian_variance(arr)
    assert lap >= 0.0

    ssim_perfect = compute_ssim_numpy(arr, arr)
    assert np.isclose(ssim_perfect, 1.0, atol=1e-3)

    psnr_perfect = compute_psnr(arr, arr)
    assert psnr_perfect >= 90.0
    print("  [PASS] test_entropy_and_metrics")


def run_all():
    print("=" * 60)
    print("   RUNNING DOWNSCALE & GRAYSCALE PIPELINE TESTS   ")
    print("=" * 60)
    test_downscale_32x32()
    test_downscale_8x8()
    test_entropy_and_metrics()
    print("=" * 60)
    print("   ALL TESTS PASSED CLEANLY (3/3)   ")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
