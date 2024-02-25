import torch
from torch.nn import functional as f
from torchist import histogram


def otsu(img):
    # adapted from cv2
    hist = histogram(img, bins=256)
    hist_norm = hist.ravel() / hist.sum()
    Q = hist_norm.cumsum(dim=0)
    bins = torch.arange(256, device=img.device)
    fn_min = torch.inf
    thresh = -1
    for i in range(1, 256):
        p1, p2 = torch.hsplit(hist_norm, [i]) # probabilities
        q1, q2 = Q[i], Q[255] - Q[i] # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1, b2 = torch.hsplit(bins, [i]) # weights
        # finding means and variances
        m1, m2 = torch.sum(p1 * b1) / q1, torch.sum(p2 * b2) / q2
        v1, v2 = torch.sum(((b1 - m1) ** 2) * p1) / q1, torch.sum(((b2 - m2) ** 2) * p2) / q2
        # calculates the minimization function
        fn = v1 * q1 + v2 * q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    return thresh


def su(img: torch.Tensor, w=3, n_min=3) -> torch.Tensor:
    """Binarizes an image using the Su algorithm.
    
    Arguments:
        img: input image of shape (B, H, W) and float dtype
        w: window size, recommended to set w=n_min
        n_min: min high contrast pixels, recommended to set w=n_min

    Returns:
        image of shape (B, H, W)
    """
    eps = 1e-10
    batch_size, height, width = img.shape

    # construct contrast image
    windows = f.unfold(f.pad(img, pad=[w // 2] * 4, mode="replicate"), kernel_size=w)
    local_max = torch.max(windows, dim=0).values
    local_min = torch.min(windows, dim=0).values
    contrast = (local_max - local_min) / (local_max + local_min + eps)

    # find high-contrast pixels
    threshold = otsu(contrast) / 255
    hi_contrast = torch.where(contrast < threshold, torch.tensor(0, dtype=img.dtype), torch.tensor(1, dtype=img.dtype))
    del contrast
    hi_contrast_windows = f.unfold(f.pad(hi_contrast.view(height, width).unsqueeze(0), pad=[w // 2] * 4, mode="replicate"), kernel_size=w)

    # classify pixels
    hi_contrast_count = hi_contrast_windows.sum(axis=0)
    
    e_sum = torch.sum(windows * hi_contrast_windows, axis=0)  # matrix multiplication in axes 2 and 3
    e_mean = e_sum / hi_contrast_count  # produces nan if hi_contrast_count == 0, but since only pixels with hi_contrast_count >= n_min are considered, these values are ignored anyway
    e_mean = torch.where(torch.isnan(e_mean), 0, e_mean)
    e_std = torch.square((windows - e_mean) * hi_contrast_windows).mean(axis=0)
    del windows, hi_contrast_windows
    e_std = torch.sqrt(e_std)
    e_std = torch.where(torch.isnan(e_std), 0, e_std)
    result = torch.zeros_like(img)
    result[(hi_contrast_count.view(height, width) >= n_min) & (img <= e_mean.view(height, width) + e_std.view(height, width) / 2)] = 1

    return result
