import numpy as np
import cv2
from scipy.ndimage import convolve, median_filter


# ─────────────────────────────────────────────
#  Category 1 – Point / Intensity Transformations
# ─────────────────────────────────────────────

class PointFilters:
    """Pixel-wise (point) intensity transformation filters."""

    def __init__(self):
        pass

    def negative(self, img: np.ndarray) -> np.ndarray:
        """1.1 Negative – invert pixel intensities."""
        neg_img = 255 - img
        return neg_img

    def threshold(self, img: np.ndarray, threshold_val: int = (255 // 2 + 1)) -> np.ndarray:
        """1.2 Threshold – binarise the image around a mid-point.

        Parameters
        ----------
        threshold_val : int, optional
            Threshold value. Defaults to 255 // 2 + 1 = 128.
        """
        threshold_img = np.where(img > threshold_val, 255, 0)
        return threshold_img

    def logarithmic(self, img: np.ndarray) -> np.ndarray:
        """1.3 Logarithmic transformation."""
        normed_img = img.astype(np.float32) / 255
        log_img = np.log1p(1 + normed_img)
        log_img = (log_img / np.max(log_img)) * 255
        return log_img

    def power_law(self, img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """1.4 Power-law (gamma) transformation.

        Parameters
        ----------
        gamma : float, optional
            Gamma value. Default is 2.2.
        """
        normed_img = img.astype(np.float32) / 255
        power_img = np.power(normed_img, gamma)
        power_img = (power_img / np.max(power_img)) * 255
        return power_img

    def grey_level_slicing(self, img: np.ndarray, a: int = 100, b: int = 150) -> np.ndarray:
        """1.5 Grey-level slicing – highlight a specific intensity band.

        Parameters
        ----------
        a : int
            Lower bound of the intensity range (inclusive). Default 100.
        b : int
            Upper bound of the intensity range (inclusive). Default 150.
        """
        graylvl_img = np.where((img >= a) & (img <= b), 255, 0)
        return graylvl_img

    def bit_plane_slicing(self, img: np.ndarray, mask: int = 0b11111000) -> np.ndarray:
        """1.6 Bit-plane slicing – zero the LSBs.

        Parameters
        ----------
        mask : int
            Bitmask applied with AND. Default zeros the 3 LSBs (0b11111000).
        """
        bitpln_img = img & mask
        return bitpln_img


# ─────────────────────────────────────────────
#  Category 2 – Spatial Domain Filters
# ─────────────────────────────────────────────

class SpatialFilters:
    """Spatial-domain convolution-based filters (smoothing & sharpening)."""

    def __init__(self):
        pass

    def average(self, img: np.ndarray, n: int = 15) -> np.ndarray:
        """2.1 / 2.3 Smoothing (box / average) filter with an N×N kernel.

        Parameters
        ----------
        n : int
            Kernel size. Default is 15.
        """
        kernel_avg = np.ones((n, n)) / n ** 2
        box_img = convolve(img, kernel_avg, mode="reflect")
        return box_img

    def weighted_smoothing(self, img: np.ndarray) -> np.ndarray:
        """2.2 Weighted smoothing filter (tiled 3×3 Gaussian-like kernel)."""
        kernel_weg = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ]).astype(float)
        kernel_weg = np.tile(kernel_weg, 12)   # repeat the original kernel
        kernel_weg /= kernel_weg.sum()
        weg_img = convolve(img, kernel_weg, mode="reflect")
        return weg_img

    def median(self, img: np.ndarray, med_size: int = 7) -> np.ndarray:
        """2.4 Median filter.

        Parameters
        ----------
        size : int
            Neighbourhood size. Default is 7.
        """
        median_img = median_filter(img, size=med_size, mode="reflect")
        return median_img

    def laplacian(self, img: np.ndarray) -> np.ndarray:
        """2.5 Laplacian filter – 2nd derivative sharpening (4-connectivity kernel)."""
        w = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ])
        lap_img = convolve(img.astype(np.float32), w, mode="reflect")
        return lap_img

    def sobel(self, img: np.ndarray) -> np.ndarray:
        """2.6 Sobel filter – 1st derivative edge detection."""
        Sx = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ])
        Sy = np.array([
            [-1,  0,  1],
            [-2,  0,  2],
            [-1,  0,  1]
        ])
        Gx = convolve(img.astype(np.float32), Sx, mode="reflect")
        Gy = convolve(img.astype(np.float32), Sy, mode="reflect")
        sobel_img = np.sqrt(Gx ** 2 + Gy ** 2)
        sobel_img = (sobel_img / np.max(sobel_img)) * 255
        return sobel_img


# ─────────────────────────────────────────────
#  Category 3 – Frequency Domain Filters
# ─────────────────────────────────────────────

class FrequencyFilters:
    """Frequency-domain filters implemented via the 2-D DFT."""

    def __init__(self):
        pass

    # ── DFT helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _dft(temp_img: np.ndarray) -> np.ndarray:
        """Forward 2-D DFT with zero-frequency shifted to centre."""
        img2DFT = temp_img.astype(np.float32)
        img2DFT = np.fft.fft2(img2DFT)
        img2DFT = np.fft.fftshift(img2DFT)
        return img2DFT

    @staticmethod
    def _idft(temp_img: np.ndarray) -> np.ndarray:
        """Inverse 2-D DFT; returns a normalised uint8 image."""
        img2IDFT = np.fft.ifftshift(temp_img)
        img2IDFT = np.fft.ifft2(img2IDFT)
        img2IDFT = np.abs(img2IDFT)
        img2IDFT = (img2IDFT / np.max(img2IDFT)) * 255
        img2IDFT = img2IDFT.astype(np.uint8)
        return img2IDFT

    @staticmethod
    def _complex_to_normal(complex_img: np.ndarray) -> np.ndarray:
        """Convert a complex spectrum to a log-scaled displayable uint8 image."""
        normal_img = np.log1p(np.abs(complex_img))
        normal_img = (normal_img / np.max(normal_img)) * 255
        normal_img = normal_img.astype(np.uint8)
        return normal_img

    # ── Distance grid helper ───────────────────────────────────────────────

    @staticmethod
    def _dist_grid(img: np.ndarray):
        """Return a distance-from-centre grid matching the image shape."""
        cnt_row = img.shape[0] // 2
        cnt_col = img.shape[1] // 2
        u = np.arange(stop=img.shape[0])
        v = np.arange(stop=img.shape[1])
        U, V = np.meshgrid(u, v, indexing='ij')
        dist2cnt = np.sqrt((U - cnt_row) ** 2 + (V - cnt_col) ** 2)
        return dist2cnt

    # ── 3.1 Ideal filters ─────────────────────────────────────────────────

    def ideal_low_pass(self, img: np.ndarray, radius: int = 30) -> np.ndarray:
        """3.1.1 Ideal Low-Pass Filter (ILPF).

        Parameters
        ----------
        radius : int
            Cut-off radius in pixels. Default is 30.
        """
        complex_img = FrequencyFilters._dft(img)
        dist2cnt = FrequencyFilters._dist_grid(img)
        LPF = (dist2cnt <= radius).astype(np.float32)
        ideal_img = FrequencyFilters._idft(complex_img * LPF)
        return ideal_img

    def ideal_high_pass(self, img: np.ndarray, radius: int = 135) -> np.ndarray:
        """3.1.2 Ideal High-Pass Filter (IHPF).

        Parameters
        ----------
        radius : int
            Cut-off radius in pixels. Default is 135.
        """
        complex_img = FrequencyFilters._dft(img)
        dist2cnt = self._dist_grid(img)
        HPF = (dist2cnt > radius).astype(np.float32)
        ideal_img = FrequencyFilters._idft(complex_img * HPF)
        return ideal_img

    # ── 3.2 Butterworth filters ────────────────────────────────────────────

    def butterworth_low_pass(self, img: np.ndarray, radius: int = 13, order: int = 2) -> np.ndarray:
        """3.2.1 Butterworth Low-Pass Filter (BLPF).

        Parameters
        ----------
        radius : int
            Cut-off radius in pixels. Default is 13.
        order : int
            Filter order. Default is 2.
        """
        complex_img = FrequencyFilters._dft(img)
        dist2cnt = FrequencyFilters._dist_grid(img)
        BLPF = 1 / (1 + (dist2cnt / radius) ** (2 * order))
        BLPF_img = FrequencyFilters._idft(complex_img * BLPF)
        return BLPF_img

    def butterworth_high_pass(self, img: np.ndarray, radius: int = 37, order: int = 5) -> np.ndarray:
        """3.2.2 Butterworth High-Pass Filter (BHPF).

        Parameters
        ----------
        radius : int
            Cut-off radius in pixels. Default is 37.
        order : int
            Filter order. Default is 5.
        """
        complex_img = FrequencyFilters._dft(img)
        dist2cnt = FrequencyFilters._dist_grid(img)
        # avoid division by zero
        dist2cnt = np.where(dist2cnt == 0, 1e-6, dist2cnt)
        BHPF = 1 / (1 + (radius / dist2cnt) ** (2 * order))
        BHPF_img = FrequencyFilters._idft(complex_img * BHPF)
        return BHPF_img


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Read and pre-process image
    img = cv2.imread("./img.jpg")
    img = np.mean(img, axis=2).astype(np.uint8)   # convert to grayscale
    img = np.abs(img)
    img[img > 255] = 255

    # ── Category 1: Point filters ──────────────────────────────────────────
    pf = PointFilters()
    pf.negative(img)
    pf.threshold(img)
    pf.logarithmic(img)
    pf.power_law(img, gamma=2.2)
    pf.grey_level_slicing(img, a=100, b=150)
    pf.bit_plane_slicing(img)

    # ── Category 2: Spatial filters ────────────────────────────────────────
    sf = SpatialFilters()
    sf.average(img, n=15)
    sf.weighted_smoothing(img)
    sf.median(img, med_size=7)
    sf.laplacian(img)
    sf.sobel(img)

    # ── Category 3: Frequency filters ─────────────────────────────────────
    ff = FrequencyFilters()
    ff.ideal_low_pass(img, radius=30)
    ff.ideal_high_pass(img, radius=135)
    ff.butterworth_low_pass(img, radius=13, order=2)
    ff.butterworth_high_pass(img, radius=37, order=5)