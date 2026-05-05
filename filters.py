import numpy as np
import cv2
from scipy.ndimage import convolve, median_filter


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def showimage(original_img, filtered_img, filter_name, imgsize=(512, 512)):
    """Display the original and filtered images side by side.

    Parameters
    ----------
    original_img : numpy.ndarray
        The original input image.
    filtered_img : numpy.ndarray
        The processed or filtered image. It will be converted to uint8 before display.
    filter_name : str
        Label shown in the window title.
    imgsize : tuple of int, optional
        The desired size (width, height) to which both images will be resized.
        Default is (512, 512).
    """
    filtered_img = filtered_img.astype(np.uint8)
    filtered_img = cv2.resize(filtered_img, imgsize, interpolation=cv2.INTER_AREA)
    original_img = cv2.resize(original_img, imgsize, interpolation=cv2.INTER_AREA)
    two_imgs = np.hstack((original_img, filtered_img))
    cv2.imshow(f'Apply {filter_name} filter', two_imgs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
#  Category 1 – Point / Intensity Transformations
# ─────────────────────────────────────────────

class PointFilters:
    """Pixel-wise (point) intensity transformation filters."""

    def __init__(self, img: np.ndarray):
        """
        Parameters
        ----------
        img : numpy.ndarray
            Grayscale uint8 image.
        """
        self.img = img

    def negative(self) -> np.ndarray:
        """1.1 Negative – invert pixel intensities."""
        neg_img = 255 - self.img
        showimage(self.img, neg_img, "Negative")
        return neg_img

    def threshold(self, threshold_val: int = None) -> np.ndarray:
        """1.2 Threshold – binarise the image around a mid-point.

        Parameters
        ----------
        threshold_val : int, optional
            Threshold value. Defaults to 255 // 2 + 1 = 128.
        """
        if threshold_val is None:
            threshold_val = 255 // 2 + 1
        threshold_img = np.where(self.img > threshold_val, 255, 0)
        showimage(self.img, threshold_img, "Threshold")
        return threshold_img

    def logarithmic(self) -> np.ndarray:
        """1.3 Logarithmic transformation."""
        normed_img = self.img.astype(np.float32) / 255
        log_img = np.log1p(1 + normed_img)
        log_img = (log_img / np.max(log_img)) * 255
        showimage(self.img, log_img, "Logarithmic")
        return log_img

    def power_law(self, gamma: float = 2.2) -> np.ndarray:
        """1.4 Power-law (gamma) transformation.

        Parameters
        ----------
        gamma : float, optional
            Gamma value. Default is 2.2.
        """
        normed_img = self.img.astype(np.float32) / 255
        power_img = np.power(normed_img, gamma)
        power_img = (power_img / np.max(power_img)) * 255
        showimage(self.img, power_img, "Power law transformation")
        return power_img

    def grey_level_slicing(self, a: int = 100, b: int = 150) -> np.ndarray:
        """1.5 Grey-level slicing – highlight a specific intensity band.

        Parameters
        ----------
        a : int
            Lower bound of the intensity range (inclusive). Default 100.
        b : int
            Upper bound of the intensity range (inclusive). Default 150.
        """
        graylvl_img = np.where((self.img >= a) & (self.img <= b), 255, 0)
        showimage(self.img, graylvl_img, "Grey Level Slicing")
        return graylvl_img

    def bit_plane_slicing(self, mask: int = 0b11111000) -> np.ndarray:
        """1.6 Bit-plane slicing – zero the LSBs.

        Parameters
        ----------
        mask : int
            Bitmask applied with AND. Default zeros the 3 LSBs (0b11111000).
        """
        bitpln_img = self.img & mask
        showimage(self.img, bitpln_img, "Bit Plane Slicing")
        return bitpln_img


# ─────────────────────────────────────────────
#  Category 2 – Spatial Domain Filters
# ─────────────────────────────────────────────

class SpatialFilters:
    """Spatial-domain convolution-based filters (smoothing & sharpening)."""

    def __init__(self, img: np.ndarray):
        """
        Parameters
        ----------
        img : numpy.ndarray
            Grayscale uint8 image.
        """
        self.img = img

    def average(self, n: int = 15) -> np.ndarray:
        """2.1 / 2.3 Smoothing (box / average) filter with an N×N kernel.

        Parameters
        ----------
        n : int
            Kernel size. Default is 15.
        """
        kernel_avg = np.ones((n, n)) / n ** 2
        box_img = convolve(self.img, kernel_avg, mode="reflect")
        showimage(self.img, box_img, "Average Filter")
        return box_img

    def weighted_smoothing(self) -> np.ndarray:
        """2.2 Weighted smoothing filter (tiled 3×3 Gaussian-like kernel)."""
        kernel_weg = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ]).astype(float)
        kernel_weg = np.tile(kernel_weg, 12)   # repeat kernel
        kernel_weg /= kernel_weg.sum()
        weg_img = convolve(self.img, kernel_weg, mode="reflect")
        showimage(self.img, weg_img, "Weighted Smoothing")
        return weg_img

    def median(self, size: int = 7) -> np.ndarray:
        """2.4 Median filter.

        Parameters
        ----------
        size : int
            Neighbourhood size. Default is 7.
        """
        median_img = median_filter(self.img, size=size, mode="reflect")
        showimage(self.img, median_img, "Median")
        return median_img

    def laplacian(self) -> np.ndarray:
        """2.5 Laplacian filter – 2nd derivative sharpening (4-connectivity kernel)."""
        w = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ])
        lap_img = convolve(self.img.astype(np.float32), w, mode="reflect")
        showimage(self.img, lap_img, "Laplacian")
        return lap_img

    def sobel(self) -> np.ndarray:
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
        Gx = convolve(self.img.astype(np.float32), Sx, mode="reflect")
        Gy = convolve(self.img.astype(np.float32), Sy, mode="reflect")
        sobel_img = np.sqrt(Gx ** 2 + Gy ** 2)
        sobel_img = (sobel_img / np.max(sobel_img)) * 255
        print(np.max(sobel_img))
        showimage(self.img, sobel_img, "Sobel")
        return sobel_img


# ─────────────────────────────────────────────
#  Category 3 – Frequency Domain Filters
# ─────────────────────────────────────────────

class FrequencyFilters:
    """Frequency-domain filters implemented via the 2-D DFT."""

    def __init__(self, img: np.ndarray):
        """
        Parameters
        ----------
        img : numpy.ndarray
            Grayscale uint8 image.
        """
        self.img = img

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

    def _dist_grid(self):
        """Return a distance-from-centre grid matching the image shape."""
        cnt_row = self.img.shape[0] // 2
        cnt_col = self.img.shape[1] // 2
        u = np.arange(stop=self.img.shape[0])
        v = np.arange(stop=self.img.shape[1])
        U, V = np.meshgrid(u, v, indexing='ij')
        dist2cnt = np.sqrt((U - cnt_row) ** 2 + (V - cnt_col) ** 2)
        return dist2cnt

    # ── 3.1 Ideal filters ─────────────────────────────────────────────────

    def ideal_low_pass(self, radius: int = 30) -> np.ndarray:
        """3.1.1 Ideal Low-Pass Filter (ILPF).

        Parameters
        ----------
        radius : int
            Cut-off radius in pixels. Default is 30.
        """
        complex_img = self._dft(self.img)
        dist2cnt = self._dist_grid()
        LPF = (dist2cnt <= radius).astype(np.float32)
        ideal_img = self._idft(complex_img * LPF)
        showimage(self.img, ideal_img, "Ideal Low Pass")
        return ideal_img

    def ideal_high_pass(self, radius: int = 135) -> np.ndarray:
        """3.1.2 Ideal High-Pass Filter (IHPF).

        Parameters
        ----------
        radius : int
            Cut-off radius in pixels. Default is 135.
        """
        complex_img = self._dft(self.img)
        dist2cnt = self._dist_grid()
        HPF = (dist2cnt > radius).astype(np.float32)
        ideal_img = self._idft(complex_img * HPF)
        showimage(self.img, ideal_img, "Ideal High Pass")
        return ideal_img

    # ── 3.2 Butterworth filters ────────────────────────────────────────────

    def butterworth_low_pass(self, radius: int = 13, order: int = 2) -> np.ndarray:
        """3.2.1 Butterworth Low-Pass Filter (BLPF).

        Parameters
        ----------
        radius : int
            Cut-off radius in pixels. Default is 13.
        order : int
            Filter order. Default is 2.
        """
        complex_img = self._dft(self.img)
        dist2cnt = self._dist_grid()
        BLPF = 1 / (1 + (dist2cnt / radius) ** (2 * order))
        BLPF_img = self._idft(complex_img * BLPF)
        showimage(self.img, BLPF_img, "Butterworth Low Pass")
        return BLPF_img

    def butterworth_high_pass(self, radius: int = 37, order: int = 5) -> np.ndarray:
        """3.2.2 Butterworth High-Pass Filter (BHPF).

        Parameters
        ----------
        radius : int
            Cut-off radius in pixels. Default is 37.
        order : int
            Filter order. Default is 5.
        """
        complex_img = self._dft(self.img)
        dist2cnt = self._dist_grid()
        # avoid division by zero
        dist2cnt = np.where(dist2cnt == 0, 1e-6, dist2cnt)
        BHPF = 1 / (1 + (radius / dist2cnt) ** (2 * order))
        BHPF_img = self._idft(complex_img * BHPF)
        showimage(self.img, BHPF_img, "Butterworth High Pass")
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

    print(f"""
      Shape: {np.shape(img)}
      dtype: {img.dtype}
      MAX val: {np.max(img)}
      MIN val: {np.min(img)}
      """)

    # ── Category 1: Point filters ──────────────────────────────────────────
    pf = PointFilters(img)
    pf.negative()
    pf.threshold()
    pf.logarithmic()
    pf.power_law(gamma=2.2)
    pf.grey_level_slicing(a=100, b=150)
    pf.bit_plane_slicing()

    # ── Category 2: Spatial filters ────────────────────────────────────────
    sf = SpatialFilters(img)
    sf.average(n=15)
    sf.weighted_smoothing()
    sf.median(size=7)
    sf.laplacian()
    sf.sobel()

    # ── Category 3: Frequency filters ─────────────────────────────────────
    ff = FrequencyFilters(img)
    ff.ideal_low_pass(radius=30)
    ff.ideal_high_pass(radius=135)
    ff.butterworth_low_pass(radius=13, order=2)
    ff.butterworth_high_pass(radius=37, order=5)