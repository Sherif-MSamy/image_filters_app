# Digital Image Processing — Filter Equations & Mathematical Application Guide

> Based on lecture slides from *Digital Image Processing* (Gonzalez & Woods, 2002)

---

## General Spatial Domain Framework

Before diving into individual filters, every spatial domain operation fits into the general form:

$$g(x, y) = T[f(x, y)]$$

where $f(x,y)$ is the input image, $g(x,y)$ is the processed output image, and $T$ is an operator applied over some neighbourhood of pixel $(x, y)$.

For **point processing**, the neighbourhood is just the pixel itself, reducing to:

$$s = T(r)$$

where $r$ is the original pixel value and $s$ is the transformed output value.

---

## Part 1 — Point Processing

### 1. Negative Image

**Equation:**

$$s = L - 1 - r$$

or equivalently, when intensities are normalised to $[0.0,\ 1.0]$:

$$s = 1.0 - r$$

**How to apply on an image:**

For every pixel at position $(x, y)$ in the image $f$:

$$g(x, y) = (L - 1) - f(x, y)$$

where $L = 256$ for an 8-bit image (so $L - 1 = 255$).

**Example (8-bit):** A pixel with intensity $r = 200$ becomes $s = 255 - 200 = 55$ (turns bright pixels dark and vice versa).

**Purpose:** Useful for enhancing white or grey detail embedded in dark regions, such as medical images (e.g. mammograms).

---

### 2. Thresholding

**Equation:**

$$s = \begin{cases} 1.0 & \text{if } r > T \\ 0.0 & \text{if } r \leq T \end{cases}$$

where $T$ is a chosen threshold value (in $[0, 1]$ for normalised intensities, or $[0, 255]$ for 8-bit).

**How to apply on an image:**

For every pixel at $(x, y)$:

$$g(x, y) = \begin{cases} 255 & \text{if } f(x, y) > T \\ 0 & \text{if } f(x, y) \leq T \end{cases}$$

**Steps:**
1. Choose a threshold $T$ (e.g. $T = 128$ for mid-range in an 8-bit image).
2. Scan every pixel; if its value exceeds $T$, set it to 255 (white); otherwise set it to 0 (black).
3. The result is a binary image.

**Purpose:** Object segmentation — isolating a foreground object from a background by binarising the image.

---

### 3. Logarithmic Transformation

**Equation:**

$$s = c \cdot \log(1 + r)$$

where $c$ is a scaling constant (usually set to 1) and $r \in [0.0,\ 1.0]$.

**How to apply on an image:**

For every pixel at $(x, y)$:

$$g(x, y) = c \cdot \log\!\bigl(1 + f(x, y)\bigr)$$

**Steps:**
1. Normalise the image to $[0, 1]$ if not already.
2. Apply $\log(1 + r)$ to every pixel value. The $+1$ prevents $\log(0) = -\infty$.
3. If using integer pixel values (0–255), scale $r$ before and rescale after: $s = c \cdot \log(1 + r/255) \times 255$.

**Effect on pixel values:**
- **Dark pixels** (small $r$): $\log$ rises steeply → they are spread into a wider output range (enhanced).
- **Bright pixels** (large $r$): $\log$ grows slowly → they are compressed together.

**Purpose:** Revealing detail in very dark areas; commonly used to display the Fourier spectrum (which has a very large dynamic range).

**Inverse log** performs the opposite — compresses low values and expands high ones:

$$s = c \cdot (e^r - 1)$$

---

### 4. Power Law (Gamma) Transformation

**Equation:**

$$s = c \cdot r^{\gamma}$$

where $c$ is a scaling constant (usually 1) and $\gamma$ (gamma) controls the shape of the curve. $r \in [0.0,\ 1.0]$.

**How to apply on an image:**

For every pixel at $(x, y)$:

$$g(x, y) = c \cdot f(x, y)^{\gamma}$$

**Effect of $\gamma$:**

| $\gamma < 1$ | Brightens the image — expands dark intensities, compresses bright ones |
|---|---|
| $\gamma = 1$ | Identity — no change |
| $\gamma > 1$ | Darkens the image — compresses dark intensities, expands bright ones |

**Steps:**
1. Normalise pixel values to $[0.0, 1.0]$.
2. Raise each pixel to the power $\gamma$.
3. Rescale back to $[0, 255]$ if needed.

**Purpose:** Gamma correction for display devices (monitors don't respond linearly), contrast enhancement in medical imaging (MRI, aerial photography).

---

### 5. Grey Level Slicing

**Concept:** Highlights a specific range (band) of grey levels $[a, b]$, leaving the rest either suppressed or unchanged.

**Two variants:**

**Variant A — Binary slicing (suppress background):**

$$s = \begin{cases} L_{high} & \text{if } a \leq r \leq b \\ 0 & \text{otherwise} \end{cases}$$

**Variant B — Preserve background:**

$$s = \begin{cases} L_{high} & \text{if } a \leq r \leq b \\ r & \text{otherwise} \end{cases}$$

where $L_{high}$ is a bright highlight value (e.g. 255), and $[a, b]$ is the grey level range of interest.

**How to apply on an image:**

For every pixel at $(x, y)$:
1. Check whether $f(x, y)$ falls within the target range $[a, b]$.
2. Apply the desired output (highlight or suppress) based on the chosen variant.

**Purpose:** Highlighting features of interest — e.g. isolating specific tissue densities in a medical scan, or highlighting flaws in industrial inspection.

---

### 6. Bit Plane Slicing

**Concept:** For an 8-bit pixel value $r$, each pixel can be represented as 8 binary bits:

$$r = b_7 \cdot 2^7 + b_6 \cdot 2^6 + b_5 \cdot 2^5 + \cdots + b_1 \cdot 2^1 + b_0 \cdot 2^0$$

**Extracting bit plane $k$:**

$$B_k(x, y) = \left\lfloor \frac{f(x, y)}{2^k} \right\rfloor \mod 2$$

which is equivalently done via a bitwise AND mask:

$$B_k(x, y) = f(x, y) \ \text{AND} \ 2^k$$

Multiply the result by 255 to display as a binary image.

**Bit plane significance:**

| Plane | Bits | Content |
|---|---|---|
| $b_7$ (MSB) | $2^7 = 128$ | Major structural/visual information |
| $b_6, b_5$ | $2^6, 2^5$ | General shapes and gradients |
| $b_4, b_3$ | $2^4, 2^3$ | Finer details and textures |
| $b_2, b_1, b_0$ (LSB) | $2^2, 2^1, 2^0$ | Noise, subtle variations |

**Image reconstruction from selected planes:**

$$f_{reconstructed}(x,y) = \sum_{k \in S} B_k(x,y) \cdot 2^k$$

where $S$ is the set of selected bit planes (e.g. $S = \{7, 6\}$).

**How to apply:**
1. For each pixel, extract the binary bit at position $k$ using the mask $2^k$.
2. Display the resulting binary image (0 or 255).
3. To reconstruct, sum the desired planes weighted by their power of 2.

**Purpose:** Image compression (discard low-order planes), steganography (hide data in LSBs), understanding image structure and information content.

---

## Part 2 — Spatial Filters

Spatial filtering applies a **kernel** (mask) to every pixel via convolution or correlation. The general equation is:

$$g(x, y) = \sum_{s=-a}^{a} \sum_{t=-b}^{b} w(s, t) \cdot f(x+s,\ y+t)$$

where $w(s,t)$ is the filter kernel of size $(2a+1) \times (2b+1)$.

---

### 1. Smoothing Filter (Simple Averaging / Box Filter)

**Kernel (3×3):**

$$w = \frac{1}{9} \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}$$

**Equation:**

$$g(x, y) = \frac{1}{mn} \sum_{s=-a}^{a} \sum_{t=-b}^{b} f(x+s,\ y+t)$$

where $mn$ is the total number of pixels in the neighbourhood ($m \times n$).

**How to apply on an image:**
1. Place the kernel over each pixel in the image.
2. Multiply each kernel weight by the corresponding pixel value.
3. Sum all products: $g = \frac{1}{9}(f_{11} + f_{12} + \cdots + f_{33})$.
4. Assign the result to the output pixel.
5. Repeat for every pixel (handle edges by zero-padding, replication, or cropping).

**Purpose:** Noise removal, blurring fine detail, highlighting gross structure.

---

### 2. Weighted Smoothing Filter (Weighted Average)

**Kernel (3×3):**

$$w = \frac{1}{16} \begin{bmatrix} 1 & 2 & 1 \\ 2 & 4 & 2 \\ 1 & 2 & 1 \end{bmatrix}$$

**Equation:**

$$g(x, y) = \frac{1}{W} \sum_{s=-a}^{a} \sum_{t=-b}^{b} w(s, t) \cdot f(x+s,\ y+t)$$

where $W = \sum_{s,t} w(s,t)$ (the sum of all kernel weights = 16 in this case).

**How to apply on an image:**
1. Same convolution procedure as the simple averaging filter.
2. Multiply each pixel in the neighbourhood by its corresponding weight (the central pixel gets the highest weight of 4, corners get the lowest weight of 1).
3. Sum the products and divide by 16.

**Key difference from simple average:** Pixels closer to the centre contribute more, producing smoother blurring with less ringing artefacts.

**Purpose:** More effective noise suppression than a simple average, while preserving slightly more structure.

---

### 3. Average Filter (General N×N)

This is a generalisation of the smoothing filter for any kernel size $N \times N$:

$$g(x, y) = \frac{1}{N^2} \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} f(x - \lfloor N/2 \rfloor + i,\ y - \lfloor N/2 \rfloor + j)$$

Common kernel sizes: $3\times3$, $5\times5$, $9\times9$, $15\times15$, $35\times35$.

**How to apply on an image:**
1. Select kernel size $N$ (larger $N$ → stronger smoothing).
2. For each pixel, average all $N^2$ values in its neighbourhood.
3. Larger filters progressively remove finer detail until only gross features remain.

**Purpose:** Progressive blur and noise removal; larger kernels produce stronger smoothing effects.

---

### 4. Median Filter

**Not a linear convolution** — instead, a rank-order operation.

**Algorithm:**
1. Collect all pixel values in the neighbourhood window (e.g. $3 \times 3$ = 9 pixels).
2. Sort the values in ascending order.
3. Assign the **median** (middle value) to the output pixel:

$$g(x, y) = \text{median}\{f(x+s, y+t) \mid (s,t) \in \text{neighbourhood}\}$$

For a $3 \times 3$ window: sort 9 values, pick the 5th value.

**How to apply on an image:**
1. Slide the window over every pixel.
2. Gather the $N^2$ pixel values in the window.
3. Sort them: e.g. $[1, 7, 15, 18, 24, 25, 30, 31, 45]$ → median = 24.
4. Place the median value in the output image.

**Purpose:** Extremely effective at removing **salt-and-pepper (impulse) noise** while preserving edges — outperforms averaging filters for this noise type.

---

### 5. Laplacian Filter (2nd Derivative Sharpening)

**Mathematical definition:**

The Laplacian is the sum of second-order partial derivatives:

$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$$

**Discrete approximations:**

$$\frac{\partial^2 f}{\partial x^2} = f(x+1, y) + f(x-1, y) - 2f(x, y)$$

$$\frac{\partial^2 f}{\partial y^2} = f(x, y+1) + f(x, y-1) - 2f(x, y)$$

**Combined discrete Laplacian:**

$$\nabla^2 f = f(x+1,y) + f(x-1,y) + f(x,y+1) + f(x,y-1) - 4f(x,y)$$

**Standard kernel (4-connectivity):**

$$w = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

**8-connectivity variant:**

$$w = \begin{bmatrix} 1 & 1 & 1 \\ 1 & -8 & 1 \\ 1 & 1 & 1 \end{bmatrix}$$

**Image enhancement using the Laplacian:**

Subtract the Laplacian from the original image to sharpen:

$$g(x, y) = f(x, y) - \nabla^2 f$$

This can be combined into a single sharpening kernel:

$$g(x, y) = 5f(x,y) - f(x+1,y) - f(x-1,y) - f(x,y+1) - f(x,y-1)$$

Corresponding **all-in-one sharpening kernel:**

$$w = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix}$$

**How to apply on an image:**
1. Convolve $f(x,y)$ with the Laplacian kernel $w$ to get $\nabla^2 f$.
2. Subtract: $g = f - \nabla^2 f$ (or use the combined kernel directly).
3. The result enhances edges and fine detail.

**Purpose:** Sharpening images, edge detection, highlighting discontinuities. The Laplacian is **isotropic** (rotation-invariant — responds equally to edges in all directions).

---

### 6. Sobel Filter (1st Derivative Edge Detection)

**Mathematical background:**

The gradient magnitude of $f(x,y)$ is:

$$|\nabla f| = \sqrt{G_x^2 + G_y^2}$$

Approximated in practice as:

$$|\nabla f| \approx |G_x| + |G_y|$$

where $G_x$ and $G_y$ are the gradient components in $x$ and $y$.

**Sobel Operators:**

Horizontal gradient kernel $G_x$ (detects vertical edges):

$$S_x = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

Vertical gradient kernel $G_y$ (detects horizontal edges):

$$S_y = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}$$

These are derived from the 1st derivative formulas based on the 3×3 neighbourhood:

$$G_x = (z_7 + 2z_8 + z_9) - (z_1 + 2z_2 + z_3)$$

$$G_y = (z_3 + 2z_6 + z_9) - (z_1 + 2z_4 + z_7)$$

where $z_1 \ldots z_9$ are the 3×3 pixel neighbourhood values.

**How to apply on an image:**
1. Convolve $f(x,y)$ with $S_x$ → get $G_x$ image.
2. Convolve $f(x,y)$ with $S_y$ → get $G_y$ image.
3. Combine: $g(x,y) = |G_x| + |G_y|$ (or $\sqrt{G_x^2 + G_y^2}$ for exact magnitude).
4. Optionally threshold the result to produce a binary edge map.

**Purpose:** Edge detection — highlights boundaries between regions. Responds strongly to edges, less strongly to uniform regions or smooth gradients.

---

## Part 3 — Frequency Domain Filters

### Theoretical Foundation

**Discrete Fourier Transform (DFT):**

$$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x,y) \cdot e^{-j2\pi\left(\frac{ux}{M} + \frac{vy}{N}\right)}$$

for $u = 0, 1, \ldots, M-1$ and $v = 0, 1, \ldots, N-1$.

**Inverse DFT:**

$$f(x, y) = \frac{1}{MN} \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} F(u,v) \cdot e^{+j2\pi\left(\frac{ux}{M} + \frac{vy}{N}\right)}$$

**General frequency domain filtering pipeline:**

$$G(u,v) = H(u,v) \cdot F(u,v)$$

1. Compute $F(u,v)$ = DFT of the input image $f(x,y)$.
2. Multiply by filter $H(u,v)$ element-wise in the frequency domain.
3. Compute $g(x,y) = \text{IDFT}[G(u,v)]$ to recover the filtered image.

**Distance from centre (DC component):**

$$D(u, v) = \sqrt{\left(u - \frac{M}{2}\right)^2 + \left(v - \frac{N}{2}\right)^2}$$

The relationship between filter variants:

$$H_{hp}(u,v) = 1 - H_{lp}(u,v)$$

---

### 1. Ideal Filter

#### Ideal Low Pass Filter (ILPF)

**Transfer function:**

$$H(u,v) = \begin{cases} 1 & \text{if } D(u,v) \leq D_0 \\ 0 & \text{if } D(u,v) > D_0 \end{cases}$$

#### Ideal High Pass Filter (IHPF)

**Transfer function:**

$$H(u,v) = \begin{cases} 0 & \text{if } D(u,v) \leq D_0 \\ 1 & \text{if } D(u,v) > D_0 \end{cases}$$

where $D_0$ is the **cutoff frequency** (radius from the centre of the spectrum).

**How to apply on an image:**
1. Compute the 2D DFT: $F(u,v) = \mathcal{F}\{f(x,y)\}$.
2. Shift the spectrum so the DC (zero-frequency) component is at the centre.
3. For each frequency pair $(u,v)$, compute $D(u,v)$.
4. Apply the mask: multiply $F(u,v)$ by $H(u,v)$ — this zeros out frequencies beyond (LPF) or within (HPF) radius $D_0$.
5. Inverse shift, then compute the IDFT to get the filtered image.

**Characteristic:** The ideal filter creates a perfectly sharp circular cutoff. This sharp boundary in the frequency domain causes **ringing artefacts** (Gibbs phenomenon) in the spatial domain.

---

### 2. Butterworth Filter

#### Butterworth Low Pass Filter (BLPF)

**Transfer function (order $n$, cutoff $D_0$):**

$$H(u,v) = \frac{1}{1 + \left[\dfrac{D(u,v)}{D_0}\right]^{2n}}$$

#### Butterworth High Pass Filter (BHPF)

**Transfer function:**

$$H(u,v) = \frac{1}{1 + \left[\dfrac{D_0}{D(u,v)}\right]^{2n}}$$

**How to apply on an image:**
1. Compute the DFT $F(u,v)$ and centre the spectrum.
2. For each $(u,v)$, compute $D(u,v)$ and evaluate $H(u,v)$ using the formula above.
3. Multiply: $G(u,v) = H(u,v) \cdot F(u,v)$.
4. Take the IDFT of $G(u,v)$.

**Effect of order $n$:**
- $n = 1$: Very gentle rolloff — gradual transition at the cutoff.
- Higher $n$: Steeper rolloff — approaches ideal filter behaviour but may introduce slight ringing.
- $n = 2$ is commonly used in practice.

**At cutoff distance $D_0$:** $H(u,v) = 0.5$ (i.e. the filter passes 50% of the signal at the cutoff boundary).

**Advantage over ideal:** Smooth transition band eliminates ringing artefacts seen with the ideal filter.

---

### 3. Gaussian Filter

#### Gaussian Low Pass Filter (GLPF)

**Transfer function:**

$$H(u,v) = e^{-D^2(u,v) \,/\, 2D_0^2}$$

#### Gaussian High Pass Filter (GHPF)

**Transfer function:**

$$H(u,v) = 1 - e^{-D^2(u,v) \,/\, 2D_0^2}$$

where $D_0$ is the cutoff frequency (controls the spread of the Gaussian bell curve).

**How to apply on an image:**
1. Compute the DFT $F(u,v)$ and centre the spectrum.
2. For each frequency pair $(u,v)$, compute $D(u,v)$.
3. Compute $H(u,v)$ using the Gaussian formula.
4. Multiply: $G(u,v) = H(u,v) \cdot F(u,v)$.
5. Compute the IDFT to recover $g(x,y)$.

**Effect of $D_0$:**
- Small $D_0$: Aggressive smoothing (LPF) — most high frequencies removed.
- Large $D_0$: Mild smoothing — most frequencies pass through.

**Key property:** The Fourier transform of a Gaussian is itself a Gaussian. This means applying a Gaussian filter in the frequency domain is equivalent to convolution with a Gaussian kernel in the spatial domain — and importantly, **no ringing artefacts** are produced.

---

## Summary Comparison Table

| Filter | Domain | Type | Equation |
|---|---|---|---|
| Negative | Point | — | $s = L - 1 - r$ |
| Threshold | Point | — | $s = 1$ if $r > T$, else $0$ |
| Logarithmic | Point | — | $s = c \cdot \log(1 + r)$ |
| Power Law | Point | — | $s = c \cdot r^\gamma$ |
| Grey Level Slicing | Point | — | $s = L_{high}$ if $a \leq r \leq b$ |
| Bit Plane Slicing | Point | — | $B_k = \lfloor r / 2^k \rfloor \mod 2$ |
| Smoothing (Avg) | Spatial | LPF | $g = \frac{1}{mn}\sum w \cdot f$ |
| Weighted Avg | Spatial | LPF | $g = \frac{1}{16}\sum w(s,t) \cdot f$ |
| Average (N×N) | Spatial | LPF | $g = \frac{1}{N^2}\sum f$ |
| Median | Spatial | LPF | $g = \text{median}(f_{neighbourhood})$ |
| Laplacian | Spatial | HPF | $g = f - \nabla^2 f$ |
| Sobel | Spatial | HPF | $g = \|G_x\| + \|G_y\|$ |
| Ideal | Frequency | LP/HP | $H = 1$ or $0$ based on $D(u,v)$ vs $D_0$ |
| Butterworth | Frequency | LP/HP | $H = \frac{1}{1 + [D/D_0]^{2n}}$ |
| Gaussian | Frequency | LP/HP | $H = e^{-D^2/2D_0^2}$ |
