# statfusion: training-free binarization of degraded documents

Reference implementation of

> M. Ćurković, A. Ćurković. *An interpretable, training-free statistical fusion system for
> degraded document image binarization.* (submitted)

The method needs no training data and no GPU. Every pixel decision follows from three
quantities measured on the page itself (an ink intensity, a contour intensity and a window
size) and from the local median and mean around the pixel.

## Usage

```
pip install -r requirements.txt
python statfusion.py page.jpg page_bin.png          # one image
python statfusion.py input_folder output_folder     # all images in a folder
```

Example with the camera pages of this repository:

```
python statfusion.py data/camera/page1.bmp output/page1_bin.png
python statfusion.py data/camera output
```

The output folder is created if it does not exist, and every result is saved as PNG with the name
of the input image. Input images may be PNG, BMP, JPG or TIF, in colour or grayscale (colour images
are converted to grayscale first). In the result, text is black (0) and background white (255).

The script prints, for every image, the run time and the quantities that drive the decision:
the ink intensity `T_I`, the contour intensity `B_I`, the window radius `K` and the scale factor `f`:

```
page1.bmp: 8.54 s  T_I=63 B_I=114 K=13 f=0.655 -> output/page1_bin.png
```

The first run takes a few seconds longer because Numba compiles the code; the compiled code is
cached for later runs.

From Python:

```python
from statfusion import binarize, load_gray
binary, info = binarize(load_gray("page.jpg"))     # binary: text = 0, background = 255
```

## Parameters

The method has four constants, set at the top of `statfusion.py`. They were fixed before the
evaluation and were not tuned.

| Constant | Value | Meaning |
|---|---|---|
| `SIGMA` | 1.6 | Gaussian smoothing (pixels) before the gradient |
| `GAMMA` | 0.5 | selectivity of the ink samples next to strong edges |
| `C` | 2 | window factor (relative to the size of the edge components) |
| `RHO` | 0.5 | upper limit of the window radius (fraction of the image size) |

## Data

| Folder | Content |
|---|---|
| `data/camera` | six physically degraded pages photographed with a smartphone (Samsung Galaxy S25) |
| `data/camera_txt` | exact transcriptions of these pages, used for the OCR evaluation |
| `data/dibco_txt` | transcriptions of the DIBCO images used for the OCR evaluation |
| `data/dibco_images.csv` | the 122 DIBCO and H-DIBCO images used in the paper (collection, design or held-out group, and whether a transcription was used for OCR) |

The DIBCO and H-DIBCO images and ground truth are not redistributed here; they are available from
their organizers. The paper uses 122 images of ten collections, listed in `data/dibco_images.csv`.

## Test

```
python tests/test_statfusion.py
```

## License

MIT (see `LICENSE`). If you use this code or the data, please cite the paper.
