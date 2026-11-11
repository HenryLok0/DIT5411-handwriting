# DIT5411 Traditional Chinese Handwriting Recognition

THEi DIT5411 Machine Learning assignment. The task is to recognize handwritten traditional Chinese characters with TensorFlow. The dataset has 13,065 characters, and the network output layer has 13,065 neurons.

## Dataset

The images come from the [Traditional Chinese Handwriting Dataset](https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset) by AI.FREE Team. This project uses the full archive in [handwritting_data_all](https://github.com/chenkenanalytic/handwritting_data_all).

| Item | Value |
| --- | --- |
| Characters | 13,065 |
| Images | 684,677 |
| Original size | 300 x 300 |
| Samples per character | about 50 on average |

The 4,803-character common-word subset is not used. Its output layer would be smaller than 13,065.

## Train and test split

Files for one character are sorted in natural order, so `字_2` comes before `字_10`.

- If a character has at least 41 images, the first 40 are training images and the rest are test images.
- If it has fewer than 41 images, about 80% are used for training and 20% for testing, with at least one image left for testing.

Test images are not rotated, sheared, or scaled.

## Augmentation

Each character has only about 40 real training photos. OpenCV `warpAffine` expands those photos to at least 200 training samples per character.

With 40 photos, each photo is stored as five fixed versions:

| Id | Name | Transform |
| --- | --- | --- |
| 0 | original | unchanged |
| 1 | rotate_p12 | 12 degrees counter-clockwise |
| 2 | rotate_m12 | 12 degrees clockwise |
| 3 | shear_x | horizontal shear of 0.25 |
| 4 | scale_080 | scale of 0.8 |

Transforms are applied about the image center, in this order: scale, shear, then rotation. Empty borders use the median color of the four corners. The parameters stay fixed across epochs.

If a character has fewer than 40 usable training photos, extra rotation, shear, and scale settings are added until the class reaches 200 samples.

Each photo is resized to 128 x 128 before the warp, then resized to the model input of 64 x 64. Use the same `--img-size` for preparation and training. The geometric transforms follow [Introduction to OpenCV](https://github.com/darshitajain/Introduction-to-OpenCV).

Preview the five versions of one character:

```powershell
python preview_augmentation.py --zip data/all_data.zip --out results/augmentation_preview.png
```

## Models

All three models take a grayscale 64 x 64 image. The loss is `sparse_categorical_crossentropy`. The optimizer is Adam with a learning rate of 0.001. The softmax width equals the number of characters, which is 13,065 on the full dataset.

Recognition rate is top-1 accuracy on the test set. Top-5 accuracy is also recorded. Every model is trained for the same number of epochs. The test set is scored once, after the last epoch, and is not used to choose an epoch.

If training loss does not fall for one epoch, the learning rate is halved, down to 0.00001.

Parameter counts below are from `count_params()` at 64 x 64 input and 13,065 classes.

### shallow_cnn

Two convolution stages. The first uses two 3 x 3 convolutions with 32 filters, then 2 x 2 max pooling and dropout 0.25. The second uses 64 filters with the same pattern. A flatten layer, a dense layer of 256 units, and dropout 0.4 lead to the softmax. Parameters: 7,617,257.

### deep_cnn

Four stages with 32, 64, 128, and 256 filters. Each stage has two Conv-BatchNorm-ReLU blocks and 2 x 2 max pooling. Global average pooling, a dense layer of 512 units, and dropout 0.5 lead to the softmax. Parameters: 8,008,489.

### residual_cnn

A 32-filter convolution, batch normalization, and ReLU, followed by residual blocks of 32/1, 64/2, 128/2, and 128/1 (filters/stride). A 1 x 1 convolution matches the shortcut when the shape changes. Global average pooling and dropout 0.4 lead to the softmax. Parameters: 2,289,449. The shortcut follows He et al., ResNet.

`deep_cnn` and `residual_cnn` use global average pooling so the 13,065-way layer is not attached to a very large flatten.

## Results

Fill this table only after a full 13,065-class run without `--max-classes`. `train.py` writes the same numbers to `results/comparison.md`.

| Model | Test top-1 | Top-5 |
| --- | --- | --- |
| shallow_cnn | pending | pending |
| deep_cnn | pending | pending |
| residual_cnn | pending | pending |

Best model: pending.

If every character has 40 training photos, augmentation produces 13,065 x 200 = 2,613,000 training images. The default run is 5 epochs with batch size 128.

## Layout

```text
download_data.py              download and join the split archive
prepare_records.py            split, augment, and write TFRecords
preview_augmentation.py       save a five-version preview
train.py                      train the three models and score the test set
predict.py                    classify one image
requirements.txt
notebooks/DIT5411_colab.ipynb
src/augment.py                OpenCV rotation, shear, and scale
src/data.py                   split rules and TFRecords
src/models.py                 the three networks
```

`data/` and `models/` are not committed. Weights go to `models/best_model.keras`, with class names in `models/labels.txt`.

## Setup

Python 3.9 through 3.13, and TensorFlow 2.16 through 2.20. This code was developed on TensorFlow 2.20. TensorFlow has no Windows package for Python 3.14.

Train the full set on a Colab GPU. The archive is about 1.1 GB, and the TFRecords need several more gigabytes.

### Local

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_data.py --variant full --output data
.\.venv\Scripts\python.exe prepare_records.py --zip data/all_data.zip --index-only
.\.venv\Scripts\python.exe prepare_records.py --zip data/all_data.zip --out data/records
.\.venv\Scripts\python.exe train.py --records data/records --epochs 5 --batch-size 128
```

`--index-only` only reads the zip directory. Continue after it reports 13,065 characters, about 684,677 images, and readable Chinese class names.

If Colab disconnects, run the same training command with `--resume`.

Classify one image:

```powershell
.\.venv\Scripts\python.exe predict.py --image path\to\image.png
```

### Colab

1. Set the runtime to GPU.
2. Open `notebooks/DIT5411_colab.ipynb` and set `REPO` to your GitHub URL.
3. Run download, `--index-only`, the preview, TFRecord creation, then training.
4. Copy top-1 and top-5 from `results/comparison.md` into the table above and push the repository.

If training stops because the session ended:

```python
!python train.py --records /content/records --epochs 5 --batch-size 128 --img-size 64 --resume
```

## References

- Po-Chuan Chen, Traditional Chinese Handwriting Dataset, 2020. https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset
- Full archive: https://github.com/chenkenanalytic/handwritting_data_all
- Darshita Jain, Introduction to OpenCV. https://github.com/darshitajain/Introduction-to-OpenCV
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. CVPR, 2016.
- TensorFlow. https://www.tensorflow.org/
- README layout reference: https://github.com/kimanalytics/Handwritten-Digit-Recognition-using-Keras-and-TensorFlow
