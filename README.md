# IronDepth: Iterative Refinement of Single-View Depth using Surface Normal and its Uncertainty

Official implementation of the paper

> **IronDepth: Iterative Refinement of Single-View Depth using Surface Normal and its Uncertainty** \
> BMVC 2022 \
> [Gwangbin Bae](https://baegwangbin.com), [Ignas Budvytis](https://mi.eng.cam.ac.uk/~ib255/), and [Roberto Cipolla](https://mi.eng.cam.ac.uk/~cipolla/) \
> [[arXiv]](https://arxiv.org/abs/2210.03676) [[demo]](https://www.youtube.com/watch?v=mf8keH9brF0) [[project page]](https://baegwangbin.github.io/IronDepth/)

<p align="center">
  <img width=100% src="https://github.com/baegwangbin/IronDepth/blob/main/docs/img/irondepth/IronDepth_short.gif">
</p>

## Summary

* We use [surface normal](https://github.com/baegwangbin/surface_normal_uncertainty) to propagate depth between pixels.
* We formulate depth refinement/upsampling as classification of choosing the neighboring pixel to propagate from.

## Getting Started

We recommend using a virtual environment.
```
python3.6 -m venv --system-site-packages ./venv
source ./venv/bin/activate
```

Install the necessary dependencies by
```
python3.6 -m pip install -r requirements.txt
```

Go to this [google drive](https://drive.google.com/drive/folders/1idIVqOrJOK6kuidBng1K8sth-CyOfcCj?usp=sharing), and

* Download `*.pt` and place them under `./checkpoints`. 
* Download and unzip `examples.zip` as `./examples`.

## Testing

```python
# test on scannet images, using the model trained on scannet
python test.py --train_data scannet --test_data scannet

# test on nyuv2 images, using the model trained on nyuv2
python test.py --train_data nyuv2 --test_data nyuv2

# test on your own images, using the model trained on scannet
python test.py --train_data scannet --test_data custom
```

* This generates output visualizations under `./examples/output/dataset_name/`.
* Comment out unnecessary visualization scripts to speed things up.
* When testing on your own images, you should place the images under `./examples/data/custom/`. We support `.png` and `.jpg` files. If you wish to provide the camera intrinsics, add a file named `img_name.txt`. The file should contain `fx`, `fy`, `cx` and `cy`. See `./examples/data/custom/ex01.txt` as an example.

## Training

We provide the training script for ScanNet images. It is straightforward to apply the same code for other datasets. 

### Step 1. Data Preparation

Firstly, go to this [google drive](https://drive.google.com/drive/folders/1idIVqOrJOK6kuidBng1K8sth-CyOfcCj?usp=sharing). Download and unzip `scannet.zip` as `./scannet`. The folder has two sub-folders named `train` and `test`. For each of them, there is a set of `scenes`. Images in each scene are assumed to be taken with the same camera. The camera intrinsics `(fx, fy, cx, cy)` should be provided as `intrins.txt`. For each image, you should have four files:

* `000000_img.png`: RGB image
* `000000_depth.png`: GT depth map
* `000000_norm.png`: Predicted normal map
* `000000_kappa.png`: Predicted normal uncertainty

We generated normal predictions *offline* instead of generating them on the fly. If you have a dataset with no surface normal prediction, add additional scenes/images and run

```
python preprocess.py
```

### Step 2. Training

To train the network, run

```
python train.py
```

Note that the provided `scannet` mini dataset only contains 100 images for training and 10 images for testing. You should train the network on a bigger dataset to obtain satisfactory results.

## Citation

If you find our work useful in your research please consider citing our papers:

```
@InProceedings{Bae2022,
    title   = {IronDepth: Iterative Refinement of Single-View Depth using Surface Normal and its Uncertainty}
    author  = {Gwangbin Bae and Ignas Budvytis and Roberto Cipolla},
    booktitle = {British Machine Vision Conference (BMVC)},
    year = {2022}                         
}
```

```
@InProceedings{Bae2021,
    title   = {Estimating and Exploiting the Aleatoric Uncertainty in Surface Normal Estimation}
    author  = {Gwangbin Bae and Ignas Budvytis and Roberto Cipolla},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year = {2021}                         
}
```