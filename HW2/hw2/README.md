# Environment Setup (conda is strongly suggested)

Create virtual environment via `conda`.

```bash
# for Windows
conda create -n cv_hw2_py36 python==3.6.13
conda activate cv_hw2_py36
conda install menpo::cyvlfeat
pip install -r requirements_py36.txt

# for Mac(Intel) / Linux
conda create -n cv_hw2_py38 python==3.8.20
conda activate cv_hw2_py38
conda install -c conda-forge cyvlfeat
pip install -r requirements_py38.txt

# for Mac(ARM)
conda create -n cv_hw2_py38
conda activate cv_hw2_py38
conda config --env --set subdir osx-64
conda install python==3.8.20
conda install -c conda-forge cyvlfeat
pip install -r requirements_py38.txt
```

This part is for those who encounter errors while installing `cyvlfeat`. Any version of `cyvlfeat` is allowed. Try one of the following commands to install,

```bash
# (recommend)
conda install -c conda-forge cyvlfeat
# (alternative #1)
conda install menpo::cyvlfeat
# (alternative #2)
pip install cyvlfeat
```

For people have trouble downloading cyvlfeat, try opencv-python
```bash
pip install opencv-python
```

# Download Dataset

You can manually download the HW2 dataset from [2026 CV_HW2](https://drive.google.com/file/d/15yF3-CalFD1Bn2w6VV2vDYY87lDuONo0/view?usp=sharing) or run the following command in terminal,

```bash
gdown --fuzzy https://drive.google.com/file/d/15yF3-CalFD1Bn2w6VV2vDYY87lDuONo0/view?usp=sharing
```