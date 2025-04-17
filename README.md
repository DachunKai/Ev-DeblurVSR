# [Ev-DeblurVSR (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/32438)

Official Pytorch implementation for the "Event-Enhanced Blurry Video Super-Resolution" paper (AAAI 2025).

<p align="center">
    🌐 <a href="https://dachunkai.github.io/ev-deblurvsr.github.io/" target="_blank">Project</a> | 📃 <a href="https://ojs.aaai.org/index.php/AAAI/article/view/32438" target="_blank">Paper</a>  <br>
</p>

**Authors**: [Dachun Kai](https://github.com/DachunKai/)<sup>[:email:️](mailto:dachunkai@mail.ustc.edu.cn)</sup>, [Yueyi Zhang](https://scholar.google.com.hk/citations?user=LatWlFAAAAAJ&hl=zh-CN&oi=ao), [Jin Wang](https://github.com/booker-max), [Zeyu Xiao](https://dblp.org/pid/276/3139.html), [Zhiwei Xiong](https://scholar.google.com/citations?user=Snl0HPEAAAAJ&hl=zh-CN), [Xiaoyan Sun](https://scholar.google.com/citations?user=VRG3dw4AAAAJ&hl=zh-CN), *University of Science and Technology of China*

**Feel free to ask questions. If our work helps, please don't hesitate to give us a :star:!**

## :rocket: News
- [x] 2025/04/17: Release pretrained models and test sets for quick testing
- [x] 2025/01/07: Video demos released
- [x] 2024/12/15: Initialize the repository
- [x] 2024/12/09: :tada: :tada: Our paper was accepted in AAAI'2025

## :bookmark: Table of Content
1. [Video Demos](#video-demos)
2. [Code](#code)
3. [Citation](#citation)
4. [Contact](#contact)
5. [License and Acknowledgement](#license-and-acknowledgement)

## :fire: Video Demos
A $4\times$ blurry video upsampling results on the synthetic dataset [GoPro](https://seungjunnah.github.io/Datasets/gopro.html) and real-world dataset [NCER](https://sites.google.com/view/neid2023) test sets.

https://github.com/user-attachments/assets/df54a750-25fd-4ac1-9980-20ef7f73c738

https://github.com/user-attachments/assets/4d58c85f-1a47-4292-8e4a-4ea0ccfe1b0d

https://github.com/user-attachments/assets/cb7c3a62-5927-4f5a-8aec-258d7e1d513e

https://github.com/user-attachments/assets/0c030756-f2a0-4a9d-81a2-99943a0f881f

## Code
### Installation
* Dependencies: [Miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh), [CUDA Toolkit 11.1.1](https://developer.nvidia.com/cuda-11.1.1-download-archive), [torch 1.10.2+cu111](https://download.pytorch.org/whl/cu111/torch-1.10.2%2Bcu111-cp37-cp37m-linux_x86_64.whl), and [torchvision 0.11.3+cu111](https://download.pytorch.org/whl/cu111/torchvision-0.11.3%2Bcu111-cp37-cp37m-linux_x86_64.whl).

* Run in Conda (**Recommend**)

    ```bash
    conda create -y -n ev-deblurvsr python=3.7
    conda activate ev-deblurvsr
    pip install torch-1.10.2+cu111-cp37-cp37m-linux_x86_64.whl
    pip install torchvision-0.11.3+cu111-cp37-cp37m-linux_x86_64.whl
    git clone https://github.com/DachunKai/Ev-DeblurVSR
    cd Ev-DeblurVSR && pip install -r requirements.txt && python setup.py develop
    ```
* Run in Docker :clap:

  Note: **We use the same docker image as our previous work [EvTexture](https://github.com/DachunKai/EvTexture)**.

  [Option 1] Directly pull the published Docker image we have provided from [Alibaba Cloud](https://cr.console.aliyun.com/cn-hangzhou/instances).
  ```bash
  docker pull registry.cn-hangzhou.aliyuncs.com/dachunkai/evtexture:latest
  ```

  [Option 2] We also provide a [Dockerfile](https://github.com/DachunKai/Ev-DeblurVSR/blob/main/docker/Dockerfile) that you can use to build the image yourself.
  ```bash
  cd EvTexture && docker build -t evtexture ./docker
  ```
  The pulled or self-built Docker image containes a complete conda environment named `evtexture`. After running the image, you can mount your data and operate within this environment.
  ```bash
  source activate evtexture && cd EvTexture && python setup.py develop
  ```
### Test
1. Download the pretrained models from ([Releases](https://github.com/DachunKai/Ev-DeblurVSR/releases) / [Baidu Cloud](https://pan.baidu.com/s/1Y4ZW9PDV_ff2Z4VxadzrzA?pwd=n8hg) (n8hg)) and place them to `experiments/pretrained_models/EvDeblurVSR/`. The network architecture code is in [evdeblurvsr_arch.py](https://github.com/DachunKai/Ev-DeblurVSR/blob/main/basicsr/archs/evdeblurvsr_arch.py).
    - Synthetic dataset model:
      * *EvDeblurVSR_GOPRO_BIx4.pth*: trained on [GoPro](https://seungjunnah.github.io/Datasets/gopro.html) dataset with Blur-Sharp pairs and BI degradation for $4\times$ SR scale.
      * *EvDeblurVSR_BSD_BIx4.pth*: trained on [BSD](https://github.com/zzh-tech/ESTRNN) dataset with Blur-Sharp pairs and BI degradation for $4\times$ SR scale.
    - Real-world dataset model:
      * *EvDeblurVSR_NCER_BIx4.pth*: trained on [NCER](https://sites.google.com/view/neid2023) dataset with Blur-Sharp pairs and BI degradation for $4\times$ SR scale.

2. Download the preprocessed test sets (including events) for [GoPro](https://seungjunnah.github.io/Datasets/gopro.html), [BSD](https://github.com/zzh-tech/ESTRNN), and [NCER](https://sites.google.com/view/neid2023) from ([Baidu Cloud](https://pan.baidu.com/s/1Y4ZW9PDV_ff2Z4VxadzrzA?pwd=n8hg) (n8hg) / [Google Drive](https://drive.google.com/drive/folders/1Py9uESwTAD0lhRgvhBGXo-uODxC-wGTw?usp=sharing)), and place them to `datasets/`.
    * *GoPro_h5*: HDF5 files containing preprocessed test datasets for the GoPro testset.

    * *BSD_h5*: HDF5 files containing preprocessed test datasets for the BSD dataset.

    * *NCER_h5*: HDF5 files containing preprocessed test datasets for the NCER dataset.

3. Run the following command:
    * Test on GoPro for 4x Blurry VSR:
      ```bash
      ./scripts/dist_test.sh [num_gpus] options/test/EvDeblurVSR/test_EvDeblurVSR_GoPro_x4.yml
      ```
    * Test on BSD for 4x Blurry VSR:
      ```bash
      ./scripts/dist_test.sh [num_gpus] options/test/EvDeblurVSR/test_EvDeblurVSR_BSD_x4.yml
      ```
    * Test on NCER for 4x Blurry VSR:
      ```bash
      ./scripts/dist_test.sh [num_gpus] options/test/EvDeblurVSR/test_EvDeblurVSR_NCER_x4.yml
      ```
    This will generate the inference results in `results/`. The output results on GoPro, BSD and NCER datasets can be downloaded from ([Releases](https://github.com/DachunKai/Ev-DeblurVSR/releases) / [Baidu Cloud](https://pan.baidu.com/s/1Y4ZW9PDV_ff2Z4VxadzrzA?pwd=n8hg) (n8hg)).

4. Test the number of parameters, runtime, and FLOPs:
    ```bash
    python test_scripts/test_params_runtime.py
    ```

### Input Data Structure
* Both video and event data are required as input, as shown in the [snippet](https://github.com/DachunKai/Ev-DeblurVSR/blob/main/basicsr/archs/evdeblurvsr_arch.py#L229). We package each video and its event data into an [HDF5](https://docs.h5py.org/en/stable/quick.html#quick) file.

* Example: The structure of `GOPR0384_11_00.h5` file from the GoPro dataset is shown below.

  ```arduino
  GOPR0384_11_00.h5
  ├── images
  │   ├── 000000 # frame, ndarray, [H, W, C]
  │   ├── ...
  ├── vFwd
  │   ├── 000000 # inter-frame forward event voxel, ndarray, [Bins, H, W]
  │   ├── ...
  ├── vBwd
  │   ├── 000000 # inter-frame backward event voxel, ndarray, [Bins, H, W]
  │   ├── ...
  ├── vExpo
  │   ├── 000000 # intra-frame exposure event voxel, ndarray, [Bins, H, W]
  │   ├── ...
  ```

## :blush: Citation
If you find the code and pre-trained models useful for your research, please consider citing our paper. :smiley:
```
@inproceedings{kai2025event,
  title={Event-{E}nhanced {B}lurry {V}ideo {S}uper-{R}esolution},
  author={Kai, Dachun and Zhang, Yueyi and Wang, Jin and Xiao, Zeyu and Xiong, Zhiwei and Sun, Xiaoyan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={4},
  pages={4175--4183},
  year={2025}
}
```

## Contact
If you meet any problems, please describe them in issues or contact:
* Dachun Kai: <dachunkai@mail.ustc.edu.cn>

## License and Acknowledgement
This project is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). Our work builds significantly upon our previous project [EvTexture](https://github.com/DachunKai/EvTexture). We would also like to sincerely thank the developers of [BasicSR](https://github.com/XPixelGroup/BasicSR), an open-source toolbox for image and video restoration tasks. Additionally, we appreciate the inspiration and code provided by [BasicVSR++](https://github.com/ckkelvinchan/BasicVSR_PlusPlus), [RAFT](https://github.com/princeton-vl/RAFT) and [event_utils](https://github.com/TimoStoff/event_utils).
