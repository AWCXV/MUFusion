# MUFusion
This is the code of the paper titled as "MUFusion: A general unsupervised image fusion network based on memory unit". 

The paper can be found [here](https://www.sciencedirect.com/science/article/abs/pii/S1566253522002202).

## Usage

### To test on the pre-trained model
Put your image pairs in the "input_images" directory and run the following code. 
```
python test_image.py
```
You may need to modify related variables in "test_image.py" and the model name in "args_fusion.py"

### To train
Put the training patches in the "IV_patches" directory and run the following code.
```
python train.py
```
The training informaiton (number of samples, batch size etc.) can be changed in the "args_fusion.py"

The training data will be available on the Baidu Netdisk.

## Environment
- Python 3.7.3
- torch 1.9.1
- scipy 1.2.0

## Acknowledgement
Code of this implementation is based on the [DenseFuse](https://github.com/hli1221/densefuse-pytorch).

## Contact Informaiton
If you have any questions, please contact me at <chunyang_cheng@163.com>.

# Citation
If this work is helpful to you, please cite it as (BibTeX):
```
@article{cheng2023mufusion,
  title={MUFusion: A general unsupervised image fusion network based on memory unit},
  author={Cheng, Chunyang and Xu, Tianyang and Wu, Xiao-Jun},
  journal={Information Fusion},
  volume={92},
  pages={80--92},
  year={2023},
  publisher={Elsevier}
}
```
