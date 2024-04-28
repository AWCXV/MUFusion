# MUFusion
This is the code of the paper titled as "MUFusion: A general unsupervised image fusion network based on memory unit". 

The paper can be found [here](https://www.sciencedirect.com/science/article/abs/pii/S1566253522002202).

## Environment
- Python 3.7.3
- torch 1.9.1
- scipy 1.2.0 (alternatively, you can use the cv2.imread to replace the imread from scipy.misc)

## To test on the pre-trained model
Put your image pairs in the "test_images" directory and run the following prompt: 
```
python test_image.py
```
You may need to modify related variables in "test_image.py" and the model name in "args_fusion.py"

Tips: If you have difficulty resolving the "imsave" from the scipy package, in all files, please replace "imread" with "cv2.imread(path,0)" & replace "imsave" with "cv2.imwrite(path, fuseImage)";

For calculating the image quality assessments, please refer to this [repository](https://github.com/Linfeng-Tang/SeAFusion/tree/main/Evaluation).

2023-5-25: RGB inference code for different tasks are avaialble now.

## To train
Put the training patches in the "XX_Patches" directory and run the following prompt:
```
python train.py
```
The training informaiton (number of samples, batch size etc.) can be changed in the "args_fusion.py"

The raw training patches will be available [here](https://drive.google.com/drive/folders/1Tf6wwgGhRE7X8g4pLVFAXBdSZdXfgogJ?usp=share_link).

To create your own dataset and compare algorithms in the same environment, please refer to this [code](https://github.com/AWCXV/MUFusion/blob/main/ir_vis/IV_patches/Generating_patches.py) for generating the patches.

Links to the original datasets: [TNO](https://pan.baidu.com/s/1GgzYfIA_Vs0oZbB_Kmh8vQ?pwd=18sb) (password：18sb), [RoadScene](https://github.com/hanna-xu/RoadScene), [Milti-exposure](https://github.com/csjcai/SICE), [Multi-focus](https://pan.baidu.com/s/1ciol2g8cAc-1Fp_UqA31eg?pwd=4js3) (password：4js3), [Medical](http://www.med.harvard.edu/AANLIB/home.html).

## Contact Informaiton
If you have any questions, please contact me at <chunyang_cheng@163.com>.

## Acknowledgement
Code of this implementation is based on the [DenseFuse](https://github.com/hli1221/densefuse-pytorch).

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
