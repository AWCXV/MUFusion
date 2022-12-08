
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
