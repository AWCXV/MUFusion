# test phase
import torch
import os
from torch.autograd import Variable
from net import TwoFusion_net
import utils
from scipy.misc import imread, imsave, imresize
from args_fusion import args
import numpy as np
import time
from PIL import Image
torch.set_default_tensor_type(torch.DoubleTensor)

def load_model(path, input_nc, output_nc):

    nest_model = TwoFusion_net(input_nc, output_nc)
    nest_model.load_state_dict(torch.load(path))

    para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para / 1000))

    nest_model.eval()
    #nest_model.cuda()

    return nest_model


def _generate_fusion_image(model, strategy_type, img1, img2):
    # encoder
    en_v = model.encoder(img2)
    en_r = model.encoder(img1)
    f = model.fusion(en_r, en_v, strategy_type=strategy_type)
    img_fusion = model.decoder(f);
    return img_fusion[0]

#PIL Image type "RGB mode"    
def rgb_to_ycbcr(image):
    rgb_array = np.array(image)

    transform_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.169, -0.331, 0.5],
                                 [0.5, -0.419, -0.081]])

    ycbcr_array = np.dot(rgb_array, transform_matrix.T)

    y_channel = ycbcr_array[:, :, 0]
    cb_channel = ycbcr_array[:, :, 1]
    cr_channel = ycbcr_array[:, :, 2]
    
    y_channel = np.clip(y_channel, 0, 255)
    return y_channel, cb_channel, cr_channel

def ycbcr_to_rgb(y, cb, cr):
    ycbcr_array = np.stack((y, cb, cr), axis=-1)

    transform_matrix = np.array([[1, 0, 1.402],
                                 [1, -0.344136, -0.714136],
                                 [1, 1.772, 0]])

    rgb_array = np.dot(ycbcr_array, transform_matrix.T)

    rgb_array = np.clip(rgb_array, 0, 255)

    rgb_array = np.round(rgb_array).astype(np.uint8)

    rgb_image = Image.fromarray(rgb_array, mode='RGB')

    return rgb_image

def fuse_cb_cr(Cb1,Cr1,Cb2,Cr2):
    H, W = Cb1.shape
    Cb = np.ones((H, W))
    Cr = np.ones((H, W))

    for k in range(H):
        for n in range(W):
            if abs(Cb1[k, n] - 128) == 0 and abs(Cb2[k, n] - 128) == 0:
                Cb[k, n] = 128
            else:
                middle_1 = Cb1[k, n] * abs(Cb1[k, n] - 128) + Cb2[k, n] * abs(Cb2[k, n] - 128)
                middle_2 = abs(Cb1[k, n] - 128) + abs(Cb2[k, n] - 128)
                Cb[k, n] = middle_1 / middle_2

            if abs(Cr1[k, n] - 128) == 0 and abs(Cr2[k, n] - 128) == 0:
                Cr[k, n] = 128
            else:
                middle_3 = Cr1[k, n] * abs(Cr1[k, n] - 128) + Cr2[k, n] * abs(Cr2[k, n] - 128)
                middle_4 = abs(Cr1[k, n] - 128) + abs(Cr2[k, n] - 128)
                Cr[k, n] = middle_3 / middle_4
    return Cb, Cr

def run_demo(model, infrared_path, visible_path, output_path_root, index, fusion_type, network_type, strategy_type, ssim_weight_str, mode):

    ir_img_rgb = Image.open(infrared_path).convert("RGB");
    vi_img_rgb = Image.open(visible_path).convert("RGB");
    ir_img, ir_img_cb, ir_img_cr = rgb_to_ycbcr(ir_img_rgb);    
    vi_img, vi_img_cb, vi_img_cr = rgb_to_ycbcr(vi_img_rgb);
    
    vi_img_cb, vi_img_cr = fuse_cb_cr(vi_img_cb, vi_img_cr, ir_img_cb, ir_img_cr);
    
    
    ir_img=ir_img/255.0;
    vi_img=vi_img/255.0;
    h = vi_img.shape[0];
    w = vi_img.shape[1];
    
    ir_img_patches = [];
    vi_img_patches = [];
    
    ps = args.PATCH_SIZE;
    
    for i in range(0,h-ps+1,ps-4):
        for j in range(0,w-ps+1,ps-4):
            ir_patch = ir_img[i:i+ps,j:j+ps];
            vi_patch = vi_img[i:i+ps,j:j+ps];
            ir_patch = np.resize(ir_patch,[1,ps,ps]);
            vi_patch = np.resize(vi_patch,[1,ps,ps]);
            ir_img_patches.append(ir_patch);
            vi_img_patches.append(vi_patch);
    
    for i in range(0,h-ps+1,ps-4):
        ir_patch = ir_img[i:i+ps,w-ps:w];
        vi_patch = vi_img[i:i+ps,w-ps:w];
        ir_patch = np.resize(ir_patch,[1,ps,ps]);
        vi_patch = np.resize(vi_patch,[1,ps,ps]);
        ir_img_patches.append(ir_patch);
        vi_img_patches.append(vi_patch);        

    for j in range(0,w-ps+1,ps-4):
        ir_patch = ir_img[h-ps:h,j:j+ps];
        vi_patch = vi_img[h-ps:h,j:j+ps];
        ir_patch = np.resize(ir_patch,[1,ps,ps]);
        vi_patch = np.resize(vi_patch,[1,ps,ps]);
        ir_img_patches.append(ir_patch);
        vi_img_patches.append(vi_patch);        

    ir_patch = ir_img[h-ps:h,w-ps:w];
    vi_patch = vi_img[h-ps:h,w-ps:w];
    ir_patch = np.resize(ir_patch,[1,ps,ps]);
    vi_patch = np.resize(vi_patch,[1,ps,ps]);
    ir_img_patches.append(ir_patch);
    vi_img_patches.append(vi_patch);     
    ir_img_patches = np.stack(ir_img_patches,axis=0);
    vi_img_patches = np.stack(vi_img_patches,axis=0);
    ir_img_patches = torch.from_numpy(ir_img_patches);
    vi_img_patches = torch.from_numpy(vi_img_patches);
    
    # dim = img_ir.shape
    if args.cuda:
        ir_img_patches = ir_img_patches.cuda(args.device)
        vi_img_patches = vi_img_patches.cuda(args.device)
        model = model.cuda(args.device);
    ir_img_patches = Variable(ir_img_patches, requires_grad=False)
    vi_img_patches = Variable(vi_img_patches, requires_grad=False)

    img = torch.cat([ir_img_patches,vi_img_patches],1);
    en = model.encoder(img);
    out = model.decoder(en)[0];
    ############################ multi outputs ##############################################
    fuseImage = np.zeros((h,w));
    fuseCnt = np.zeros((h,w));
    
    out = out.cpu();    
    out = out.numpy();
    
    idx = 0;
    for i in range(0,h-ps+1,ps-4):
        for j in range(0,w-ps+1,ps-4):
            fuseImage[i:i+ps,j:j+ps] += out[idx][0];
            fuseCnt[i:i+ps,j:j+ps] += 1;
            idx+=1;
    
    for i in range(0,h-ps+1,ps-4):
        fuseImage[i:i+ps,w-ps:w] += out[idx][0];
        fuseCnt[i:i+ps,w-ps:w] += 1;
        idx+=1;

    for j in range(0,w-ps+1,ps-4):
        fuseImage[h-ps:h,j:j+ps] += out[idx][0];
        fuseCnt[h-ps:h,j:j+ps] += 1;
        idx+=1;

    fuseImage[h-ps:h,w-ps:w] += out[idx][0];
    fuseCnt[h-ps:h,w-ps:w] += 1;
    fuseImage = fuseImage/fuseCnt;
    
    fuseImage = fuseImage*255;
    
    fuseImage = ycbcr_to_rgb(fuseImage, vi_img_cb, vi_img_cr);
    
    file_name = 'fuse'+str(index) + '.jpg'
    output_path = output_path_root + file_name

    imsave(output_path,fuseImage);

    print(output_path)

def main():

    #path to your images
    test_path = "input_images_RGB/"

    network_type = 'densefuse'
    fusion_type = 'auto'  # auto, fusion_layer, fusion_all
    strategy_type_list = ['AVG', 'L1','SC']  # addition, attention_weight, attention_enhance, adain_fusion, channel_fusion, saliency_mask

    strategy_type = strategy_type_list[1]
    output_path = './outputs_ME_rgb/';

    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    # in_c = 3 for RGB images; in_c = 1 for gray images
    in_c = 2
    out_c = 1
    mode = 'L'
    model_path = args.model_path_gray

    with torch.no_grad():
        print('SSIM weight ----- ' + args.ssim_path[2])
        ssim_weight_str = args.ssim_path[2]
        model = load_model(model_path, in_c, out_c)
        for i in range(1):
            index = i + 1
            infrared_path = test_path + 'IR' + str(index) + '.png'
            visible_path = test_path + 'VIS' + str(index) + '.png'
            run_demo(model, infrared_path, visible_path, output_path, index, fusion_type, network_type, strategy_type, ssim_weight_str, mode)
    print('Done......')

if __name__ == '__main__':
    main()
