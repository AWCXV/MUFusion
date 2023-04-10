# Training DenseFuse network
# auto-encoder

import os
import sys
import time
from utils import gradient, gradient2
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from testMat import showLossChart
from torch.autograd import Variable
from scipy.misc import imread, imsave, imresize
import utils
from net import TwoFusion_net
from args_fusion import args
from utils import sumPatch
import pytorch_msssim
import torchvision.models as models


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    #train_num = 40000
    train_num = 70000
    #train_num = 50;
    # for i in range(5):
    train()


def train():
    vgg_model = models.vgg19(pretrained=True)
    if (args.cuda):
        vgg_model = vgg_model.cuda(args.device);
    vggFeatures = [];
    vggFeatures.append(vgg_model.features[:4]);#64
    vggFeatures.append(vgg_model.features[:9]);#32
    vggFeatures.append(vgg_model.features[:18]);#16
    vggFeatures.append(vgg_model.features[:27]);#8
    vggFeatures.append(vgg_model.features[:36]);#4    
    for i in range(0,5):
        for parm in vggFeatures[i].parameters():
            parm.requires_grad = False;
            
    patchPrePath = "MF_patches";
    PatchPaths = utils.loadPatchesPairPaths()

    batch_size = args.batch_size

    # load network model, RGB
    in_c = 2 # 1 - gray; 3 - RGB
    if in_c == 2:
        img_model = 'L'
    else:
        img_model = 'RGB'
    input_nc = in_c
    output_nc = 1
    densefuse_model = TwoFusion_net(input_nc, output_nc)
    # for k,v in densefuse_model.named_parameters():
        # for j in range(3):
                # if (k.startswith('conv'+str(j))):
                    # v.requires_grad = False;    
    
    print(densefuse_model)
    optimizer = Adam(densefuse_model.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss(reduction="mean")
    ssim_loss = pytorch_msssim.msssim
    if (args.cuda):
        densefuse_model.cuda(int(args.device));

    tbar = trange(args.epochs)
    print('Start training.....')

    Loss_pixel = []
    Loss_pixel2 = []
    Loss_pixel3 = []
    Loss_ssim = []
    Loss_ssim2 = []
    Loss_grad = []
    Loss_grad2 = []
    Loss_all = []
    all_ssim_loss = 0.
    all_ssim_loss2 = 0.
    all_pixel_loss = 0.
    all_pixel_loss2 = 0.
    all_pixel_loss3 = 0.
    all_grad_loss = 0.
    all_grad_loss2 = 0.
    pow2 = 0;
    for e in tbar:
        print('Epoch %d.....' % e)
        # load training database
        patchesPaths,batches = utils.load_datasetPair(PatchPaths,batch_size);
        densefuse_model.train()
        count = 0
        if (e>0):
            pow2 = pow2+0.5;
        for batch in range(batches):
            image_paths = patchesPaths[batch * batch_size:(batch * batch_size + batch_size)]
            
            image_ir = utils.get_train_images_auto(patchPrePath+"/IR",image_paths, mode="L");
            image_vi = utils.get_train_images_auto(patchPrePath+"/VIS",image_paths, mode="L");
            h = image_ir.shape[2];
            w = image_ir.shape[3];

            count += 1
            optimizer.zero_grad()
            
            img_ir = Variable(image_ir, requires_grad=False)
            img_vi = Variable(image_vi, requires_grad=False)
            
            if args.cuda:
                img_ir = img_ir.cuda(args.device)
                img_vi = img_vi.cuda(args.device)            
            
            img_irdup = torch.cat([img_ir,img_ir,img_ir],1);
            img_vidup = torch.cat([img_vi,img_vi,img_vi],1);
            
            g_ir = vggFeatures[0](img_irdup);
            g_vi = vggFeatures[0](img_vidup);
            g_ir = g_ir.sum(dim=1, keepdim=True);
            g_vi = g_vi.sum(dim=1, keepdim=True);

            g_ir = sumPatch(g_ir,9);
            g_vi = sumPatch(g_vi,9);

            w_ir = g_ir.greater(g_vi);
            w_vi = ~w_ir;

            en = densefuse_model.encoder(torch.cat([img_ir,img_vi],1));
            outputs = densefuse_model.decoder(en)
            ssim_loss_value = 0.
            ssim_loss_value2 = 0.
            ssim_loss_value3 = 0.
            pixel_loss_value = 0.
            pixel_loss_value2 = 0.
            pixel_loss_value3 = 0.
            grad_loss_value = 0.;
            grad_loss_value2 = 0.;
            grad_loss_value3 = 0.;
            output = outputs[0];
            
                    
                    

            
            label = Variable(output.data.clone(),requires_grad=False);
            if (e>0):
                label = utils.get_train_images_auto("./myLabel",image_paths, mode="L");
                if (args.cuda):
                    label = label.cuda(args.device)
            label.requires_grad=False;
            grad_loss_temp = mse_loss(gradient(output*w_vi),gradient(img_vi*w_vi));
            grad_loss_temp2 = mse_loss(gradient(output*w_ir),gradient(img_ir*w_ir));
            grad_loss_temp3 = mse_loss(gradient(output),gradient(label));
            
            pixel_loss_temp = mse_loss(output*w_vi, img_vi*w_vi);
            pixel_loss_temp2 = mse_loss(output*w_ir, img_ir*w_ir);
            pixel_loss_temp3 = mse_loss(output, label);
            
            ssim_loss_temp = ssim_loss(output*w_vi, img_vi*w_vi, normalize=True)
            ssim_loss_temp2 = ssim_loss(output*w_ir, img_ir*w_ir, normalize=True)
            ssim_loss_temp3 = ssim_loss(output, label, normalize=True)

            grad_loss_value += grad_loss_temp
            grad_loss_value2 += grad_loss_temp2
            
            ssim_loss_value += (1-ssim_loss_temp)
            ssim_loss_value2 += (1-ssim_loss_temp2)
            ssim_loss_value3 += (1-ssim_loss_temp3)
            
            pixel_loss_value += pixel_loss_temp
            pixel_loss_value2 += pixel_loss_temp2
            pixel_loss_value3 += pixel_loss_temp3
            
            grad_loss_value = grad_loss_value*10;
            grad_loss_value2 = grad_loss_value2*10;
            grad_loss_value3 = grad_loss_temp3*10;


            output_copy = Variable(output.data.clone(),requires_grad=False);
            label_copy = Variable(label.data.clone(),requires_grad=False);
            
            img_ir_copy = Variable(img_ir.data.clone(),requires_grad=False);
            img_vi_copy = Variable(img_vi.data.clone(),requires_grad=False);
            
            W1 = 0.;
            W2 = 0.;
            W1 += ssim_loss(output_copy,img_ir_copy, normalize=True)+ssim_loss(output_copy,img_vi_copy, normalize=True);
            W2 += ssim_loss(label_copy,img_ir_copy, normalize=True)+ssim_loss(label_copy,img_vi_copy, normalize=True);

            #print("w1="+str(W1)+".");
            #print("w2="+str(W2)+".");
                
            aW1 = torch.exp(W1)/(torch.exp(W1)+torch.exp(W2));
            aW2 = torch.exp(W2)/(torch.exp(W1)+torch.exp(W2));
            aW1 = aW1.item();
            aW2 = aW2.item();
            #print("aw1="+str(aW1)+".");
            #print("aw2="+str(aW2)+".");
            
            L_content = pixel_loss_value + 10*ssim_loss_value + pixel_loss_value2 + 10*ssim_loss_value2 + grad_loss_value + grad_loss_value2;
            L_se = pixel_loss_value3+10*ssim_loss_value3+grad_loss_value3;
            total_loss = aW1*L_content + aW2*L_se;
            total_loss.backward()
            optimizer.step()
            
            outputCopy = output.cpu().detach().numpy();
            for j,path in enumerate(image_paths):
                imsave("./myLabel/"+path+".png",outputCopy[j,0,:,:]*255);            

            all_ssim_loss += ssim_loss_value.item()
            all_ssim_loss2 += ssim_loss_value2.item()
            
            all_pixel_loss += pixel_loss_value.item()
            all_pixel_loss2 += pixel_loss_value2.item()
            all_pixel_loss3 += pixel_loss_value3.item()
            
            all_grad_loss += grad_loss_value.item();
            all_grad_loss2 += grad_loss_value2.item();
            if (batch + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t grad loss: {:.6f}\t total: {:.6f}".format(
                    time.ctime(), e + 1, count, batches,
                                  all_pixel_loss / args.log_interval,
                                  all_ssim_loss / args.log_interval,all_grad_loss / args.log_interval,
                                  (all_ssim_loss + all_pixel_loss+all_ssim_loss2 + all_pixel_loss2) / args.log_interval
                )
                tbar.set_description(mesg)
                Loss_pixel.append(all_pixel_loss / args.log_interval)
                Loss_pixel2.append(all_pixel_loss2 / args.log_interval)
                Loss_pixel3.append(all_pixel_loss3 / args.log_interval)
                
                Loss_ssim.append(all_ssim_loss / args.log_interval)
                Loss_ssim2.append(all_ssim_loss2 / args.log_interval)
                Loss_grad.append(all_grad_loss/args.log_interval);
                Loss_grad2.append(all_grad_loss2/args.log_interval);
                Loss_all.append((all_ssim_loss + all_pixel_loss+all_ssim_loss2 + all_pixel_loss2) / args.log_interval)

                all_ssim_loss = 0.
                all_ssim_loss2 = 0.
                
                all_pixel_loss = 0.
                all_pixel_loss2 = 0.
                all_pixel_loss3 = 0.
                
                all_grad_loss = 0.
                all_grad_loss2 = 0.

            if (batch + 1) % (100) == 0:
                # save model
                densefuse_model.eval()
                densefuse_model.cpu()
                save_model_filename = "Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
                                      str(time.ctime()).replace(' ', '_').replace(':', '_') + ".model"
                save_model_path = os.path.join(args.save_model_dir, save_model_filename)
                torch.save(densefuse_model.state_dict(), save_model_path)
                # save loss data
                # pixel loss
                loss_data_pixel = np.array(Loss_pixel)
                loss_filename_path = "loss_pixel_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + ".mat"
                save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
                scio.savemat(save_loss_path, {'Loss': loss_data_pixel})
                showLossChart(save_loss_path,args.save_loss_dir+'/loss_pixel.png')
                
                # pixel loss
                loss_data_pixel = np.array(Loss_pixel2)
                loss_filename_path = "loss_pixel2_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + ".mat"
                save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
                scio.savemat(save_loss_path, {'Loss': loss_data_pixel})
                showLossChart(save_loss_path,args.save_loss_dir+'/loss_pixe2.png')                
                
                # pixel loss
                loss_data_pixel = np.array(Loss_pixel3)
                loss_filename_path = "loss_pixel3_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + ".mat"
                save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
                scio.savemat(save_loss_path, {'Loss': loss_data_pixel})
                showLossChart(save_loss_path,args.save_loss_dir+'/loss_pixe3.png')                                
                
                # grad loss
                loss_data_grad = np.array(Loss_grad)
                loss_filename_path = "loss_grad_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + ".mat"
                save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
                scio.savemat(save_loss_path, {'Loss': loss_data_grad})
                showLossChart(save_loss_path,args.save_loss_dir+"/loss_grad.png");
                
                # grad loss2
                loss_data_grad = np.array(Loss_grad2)
                loss_filename_path = "loss_grad2_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + ".mat"
                save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
                scio.savemat(save_loss_path, {'Loss': loss_data_grad})
                showLossChart(save_loss_path,args.save_loss_dir+"/loss_grad2.png");                
                
                # SSIM loss
                loss_data_ssim = np.array(Loss_ssim)
                loss_filename_path = "loss_ssim_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') +  ".mat"
                save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
                scio.savemat(save_loss_path, {'Loss': loss_data_ssim})
                showLossChart(save_loss_path,args.save_loss_dir+"/loss_ssim.png");
                
                # SSIM loss
                loss_data_ssim = np.array(Loss_ssim2)
                loss_filename_path = "loss_ssim2_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_')  + ".mat"
                save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
                scio.savemat(save_loss_path, {'Loss': loss_data_ssim})
                showLossChart(save_loss_path,args.save_loss_dir+"/loss_ssim2.png");                
    
                # all loss
                loss_data_total = np.array(Loss_all)
                loss_filename_path = "loss_total_epoch_" + str(
                    args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_')  + ".mat"
                save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
                scio.savemat(save_loss_path, {'Loss': loss_data_total})
                showLossChart(save_loss_path,args.save_loss_dir+"/allLoss.png");

                densefuse_model.train()
                if (args.cuda):
                    densefuse_model.cuda(int(args.device));
                tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)
    # pixel loss
    loss_data_pixel = np.array(Loss_pixel)
    loss_filename_path = "Final_loss_pixel_epoch_" + str(
        args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':','_') + ".mat"
    save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
    scio.savemat(save_loss_path, {'Loss': loss_data_pixel})
    showLossChart(save_loss_path,args.save_loss_dir+"/loss_pixel.png");
    # SSIM loss
    loss_data_ssim = np.array(Loss_ssim)
    loss_filename_path = "Final_loss_ssim_epoch_" + str(
        args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_')  + ".mat"
    save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
    scio.savemat(save_loss_path, {'Loss': loss_data_ssim})
    showLossChart(save_loss_path,args.save_loss_dir+"/loss_ssim.png");
    # grad loss
    loss_data_grad = np.array(Loss_grad)
    loss_filename_path = "Final_loss_grad_epoch_" + str(
        args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + ".mat"
    save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
    scio.savemat(save_loss_path, {'Loss': loss_data_ssim})
    showLossChart(save_loss_path,args.save_loss_dir+"/loss_grad.png");
    # all loss
    loss_data_total = np.array(Loss_all)
    loss_filename_path = "Final_loss_total_epoch_" + str(
        args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_')  + ".mat"
    save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
    scio.savemat(save_loss_path, {'Loss': loss_data_total})
    showLossChart(save_loss_path,args.save_loss_dir+"/allLoss.png");    
    # save model
    densefuse_model.eval()
    densefuse_model.cpu()
    save_model_filename =  "Final_epoch_" + str(args.epochs) + "_" + \
                          str(time.ctime()).replace(' ', '_').replace(':', '_')  + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(densefuse_model.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


if __name__ == "__main__":
    main()
