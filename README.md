# JSI-GAN
**This is the official repository of JSI-GAN (AAAI2020).**

We provide the training and test code along with the trained weights and the dataset (train+test) used for JSI-GAN.
If you find this repository useful, please consider citing our paper.

**Reference**:  
> Soo Ye Kim*, Jihyong Oh*, and Munchurl Kim "JSI-GAN: GAN-Based Joint Super-Resolution and Inverse Tone-Mapping with Pixel-Wise Task-Specific Filters for UHD HDR Video", *Thirty-Fourth AAAI Conference on Artificial Intelligence*, 2020. (*equal contribution)

### Requirements
Our code is implemented using Tensorflow, and was tested under the following setting:  
* Python 3.6  
* Tensorflow 1.13 
* CUDA 9.0  
* cuDNN 7.1.4  
* NVIDIA TITAN Xp GPU

## Test code
### Quick Start
1. Download the source code in a directory of your choice **\<source_path\>**.
2. Download the test dataset from [this link](https://drive.google.com/file/d/1dZTwvRhf189L7NLkAcpij4980fyEXq3Q/view?usp=sharing) and unzip the 'test' folder in **\<source_path\>/data**.
3. Download the pre-trained weights from [this link](). (**This link will be updated soon**)
4. Run **main.py** with the following options in parse_args:  
**(i) For testing the .mat file input with scale factor 2:**  
'--phase' as **'test_mat'**, '--scale_factor' as **2**, '--test_data_path_LR_SDR' as **'./data/test/testset_SDR_x2.mat'**, '--test_data_path_HR_HDR' as **'./data/test/testset_HDR.mat'**  
**(ii) For testing the .mat file input with scale factor 4:**  
'--phase' as **'test_mat'**, '--scale_factor' as **4**, '--test_data_path_LR_SDR' as **'./data/test/testset_SDR_x4.mat'**, '--test_data_path_HR_HDR' as **'./data/test/testset_HDR.mat'**  
**(iii) For testing the .png file input with scale factor 2:**  
'--phase' as **'test_png'**, '--scale_factor' as **2**, '--test_data_path_LR_SDR' as **'./data/test/PNG/SDR_x2'**, '--test_data_path_HR_HDR' as **'./data/test/PNG/HDR'**  
**(iv) For testing the .mat file input with scale factor 4:**  
'--phase' as **'test_png'**, '--scale_factor' as **4**, '--test_data_path_LR_SDR' as **'./data/test/PNG/SDR_x4'**, '--test_data_path_HR_HDR' as **'./data/test/PNG/HDR'**  

### Description
* **Running the test_mat option** will read the **.mat** file and save the predicted HR HDR results in .png format in **\<source_path\>/test_img_dir**. The YUV channels are saved separately as 16-bit PNG files.
* **Running the test_png option** will read **.png** files in the designated folder and save the predicted HR HDR results in .png format in **\<source_path\>/test_img_dir**. The YUV channels are saved separately as 16-bit PNG files.
* If you wish to convert the produced .png files to a **.yuv video**, you could run the provided **PNGtoYUV.m** Matlab code, which would save the .yuv video in the same location as the .png files. **Please set the directory name accordingly.** (Encoding the raw .yuv video to a compressed video format such as .mp4 can be done using ffmpeg)
* If you wish to **test your own LR SDR .png files**, you can designate a folder of your choice as the option for '--test_data_path_LR_SDR' and '--test_data_path_HR_HDR'.
* If you **do not have the HR HDR ground truth**, you could comment the lines related to 'label', 'GT' and 'PSNR' in the test_png function in net.py.
* **For faster testing** (to acquire PSNR results), you may comment the line for saving the predicted images as .png files.
* **Due to GPU memory constraints**, the full 4K frame may fail to be tested at one go. The '--test_patch' option defines the number of patches (H, W) to divide the input frame (e.g. (1, 1) means the full 4K frame will be entered, (2, 2) means that it will be divided into 2x2 2K frame patches and processed serially). You may modify this variable so that the testing works with your GPU.

## Training code
### Quick Start
1. Download the source code in a directory of your choice **\<source_path\>**.
2. Download the train dataset from [this link](https://drive.google.com/file/d/19cp91wSRSrOoEdPeQkfMWisou3gJoh-7/view?usp=sharing) and unzip the 'train' folder in **\<source_path\>/data**.   
3. Run **main.py** with the following options in parse_args:  
**(i) For training the model with scale factor 2:**  
'--phase' as **'train'**, '--scale_factor' as **2**, '--train_data_path_LR_SDR' as **'./data/train/SDR_youtube_80.mat'**, '--train_data_path_HR_HDR' as **'./data/train/HDR_youtube_80.mat'**  
**(ii) For training the model with scale factor 4:**  
'--phase' as **'train'**, '--scale_factor' as **4**, '--train_data_path_LR_SDR' as **'./data/train/SDR_youtube_80_x4.mat'**, '--train_data_path_HR_HDR' as **'./data/train/HDR_youtube_80.mat'**  

### Description
* **Running the train option** will train JSI-GAN with the proposed training scheme (**training JSInet first and then fine-tuning JSI-GAN**) and save the trained weights in **\<source_path\>/checkpoint_dir**.
* The trained model can be tested with **test_mat** or **test_png** options.
* If you wish to compare with the provided weights, **change the '--exp_num' option** before training to another number than 1 to avoid overwriting the provided pre-trained weights so that the new weights are saved in a different folder (e.g. JSI-GAN_x2_exp2).
* **The training process can be monitored using Tensorboard.** The log directory for using Tensorboard is **\<source_path\>/logs**.

## Contact
Please contact the authors via email (sooyekim@kaist.ac.kr or jhoh94@kaist.ac.kr) for any problems regarding the released code.
