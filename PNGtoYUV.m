clear all;

%%%====== Settings ======%%%
format = '420';
wid = 3840;
hei = 2160;
dir_name = './test_img_dir/JSI-GAN_x2_exp1';
%%%======================%%%

input_dir = dir(fullfile(dir_name, '*.png'));
file_new = fullfile(dir_name, 'result.yuv');
fclose(fopen(file_new, 'w'));
[fwidth,fheight] = yuv_factor(format);

for i = 1:length(input_dir)/3
    pred_YUV(:, :, 1) = imread(fullfile(input_dir(3*i).folder, input_dir(3*i).name));
    pred_YUV(:, :, 2) = imread(fullfile(input_dir(3*i-2).folder, input_dir(3*i-2).name));
    pred_YUV(:, :, 3) = imread(fullfile(input_dir(3*i-1).folder, input_dir(3*i-1).name));
    save_yuv(pred_YUV, file_new, hei, wid, fheight, fwidth, 'HDR');
end

