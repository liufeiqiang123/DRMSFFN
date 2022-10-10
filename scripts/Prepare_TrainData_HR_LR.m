function Prepare_TrainData_HR_LR()
clear; close all; clc
scale = 4;

source_folder = '/home/clh/Documents/Python/DIV2K/DIV2K_train_HR';
savedir = '/home/clh/Documents/Python/DIV2KLr/train_set';
saveHRpath = fullfile(savedir, 'DIV2K_train_HR_aug', ['x' num2str(scale)]);
saveLRpath = fullfile(savedir, 'DIV2K_train_LR_aug', ['x' num2str(scale)]);

downsizes = [0.8, 0.7, 0.6, 0.5];

if ~exist(saveHRpath, 'dir')
    mkdir(saveHRpath);
end
if ~exist(saveLRpath, 'dir')
    mkdir(saveLRpath);
end

filepaths = [dir(fullfile(source_folder, '*.png'));dir(fullfile(source_folder, '*.bmp'))];
     
for i = 1 : length(filepaths)
    filename = filepaths(i).name;
    fprintf('No.%d -- Processing %s\n', i, filename);
    [add, im_name, type] = fileparts(filepaths(i).name);
    image = imread(fullfile(source_folder, filename));
%     image = modcrop(image, scale);
    
    for angle = 0 : 2 : 2
        for downidx = 0 : 1 : length(downsizes)
            image_HR = image;
            if downidx > 0
                image_HR = imresize(image_HR, downsizes(downidx), 'bicubic');
            end
            image_HR = rot90(image_HR, angle);
            image_HR = modcrop(image_HR, scale);
            image_LR = imresize(image_HR, 1/scale, 'bicubic');
                
            saveHRfile =  [im_name '_rot' num2str(angle*90) '_ds' num2str(downidx) '.png'];
            saveLRfile = [im_name '_rot' num2str(angle*90) '_ds' num2str(downidx) '.png'];
                
            imwrite(image_HR, fullfile(saveHRpath, saveHRfile));    
            imwrite(image_LR, fullfile(saveLRpath, saveLRfile));               
        end          
    end
end

end

function imgs = modcrop(imgs, modulo)
if size(imgs,3)==1
    sz = size(imgs);
    sz = sz - mod(sz, module);
    imgs = imgs(1:sz(1), 1:sz(2));
else
    tmpsz = size(imgs);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2),:);
end
end
