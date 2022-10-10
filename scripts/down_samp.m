function down_samp()
clear; close all; clc
%scale = 4;

source_folder = '/home/ser606/clh/DIV2K_Original/DIV2K_train_HR';
saveHRpath = '/home/ser606/clh/DIV2K_Original/DIV2K_train_downsize';

%saveLRpath = fullfile(savedir, 'DIV2K_train_LR_aug', ['x' num2str(scale)]);

downsizes = [0.8, 0.7, 0.6, 0.5];

if ~exist(saveHRpath, 'dir')
    mkdir(saveHRpath);
end

filepaths = [dir(fullfile(source_folder, '*.png'));dir(fullfile(source_folder, '*.bmp'))];
     
for i = 435 : length(filepaths)
    filename = filepaths(i).name;
    fprintf('No.%d -- Processing %s\n', i, filename);
    [add, im_name, type] = fileparts(filepaths(i).name);
    image = imread(fullfile(source_folder, filename));
%     image = modcrop(image, scale);
    
 
        for downidx = 0 : 1 : length(downsizes)
            image_HR = image;
            if downidx > 0
                image_HR = imresize(image_HR, downsizes(downidx), 'bicubic');
            end     
            saveHRfile =  [im_name '_ds' num2str(downidx) type]; 
            imwrite(image_HR, fullfile(saveHRpath, saveHRfile));    
                      
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
