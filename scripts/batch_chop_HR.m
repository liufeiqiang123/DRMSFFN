function batch_chop_HR()

dbstop if error;

chop_size = 480;

source_dir = '/home/ser606/clh/DIV2K_Original/DIV2K_train_downsize';
target_dir = strcat('/home/ser606/clh/DIV2K_Original/DIV2K_train_downsize', '_chop', num2str(chop_size));
if ~exist(target_dir, 'file')
    mkdir(target_dir);
end
img_files = dir(fullfile(source_dir, '*.png'));

for i = 1:length(img_files)
    img_fullpath = fullfile(source_dir, img_files(i).name);
    [~, img_name, ext] = fileparts(img_fullpath);
    img = imread(img_fullpath);
    [row, column, ~] = size(img);
    if row<chop_size || column<chop_size
        disp('the image size is lower than chop_size');
        continue;
    end
    str_disp = sprintf('processing the number of %d image -- %s%s. \n', i, img_name, ext);
    fprintf(str_disp);
    for j=1: chop_size: row-chop_size
        for k=1: chop_size: column-chop_size
            target_img = img(j:chop_size+j-1, k:chop_size+k-1, :);
            imwrite(target_img, fullfile(target_dir, [img_name, '_', num2str(j), '_', num2str(k), ext]));
        end
    end
    
    
end