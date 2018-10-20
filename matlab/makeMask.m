clear
close all

path = './';

files = dir(strcat(path,'/myfile_d16bit*.png'));

for dataNumber = 1:size(files,1)
    dataNumber
    fileName = files(dataNumber).name;
    folder = files(dataNumber).folder;
    depth = imread ([folder, '/', fileName]);
    mask = uint8(zeros(size(depth)));
    mask(depth>200 & depth<245) = 255;
    figure(1); imshow(mask);
    imwrite(mask, ['omask_', sprintf('%06d', dataNumber-1), '.png']);
    imwrite(depth, ['depth_', sprintf('%06d', dataNumber-1), '.png']);
    
    image = imread([folder, '/myfile_i', sprintf('%03d',dataNumber-1), '.png']);
    imwrite(image, ['color_', sprintf('%06d', dataNumber-1), '.png']);
end