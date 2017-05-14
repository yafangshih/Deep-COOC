function [opts, imdb] = setup(opts)

setup_matconvnet ;

% -------------------------------------------------------------------------
% setup dataset & cnn model
% -------------------------------------------------------------------------

if(~exist(opts.dataDir, 'dir'))
    vl_xmkdir(opts.dataDir);
end

opts.datasetDir = fullfile(opts.dataDir, opts.dataset);
if(~exist(opts.datasetDir, 'dir'))
    vl_xmkdir(opts.datasetDir);
    error('CUB dataset does not exist. please download it, and put the extracted files into data/cub/');
end

if(~exist(fullfile(opts.dataDir, 'models'), 'dir'))
    vl_xmkdir(fullfile(opts.dataDir, 'models'));
end

opts.model = fullfile(opts.dataDir, 'models', opts.model);
if(~exist(opts.model))
    error('network does not exist. please download from matconvnet and save it into data/models/');
end

% -------------------------------------------------------------------------
% save the imdb of cub dataset
% -------------------------------------------------------------------------

imdbPath = fullfile(opts.expDir, [opts.dataset '-imdb.mat']);
if exist(imdbPath)
    imdb = load(imdbPath) ;
else
    imdb = get_cub_database(opts.datasetDir);
    save(imdbPath, '-struct', 'imdb') ;
end

% -------------------------------------------------------------------------
% paths to save input feature maps & training results
% -------------------------------------------------------------------------

opts.fmapDir = fullfile(opts.expDir, 'feature_maps');
if(~exist(opts.fmapDir, 'dir'))
    vl_xmkdir(opts.fmapDir);
end

opts.coocDir = fullfile(opts.expDir, 'deep_cooc_models');
if(~exist(opts.coocDir, 'dir'))
    vl_xmkdir(opts.coocDir);
end

% -------------------------------------------------------------------------

opts.useGpu = [1];
opts.keepAspect = true;

end

function setup_matconvnet()

    % setup vlfeat
    run ../vlfeat/toolbox/vl_setup

    % setup matconvnet
    run ../matconvnet/matlab/vl_setupnn
    addpath ../matconvnet/examples/
    
    addpath ../from-bcnn-package/

    clear mex ;

end

function imdb = get_cub_database(cubDir)

% -------------------------------------------------------------------------
% This function is part of the BCNN.
% Copyright (C) 2015 Tsung-Yu Lin, Aruni RoyChowdhury, Subhransu Maji.
% All rights reserved. 
% -------------------------------------------------------------------------

imdb.imageDir = fullfile(cubDir, 'images');

imdb.maskDir = fullfile(cubDir, 'masks'); % doesn't exist
imdb.sets = {'train', 'val', 'test'};

% Class names
[~, classNames] = textread(fullfile(cubDir, 'classes.txt'), '%d %s');
imdb.classes.name = horzcat(classNames(:));

% Image names
[~, imageNames] = textread(fullfile(cubDir, 'images.txt'), '%d %s');
imdb.images.name = imageNames;
imdb.images.id = (1:numel(imdb.images.name));

% Class labels
[~, classLabel] = textread(fullfile(cubDir, 'image_class_labels.txt'), '%d %d');
imdb.images.label = reshape(classLabel, 1, numel(classLabel));

% Bounding boxes
[~,x, y, w, h] = textread(fullfile(cubDir, 'bounding_boxes.txt'), '%d %f %f %f %f');
imdb.images.bounds = round([x y x+w-1 y+h-1]');

% Image sets
[~, imageSet] = textread(fullfile(cubDir, 'train_test_split.txt'), '%d %d');
imdb.images.set = zeros(1,length(imdb.images.id));
imdb.images.set(imageSet == 1) = 1;
imdb.images.set(imageSet == 0) = 2;

% make this compatible with the OS imdb
imdb.meta.classes = imdb.classes.name ;
imdb.meta.inUse = true(1,numel(imdb.meta.classes)) ;
imdb.images.difficult = false(1, numel(imdb.images.id)) ; 

end