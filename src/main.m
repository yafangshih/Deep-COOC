function main()

opts.expDir = '../exp';
opts.dataDir = '../data';
opts.dataset = 'cub';

opts.model = 'imagenet-resnet-152-dag.mat';
opts.modeltype = 'ResNet-152';
opts.fmap = 'Res152fmap_res4b35_relu';

[opts, imdb] = setup(opts);

% -------------------------------------------------------------------------
% save feature maps to disk if not yet done
% -------------------------------------------------------------------------

opts.fmapDir = fullfile(opts.fmapDir, opts.fmap);
fmapdb = setup_fmap(imdb, opts);

% -------------------------------------------------------------------------
% Build a cnn network with co-occurrence layers
% -------------------------------------------------------------------------

net = buildcoocNetwork(imdb, opts);

% -------------------------------------------------------------------------
% train the 1x1 convs and the last fc layer
% -------------------------------------------------------------------------

for i=1:40
    net.params(i).learningRate = 0;
    net.params(i).weightDecay = 0;
end

opts.train.weightDecay = 0 ;
opts.train.momentum = 0.9;
opts.train.gpus = opts.useGpu(1) ;
opts.train.continue = true ;

opts.train.batchSize = 8 ;
opts.train.numEpochs = 200 ; 
opts.train.learningRate =  0.1 ; 
opts.train.expDir = fullfile(opts.coocDir, 'res152_128_concat3_bs8') ;

[net, info] = cnn_train_dag(net, fmapdb, @load_fmap_fromdisk, opts.train) ;

% -------------------------------------------------------------------------
% fine-tune the layers after the kth to the last conv layer
% -------------------------------------------------------------------------

tmp = load(fullfile(opts.train.expDir, ['net-epoch-', sprintf('%d', opts.train.numEpochs), '.mat']));
tmp = tmp.net;

for i=1:40
    tmp.params(i).learningRate = 1;
    tmp.params(i).weightDecay = 1;
end  

arraylr = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40];
for i=1:length(arraylr)
    tmp.params(arraylr(i)).learningRate = 0.1;
end
  
net = dagnn.DagNN.loadobj(tmp);
clear tmp;

opts.train.batchSize = 8 ;
opts.train.numEpochs = 200 ; 
opts.train.learningRate =  0.1 ; 
opts.train.expDir = fullfile(opts.coocDir, 'res152_128_last3_bs8') ;

[net, info] = cnn_train_dag(net, fmapdb, @load_fmap_fromdisk, opts.train) ;

end

% -------------------------------------------------------------------------
% get feature map inputs from disk
% -------------------------------------------------------------------------
function inputs = load_fmap_fromdisk(imdb, batch)

    useGpu = 1;

    for i=1:numel(batch)
        load(fullfile(imdb.imageDir, imdb.images.name{batch(i)}));
        im(:,:,:,i) = code;
    end

    if useGpu
        im1 = gpuArray(im) ;
    else
        im1 = im;
    end

    labels = imdb.images.label(batch) ;
    inputs = {'input', im1, 'label', labels} ;
    
end

