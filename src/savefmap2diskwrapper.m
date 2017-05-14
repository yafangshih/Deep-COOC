function savefmap2diskwrapper(imdb, opts)

    tmp = load(opts.model);
    if(strcmp(opts.modeltype, 'ResNet-152'))
        varId = 481;
        varName = 'res4b35x';
        
        input = 'input';
        tmp.layers(1).inputs = {input};
        tmp.vars(1).name = input;
        tmp.vars(varId).precious = 1;
    else
        warning('this release is for reproducing the result of ResNet152+3 cooc, ');
        warning('to run experiments with other nets, some minor modifications should be made.');
    end

    net = dagnn.DagNN.loadobj(tmp);
    
    savefmap2disk(net, opts, imdb, varName);

end

function savefmap2disk(net, opts, imdb, varName)

    fmapopts = net.meta.normalization;
    fmapopts.numThreads = 12 ;
    fmapopts.rgbVariance = [];
    fmapopts.scale = 2;

    alldata = find(ismember(imdb.images.set, [1 2]));

    batchSize = 64;
    useGpu = opts.useGpu > 0 ;
    if useGpu
        net.move('gpu') ;
    end

    getBatchFn = getNetworkInputWrapper(fmapopts, useGpu) ;

    for t=1:batchSize:numel(alldata)
        fprintf('Initialization: extracting feature maps of batch %d/%d\n', ceil(t/batchSize), ceil(numel(alldata)/batchSize));
        
        batch = alldata(t:min(numel(alldata), t+batchSize-1));
        input = getBatchFn(imdb, batch) ;

        input = input(1:2);
        net.mode = 'test' ;
        net.eval(input);
        fIdx = net.getVarIndex(varName); 
        code_b = net.vars(fIdx).value;
        code_b = squeeze(gather(code_b));

        for i=1:numel(batch)
            code = code_b(:,:,:,i);
            save(fullfile(opts.fmapDir, ['fmap_', num2str(batch(i), '%05d'), '.mat']), 'code', '-v7.3');
        end
    end

end