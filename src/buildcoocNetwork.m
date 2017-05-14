% -------------------------------------------------------------------------
% Description:
%     To add co-occurrence layers at the last k conv layers.
%     
%     For faster training/testing process, we take only the last k conv
%     layers to build the network.
% 
% Citation:
%     Deep Co-Occurrence Feature Learning for Visual Object Recognition
%     Ya-Fang Shih, Yang-Ming Yeh, Yen-Yu Lin, Yi-Chang Lu, Ming-Feng Weng, and Yung-Yu Chuang 
%     IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2017
% -------------------------------------------------------------------------

function net = buildcoocNetwork(imdb, opts)

tmp = load(opts.model);

if(strcmp(opts.modeltype, 'ResNet-152'))
    input = 'input';
    tmp.layers(481).inputs = {input};
    tmp.layers(483).inputs = {input};
    tmp.vars(1).name = 'input';

    tmp.layers = tmp.layers(481:512);
    tmp.vars = [tmp.vars(1) tmp.vars(482:513)];
    tmp.params = tmp.params(581:620);
    ch = 2048;
else
    warning('this release is for reproducing the result of ResNet152+3 cooc, ');
    warning('to run experiments with other nets, some minor modifications should be made.');
end

net = dagnn.DagNN.loadobj(tmp);

% -------------------------------------------------------------------------

num1x1conv = 128;
dim = num1x1conv*num1x1conv;
k = 3;
if k~=3
    warning('this release is for reproducing the result of ResNet152+3 cooc, ');
    warning('to run experiments with other ks, some minor modifications should be made.');
end

initBias = 0.1;
numClass = length(imdb.classes.name);

% -------------------------------------------------------------------------
% 1*1 conv
% -------------------------------------------------------------------------
net1x1.layers = {};
net1x1.layers{end+1} = struct('type', 'conv', 'name', 'conv1x1', ...
    'weights',  {{init_weight('xavierimproved', 1, 1, ch, num1x1conv, 'single'), ...
                  ones(num1x1conv, 1, 'single') * initBias}}, ...
    'stride', 1, ...
    'pad', 0, ...
    'learningRate', [1 1], ...
    'weightDecay', [0 0]) ;
net1x1 = vl_simplenn_tidy(net1x1) ;

% -------------------------------------------------------------------------
% classifier netc
% -------------------------------------------------------------------------
netc.layers = {};
netc.layers{end+1} = struct('type', 'conv', 'name', 'c1', ...
    'weights', {{init_weight('xavierimproved', 1, 1, dim*k, numClass, 'single'), ...
                ones(numClass, 1, 'single') * initBias}}, ...
    'stride', 1, ...
    'pad', 0, ...
    'learningRate', [1 1], ...
    'weightDecay', [0 0]) ;
netc.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;
netc = vl_simplenn_tidy(netc) ;

% -------------------------------------------------------------------------
nameweight = '_w';
namebias = '_b';
nameout = '_out';
% -------------------------------------------------------------------------

layerout = {'res5ax', 'res5bx', 'res5cx'};
layerid = {'5a', '5b', '5c'};
coocout = cell(1,k);

for i=1:k
    [net, coocout{i}] = addcoocBlock(net, layerout{i}, layerid{i}, net1x1);
end

clear net1x1;
%--------------------------------------------------------------------------

layerName = 'concat_cooc';
output = [layerName nameout];
net.addLayer(layerName, dagnn.Concat('dim', 3, 'inputSizes', dim*k), {coocout{1}  coocout{2}  coocout{3}}, output); 

%--------------------------------------------------------------------------

input = output;

layerName = 'classifier';
param(1).name = [layerName nameweight];
param(1).value = netc.layers{1}.weights{1};
param(2).name = [layerName namebias];
param(2).value = netc.layers{1}.weights{2};

net.addLayer(layerName, dagnn.Conv(), {input}, 'score', {param(1).name param(2).name});

for f = 1:2,
    varId = net.getParamIndex(param(f).name);
    if strcmp(net.device, 'gpu')
        net.params(varId).value = gpuArray(param(f).value);
    else
        net.params(varId).value = param(f).value;
    end
    net.params(varId).learningRate = 1;
    net.params(varId).weightDecay = 1;
end

% add loss functions
net.addLayer('loss', dagnn.Loss('loss', 'softmaxlog'), {'score','label'}, 'objective');
net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'score','label'}, 'top1error');
net.addLayer('top5e', dagnn.Loss('loss', 'topkerror'), {'score','label'}, 'top5error');

clear netc;

net.mode = 'normal';

end

% -------------------------------------------------------------------------
% add cooc - sqrt - l2norm layers
% -------------------------------------------------------------------------
function [net, output] = addcoocBlock(net, input, layerid, net1x1)

nameconv1x1 = 'conv1x1_';
namecooclayer = 'cooc_';
namesqrt = 'sqrt_';
namel2norm = 'l2norm_';

nameweight = '_w';
namebias = '_b';
nameout = '_out';

% -------------------------------------------------------------------------

layerName = [nameconv1x1 layerid];
param(1).name = [layerName nameweight];
param(1).value = net1x1.layers{1}.weights{1};
param(2).name =  [layerName namebias];
param(2).value = net1x1.layers{1}.weights{2};
output = [layerName nameout];

net.addLayer(layerName, dagnn.Conv(), input, output, {param(1).name param(2).name});

for f = 1:2,
    varId = net.getParamIndex(param(f).name);
    if strcmp(net.device, 'gpu')
        net.params(varId).value = gpuArray(param(f).value);
    else
        net.params(varId).value = param(f).value;
    end
    net.params(varId).learningRate = 1;
    net.params(varId).weightDecay = 1;
end

% -------------------------------------------------------------------------
input = output;

layerName = [namecooclayer layerid];
output = [layerName nameout];
net.addLayer(layerName, coocBlock(), input, output);

% -------------------------------------------------------------------------
layerName = [namesqrt layerid];
input = output;
output = [layerName nameout];
net.addLayer(layerName, SquareRoot(), {input}, output);

% -------------------------------------------------------------------------
layerName = [namel2norm layerid];
input = output;
output = [layerName nameout];
net.addLayer(layerName, L2Norm(), {input}, output);

end

% -------------------------------------------------------------------------
% Different ways to initialize the weights of the 1x1 convs and the last fc
% layer.
% This function is written by MatConvNet.
% -------------------------------------------------------------------------
function weights = init_weight(initWmethod, h, w, in, out, type)

switch lower(initWmethod)
  case 'gaussian'
    sc = 0.01/opts.scale ;
    weights = randn(h, w, in, out, type)*sc;
  case 'xavier'
    sc = sqrt(3/(h*w*in)) ;
    weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
  case 'xavierimproved'
    sc = sqrt(2/(h*w*out)) ;
    weights = randn(h, w, in, out, type)*sc ;
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end

end