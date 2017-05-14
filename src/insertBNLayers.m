function net = insertBNLayers(net)

% return if the network already contains batch norm layers 
for l = 1:numel(net.layers)
    if isa(net.layers(l).block, 'dagnn.BatchNorm')
        return;
    end
end

% loop over the network and insert batch norm layers after 
% convolutions
% here! avan not all convs
layerOrder = net.getLayerExecutionOrder();
for l = 1:6
    if isa(net.layers(l).block, 'dagnn.Conv') && ...
            ~strcmp(net.layers(l).outputs, 'prediction')
        net = addBatchNorm(net, l);
    end
end

net.rebuild()

end

% -------------------------------------------------------------------------
function net = addBatchNorm(net, layerIndex)

% pair inputs and outputs to ensure a valid network
inputs = net.layers(layerIndex).outputs;

% find the number of channels produced by the previous layer
numChannels = net.layers(layerIndex).block.size(4);

outputs = sprintf('xbn%d',layerIndex);

% Define the name and parameters for the new layer
name = sprintf('bn%d', layerIndex);

block = dagnn.BatchNorm();
paramNames = {sprintf('%sm', name) ...
              sprintf('%sb', name) ...
              sprintf('%sx', name) };

% add new layer to the network          
net.addLayer(...
    name, ...
    block, ...
    inputs, ...
    outputs, ...
    paramNames) ;

% set mu (gain parameter)
mIdx = net.getParamIndex(paramNames{1});
net.params(mIdx).value = ones(numChannels, 1, 'single');
net.params(mIdx).learningRate = 2;
net.params(mIdx).weightDecay = 0;

% set beta (bias parameter)
bIdx = net.getParamIndex(paramNames{2});
net.params(bIdx).value = zeros(numChannels, 1, 'single');
net.params(bIdx).learningRate = 1;
net.params(bIdx).weightDecay = 0;

% set moments parameter
xIdx = net.getParamIndex(paramNames{3});
net.params(xIdx).value = zeros(numChannels, 2, 'single');
net.params(xIdx).learningRate = 0.05;
net.params(xIdx).weightDecay = 0;

% modify the next layer to take the new inputs
net.layers(layerIndex + 1).inputs = {outputs};

end