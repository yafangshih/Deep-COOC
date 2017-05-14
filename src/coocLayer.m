% -------------------------------------------------------------------------
% Description:
%     The co-occurrence layer
% 
% Citation:
%     Deep Co-Occurrence Feature Learning for Visual Object Recognition
%     Ya-Fang Shih, Yang-Ming Yeh, Yen-Yu Lin, Yi-Chang Lu, Ming-Feng Weng, and Yung-Yu Chuang 
%     IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2017
% -------------------------------------------------------------------------

function [y o] = coocLayer(x, varargin)
 
halfShift = 5;    

shiftRange = 2*halfShift+1;
shiftRange2 = shiftRange*shiftRange;

backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1};
  offset = varargin{2};
end

gpuMode = isa(x, 'gpuArray');

% [height width channels batchsize]
[h, w, ch, bs] = size(x);

xOri = x;

xPad = padarray(x, [halfShift halfShift]);

g = fspecial('gaussian');
for i = 1:ch
    for j = 1:bs
        xPad(:,:,i,j) = conv2(xPad(:,:,i,j), g, 'same');
    end
end

if backMode

    o = [];
    ratio = [];
    if gpuMode
        y = gpuArray(zeros(size(x), 'single'));
    else
        y = zeros(size(x), 'single');
    end
    
    xPadCPU = gather(xPad);
    offsetCPU = gather(offset);
    
    xPad = padarray(xOri, [halfShift halfShift]);
    
    for b=1:bs
    [xHatCPU xTiltCPU] = xHatxTiltGen_mex(xPadCPU, offsetCPU(:,:,b), h, w, ch, bs, shiftRange);
    if gpuMode
         xHat = gpuArray(xHatCPU);
        xTilt = gpuArray(xTiltCPU);
    else
        xHat = xHatCPU;
        xTilt = xTiltCPU;
    end
        dzdy_b = reshape(dzdy(1,1,:,b), [ch, ch]);
        dzdy_b = dzdy_b';
        aTilt = reshape(xTilt(:,:,:,b), [h*w, ch]);
        aHat  = reshape(xHat(:,:,:,b),  [h*w, ch]);
        y(:, :, :, b) = reshape(aHat*dzdy_b'+aTilt*dzdy_b, [h, w, ch]);
    
        for i = 1:ch
            y(:, :, i, b) = conv2(y(:, :, i, b), g, 'same');
        end
    end
    clear xHat xTilt xPad aTilt aHat xOri;
    clear xHatCPU xTiltCPU offsetCPU xPadCPU;
else
    % forward pass
    if gpuMode
        yAll = gpuArray(zeros([1, 1, shiftRange2, ch*ch, bs], 'single'));
        y = gpuArray(zeros([1, 1, ch*ch, bs], 'single'));
    else
        yAll = zeros([1, 1, shiftRange2, ch*ch, bs], 'single');
        y = zeros([1, 1, ch*ch, bs], 'single');
    end
    
    xPadCPU = gather(xPad);
    xShiftCPU = xShiftGen_mex(xPadCPU, h, w, ch, bs, shiftRange);
     if gpuMode
        xShift = gpuArray(xShiftCPU);
     else
         xShift = xShiftCPU;
     end
    
    for b = 1:bs
        a = reshape(x(:,:,:,b), [h*w, ch]);
        aShift = reshape(xShift(:,:,:,:,b), [h*w, ch*shiftRange2]);
        yAll(1,1,:,:,b) = reshape( (a'*aShift)', [shiftRange2 ch*ch]);
        [y(1,1,:,b), o_temp] = max(yAll(1,1,:,:,b), [], 3);   % o indicate offset of each pair
        ratio = squeeze(y(1,1,:,b)) ./ squeeze(yAll(1,1,1+halfShift*(shiftRange+1) ,:,b));
        [o1(1:ch*ch,1), o2(1:ch*ch,1)] = ind2sub([shiftRange shiftRange],squeeze(o_temp(:)));
        
        idx = find(ratio(:)<=1);
        y(1,1,idx,b) = yAll(1,1,1+halfShift*(shiftRange+1),idx,b);
        o1(idx) = halfShift + 1;
        o2(idx) = halfShift + 1;
        o(:,:,b) = [o1 o2];
        
    end
    clear yAll aShift xShift xPad a aShift xOri o1 o2;
    clear xPadCPU xShiftCPU;
end
