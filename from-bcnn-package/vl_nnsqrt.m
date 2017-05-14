function y = vl_nnsqrt(x, param, varargin)
% VL_NNSQRT perform square root normalization for the input features
% at each location
%
% Author: Subhransu Maji, Aruni RoyChowdhury, Tsung-Yu Lin

%
% This file is part of the BCNN and is made available under
% the terms of the BSD license (see the COPYING file).

thresh = param(1);

backMode = numel(varargin) > 0 && ~isstr(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
end

if backMode
    y = 0.5./sqrt(abs(x)+thresh);
    y = y.*dzdy;
else
    y = sign(x).*sqrt(abs(x));
end
