% -------------------------------------------------------------------------
% Description:
%     The co-occurrence block for dagnn.
% 
% Citation:
%     Deep Co-Occurrence Feature Learning for Visual Object Recognition
%     Ya-Fang Shih, Yang-Ming Yeh, Yen-Yu Lin, Yi-Chang Lu, Ming-Feng Weng, and Yung-Yu Chuang 
%     IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2017
% -------------------------------------------------------------------------

classdef coocBlock < dagnn.Filter
  properties
    
  end

  properties(Transient)
    shifto
  end

  methods
    function outputs = forward(obj, inputs, params)
      [outputs{1} obj.shifto] = coocLayer(inputs{1});
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = coocLayer(inputs{1}, derOutputs{1}, obj.shifto);
     
      derParams = {} ;
    end
    
     function rfs = getReceptiveFields(obj)
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = coocBlock(varargin)
      obj.load(varargin) ;
    end


  end
end

