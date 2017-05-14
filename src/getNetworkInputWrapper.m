% -------------------------------------------------------------------------
function fn = getNetworkInputWrapper(opts, useGpu)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getNetworkInput(imdb,batch,opts,useGpu) ;

% -------------------------------------------------------------------------
function inputs = getNetworkInput(imdb, batch, opts, useGpu)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = getImgBatch(images, opts, 'prefetch', nargout == 0);
labels = imdb.images.label(batch) ;

labels = reshape(repmat(labels, 1, 1), 1, size(im{1},4));

if nargout > 0
  if useGpu
      im1 = gpuArray(im{1}) ;
  else
      im1 = im{1};
  end
  inputs = {'input', im1, 'label', labels} ;
end

