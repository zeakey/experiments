function [weight, ar, slr] = optimalSKfuc2(SK, SKareaIm, ShapeIm, orSKlen, beta, avgendpathlen)

ShapeArea = size(find(ShapeIm > 0), 1);
%sigma = sqrt(ShapeArea) * 0.1;
SKareaIm_t = SKareaIm & ShapeIm;
ResArea = size(find(SKareaIm_t > 0), 1);
ar = max(0, (ShapeArea - ResArea) / ShapeArea);
SKlen = size(find(SK > 0), 1);
alpha = beta*log(orSKlen / avgendpathlen);
%SKlen = DTlen(SK, SKDist);
slr = log(SKlen/avgendpathlen+1);
weight = alpha*ar + slr;
% weight = SKlen/avgendpathlen;