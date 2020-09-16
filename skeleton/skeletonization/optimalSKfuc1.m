function ProS = optimalSKfuc1(SK, SKareaIm, ShapeIm, orSKlen, beta, avgendpathlen)

ShapeArea = size(find(ShapeIm > 0), 1);

 SKareaIm_t = SKareaIm & ShapeIm;
ResArea = size(find(SKareaIm_t > 0), 1);
% ResCl = size(find(bwperim(SKareaIm_t)>0),1);
% Cl = size(find(bwperim(ShapeIm)>0),1);
alpha = (beta*log(orSKlen / avgendpathlen));
 ar = exp(-alpha*max(0, (ShapeArea - ResArea) / ShapeArea));
% ar = exp(-max(0, (Cl - ResCl) / Cl));
SKlen = size(find(SK > 0), 1);
 
% alpha = 2;
 slr = exp(-(log(SKlen / avgendpathlen) + 1));
% slr = exp(-alpha* SKlen / orSKlen);
ProS = ar * slr;