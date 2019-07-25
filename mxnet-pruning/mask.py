import torch
import mxnet as mx
import numpy as np

class Mask(object):
    def __init__(self, parameters, rate, context, metric="norm", debug=True):
        """
        parameters: dict containing parameters
        rate: pruning rate
        metric: norm | mulcorr
        """
        self.mask = {}
        self.params = parameters
        self.rate = rate
        self.context = context
        self.metric = metric
        self.debug = debug
        self.pruned_indices = {}

        for name, p in self.params.items():
            self.mask[name] = mx.nd.ones(p.data(ctx=context[0]).shape[0], ctx=context[0])
    
    def update_mask(self):
        for idx, (name, p) in enumerate(self.params.items()):
            print("Updating mask %d of %d, kernel shape: %s" % (idx, len(self.params), str(p.shape)))
            pdata = p.data(self.context[0])
            pgrad = p.grad(self.context[0])
            N, C, H, W = pdata.shape
            pdata = pdata.reshape(pdata.shape[0], -1)
            pgrad = pdata.reshape(pgrad.shape[0], -1)
            D = C*H*W
            num_pruned = int(N * self.rate)
            norm = mx.nd.norm(pdata, ord=2, axis=1)
            metric = np.zeros((N,), dtype=np.float32)

            if N > D or self.metric == "norm":
                metric = norm.asnumpy()
            
            elif self.metric == "mulcorr":
                npdata = pdata.asnumpy()
                rank = np.linalg.matrix_rank(npdata)
                useless_idx = []
                useful_idx = []
                for i in range(N):
                    idx1 = [True]*N
                    idx1[i] = False
                    npdata1 = npdata[idx1, :]
                    rank1 = np.linalg.matrix_rank(npdata1)
                    if rank1 == rank:
                        useless_idx.append(i)
                    else:
                        useful_idx.append(i)
                assert np.linalg.matrix_rank(npdata[useful_idx]) == len(useful_idx)
                mulcorr = multiple_correlation_torch(npdata[useful_idx, :])
                assert not np.any(np.isnan(mulcorr))
                metric[useful_idx] = 1 - mulcorr

            indices_to_be_pruned = np.argsort(metric)[:num_pruned]
            self.pruned_indices[name] = indices_to_be_pruned.astype(int).tolist()

            self.mask[name] = mx.nd.ones_like(self.mask[name])
            if len(indices_to_be_pruned) > 0:
                self.mask[name][indices_to_be_pruned] = 0

    def prune_param(self):
        for name, p in self.params.items():
            pdata = self.params[name].data(self.context[0])
            N, C, H, W = pdata.shape

            indices_to_be_pruned = np.where(self.mask[name].asnumpy()==0)[0].tolist()
            # indices_to_be_preserved = np.where(self.mask[name].asnumpy()==1)[0].tolist()
            num_pruned = len(indices_to_be_pruned)
            if num_pruned > 0:
                pdata[indices_to_be_pruned, :, :, :] = 0
                p.set_data(pdata)

    def prune_grad(self):
        for name, p in self.params.items():
            indices_to_be_pruned = np.where(self.mask[name].asnumpy()==0)[0].tolist()
            num_pruned = len(indices_to_be_pruned)
            if num_pruned > 0:
                for ctx in self.context:
                    pgrad = self.params[name].grad(ctx=ctx)
                    pgrad[indices_to_be_pruned, :, :, :] = 0

def multiple_correlation(weight):
    N = weight.shape[0]
    weight = weight.reshape(N, -1)

    weight -= weight.mean()
    norm = np.linalg.norm(weight, ord=2, axis=1).reshape(N, 1)
    weight /= norm

    corr = np.matmul(weight, weight.transpose())
    mulcorr = np.zeros(N, dtype=weight.dtype)

    for i in range(N):
        tmp1 = corr[np.arange(N)!=i, :][:, np.arange(N)!=i]
        tmp2 = corr[np.arange(N)!=i, i]

        if np.linalg.det(tmp1) == 0:
            return None

        tmp3 = np.linalg.inv(tmp1)
        # tmp3 = np.linalg.pinv(tmp1)

        mulcorr[i] = np.sqrt(
                np.einsum("i,ij,j->", tmp2, tmp3, tmp2) / corr[i, i]
                )
    return mulcorr

def multiple_correlation_torch(weight):
    weight = torch.from_numpy(weight).cuda()
    N = weight.size(0)
    weight = weight.reshape(N, -1)
    weight -= torch.mean(weight, dim=1, keepdim=True)
    weight /= torch.norm(weight, p=2, dim=1, keepdim=True)
    corr = torch.matmul(weight, weight.transpose(0, 1))

    mul_corr = torch.zeros(N, dtype=weight.dtype, device=weight.device)
    for i in range(N):
        tmp1 = corr[torch.arange(N) != i, :][:, torch.arange(N) != i]
        tmp2 = corr[torch.arange(N) != i, i]
        mul_corr[i] = torch.sqrt(
                torch.einsum("i,ij,j->", (tmp2, torch.inverse(tmp1), tmp2)) / corr[i, i]
                )
    return mul_corr.cpu().numpy()

def multiple_correlation_mx(weight):
    N = weight.shape[0]
    weight = weight.reshape(N, -1)

    weight -= weight.mean()
    norm = mx.nd.norm(weight, ord=2, axis=1).reshape(N, 1)
    weight /= norm

    corr = mx.nd.linalg.gemm2(weight, weight.transpose())
    mulcorr = mx.nd.zeros(N, dtype=weight.dtype, ctx=weight.context)

    for i in range(N):
        idx = list(range(N))
        idx.remove(i)
        tmp1 = corr[idx, :][:, idx]
        tmp2 = corr[idx, i]

        tmp3 = np.linalg.inv(tmp1)

        mulcorr[i] = mx.nd.linalg.gemm2(
                               mx.nd.linalg.gemm2(tmp2.reshape(1, -1), tmp3),
                               tmp2.reshape(-1, 3))
    return mulcorr


if __name__ == "__main__":
    weight = mx.nd.random.randn(10, 2)
    print(multiple_correlation_mx(weight))