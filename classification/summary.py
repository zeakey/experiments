import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


DEPTH = 8

def bit2mb(x):
    """
    bit to Mega Byte
    x / 8: bit to byte
    x / 1024**2: byte to MB
    """
    return x / 8 / 1024**2

def summary(model, input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
                summary[m_key]["kernel_shape"] = list(module.weight.shape)
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("-------------------------------------------------------------------------------------------------------------------------")
    line_new = "{:>20} {:>25} {:>25} {:>15} {:>15} {:>15} {:>15}".format("Layer (type)", "Kernel Shape", "Output Shape", "Mem Access", "Param #", "Cached (MB)", "Calculations")
    print(line_new)
    print("=========================================================================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    total_flops = 0
    total_cache_mem = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        total_output += np.prod(summary[layer]["output_shape"])

        if summary[layer]["nb_params"] > 0:
            kernel_shape = summary[layer]["kernel_shape"]
        else:
            kernel_shape = None

        flops = 0
        if kernel_shape and len(kernel_shape) == 4:
            H, W = summary[layer]["output_shape"][2:]
            flops = int(np.prod(summary[layer]["kernel_shape"]) * H * W)
            mem_access = np.prod(summary[layer]["input_shape"])*summary[layer]["kernel_shape"][2]*summary[layer]["kernel_shape"][3]
        else:
            mem_access = 0

        cache_mem = np.prod(summary[layer]["input_shape"]) + np.prod(summary[layer]["output_shape"]) + mem_access
        total_cache_mem += cache_mem

        line_new = "{:>20} {:>25} {:>25} {:>15} {:>15} {:>15} {:>15}".format(
            layer,
            str(kernel_shape),
            str(summary[layer]["output_shape"]),
            "{:,}".format(mem_access),
            "{0:,}".format(summary[layer]["nb_params"]),
            "{:.3f}".format(bit2mb(cache_mem * DEPTH)),
            "{0:,}".format(flops)
        )
        total_params += summary[layer]["nb_params"]
        total_flops += flops

        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = bit2mb(np.prod(input_size * DEPTH) * batch_size)
    total_output_size = bit2mb(total_output * DEPTH)   # x2 for gradients
    total_params_size = bit2mb(total_params.numpy() * DEPTH)
    total_size = total_params_size + total_output_size + total_input_size

    print("=========================================================================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("-------------------------------------------------------------------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward pass size (MB) (Total Outputs): %0.2f " % (total_output_size / 2))
    print("Forward/backward pass size (MB) (Total Outputs and Gradients): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("-------------------------------------------------------------------------------------------------------------------------")
    print("Testing Memory Access (MB) (Input + ALL Outputs + Params): %0.2f" % (total_input_size + total_output_size / 2 + total_params_size))
    print("Training Memory Access (MB) (Input + ALL Outputs/Gradients + Params): %0.2f" % (total_input_size + total_output_size + total_params_size))
    print("Calculations: %d" % total_flops)
    # print("Estimated Total Size (MB): %0.2f" % total_size)
    print("---------------------------------------------In Excel Format-------------------------------------------------------------")
    print("Data Depth: %d" % DEPTH)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Cache memory (MB): %0.2f " % (bit2mb(total_cache_mem * DEPTH)))
    print("Calculations: {:,}".format(total_flops))
    print("-------------------------------------------------------------------------------------------------------------------------")
    # return summary

if __name__ == "__main__":
    from models import msnet, msnet1, msnet2
    from torchvision.models import resnet
    model = msnet.msnet50()
    # model = resnet.resnet50()
    summary(model, batch_size=1, input_size=(3, 224, 224), device="cpu")