import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
try:
    import nvidia.dali.plugin.pytorch as plugin_pytorch
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

# DALI data loader
class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, num_gpus, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.MXNetReader(path = [data_dir+"/webface.rec"], 
        index_path=[data_dir+"/webface.idx"], random_shuffle = True, shard_id = device_id, num_shards = num_gpus)
        
        #self.input = ops.FileReader(file_root=data_dir, shard_id=0, num_shards=4, random_shuffle=True)
        #let user decide which pipeline works him bets for RN version he runs
        
        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.HostDecoder(device=dali_device, output_type=types.RGB)
        else:
            dali_device = "gpu"
            # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
            # without additional reallocations
            self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)

        self.cmn = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(112, 96),
                                            image_type=types.RGB,
                                            mean=[0.485*255, 0.456*255, 0.406*255],
                                            std=[0.229*255, 0.224*255, 0.225*255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        output = self.cmn(images.gpu(), mirror=rng)
        return [output, self.labels]

if __name__ == "__main__":
    NUM_GPUS = 1
    NUM_THREADS = 2

    pipes = [HybridTrainPipe(batch_size=int(128/NUM_GPUS), num_threads=NUM_THREADS, device_id=device_id, data_dir="/media/ssd1/CASIA-WebFace-112X96-rec" , crop=224, num_gpus=NUM_GPUS, dali_cpu=False) for device_id in range(NUM_GPUS)]
    pipes[0].build()
    train_loader = plugin_pytorch.DALIClassificationIterator(pipes, size=int(pipes[0].epoch_size("Reader")))

    for i, data in enumerate(train_loader):
        print(i)
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()
        torchvision.utils.save_image(input.cpu(), "DALI-%d.jpg" % i)
        if i == 4:
            break


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        "/media/ssd1/CASIA-WebFace-112X96",
        transforms.Compose([
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True,
        num_workers=8, pin_memory=True)

    for i, data in enumerate(train_loader):
        input = data[0]
        target = data[1]
        torchvision.utils.save_image(input.cpu(), "PTH-%d.jpg" % i)
        if i == 4:
            break
