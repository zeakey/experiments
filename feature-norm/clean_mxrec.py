import mxnet as mx
from mxnet import recordio
from PIL import Image

record = mx.recordio.MXIndexedRecordIO('/media/ssd0/faces_emore/train.idx', '/media/ssd0/faces_emore/train.rec', 'r')

new_record = mx.recordio.MXIndexedRecordIO('/media/ssd0/faces_emore-clean/train.idx', '/media/ssd0/faces_emore-clean/train.rec', 'w')

s = record.read_idx(0)
header, _ = recordio.unpack(s)
print(header.label)
N = int(header.label[0])

def check_valid_image(data):
    """Checks if the input data is valid"""
    if len(data[0].shape) == 0:
        raise RuntimeError('Data shape is wrong')

for i in range(1, 2):
    s = record.read_idx(i)
    header, img = recordio.unpack(s)
    label = header.label
    img = mx.image.imdecode(img).asnumpy()
    #img = Image.fromarray(img)
    print("label", label)
    p = recordio.pack_img(header, img)
    new_record.write_idx(i-1, p)

    try:
        check_valid_image(img)
    except RuntimeError as e:
        print('Invalid image, skipping:  %s', str(e))
        continue
    print("%d/%d"%(i, N), img.shape)

new_record.close()
