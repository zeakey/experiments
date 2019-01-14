* `MSNet1`: 

* `MSNet2`: Dismantle resnet's blocks of the same group into two parallel streams: the `high_block` and the `low_block`. BOTH high and low block are worked in **the same resolution as resnet**.

* `MSNet3`: Based on `MSNet2`, the low_block in `MSNet3` runs in 2 times lower resolution compared to the `high_block`, and the feature maps of `low_block` will be upsampled at the end to match the size of `high_block`. Low-resolution feature maps and High-resolution feature maps will be fused through summation at the end.

* `MSNet4`: Based on `MSNet3`, add a `merge block`.