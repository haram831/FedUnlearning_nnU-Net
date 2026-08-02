Params

testdata 기준 확인 결과:

Dataset301_Decathlon / 3d_fullres
Architecture: PlainConvUNet
Patch size: (24, 256, 256)
Input channels: 1
Output channels: 3
Parameters: 44,578,482

----------------------------------------------------------------------------
FLOPs

Trainer: nnUNetTrainer
Configuration: 3d_fullres
Architecture: dynamic_network_architectures.architectures.unet.PlainConvUNet
Input shape: (1, 1, 24, 256, 256)
Deep supervision: False
FLOPs: 456.278 GFLOPs
FLOPs raw: 456277635072 (456.278G)
Parameters: 44578482 (44.578M)
Trainable parameters: 44578482 (44.578M)

| name                  | #elements or shape   |
|:----------------------|:---------------------|
| model                 | 44.6M                |
| encoder               | 19.4M                |
| encoder.stages        | 19.4M                |
| encoder.stages.0      | 9.7K                 |
| encoder.stages.1      | 55.7K                |
| encoder.stages.2      | 0.7M                 |
| encoder.stages.3      | 2.7M                 |
| encoder.stages.4      | 5.0M                 |
| encoder.stages.5      | 5.5M                 |
| encoder.stages.6      | 5.5M                 |
| decoder               | 25.2M                |
| decoder.stages        | 23.4M                |
| decoder.stages.0      | 8.3M                 |
| decoder.stages.1      | 8.3M                 |
| decoder.stages.2      | 5.3M                 |
| decoder.stages.3      | 1.3M                 |
| decoder.stages.4      | 0.1M                 |
| decoder.stages.5      | 27.8K                |
| decoder.transpconvs   | 1.8M                 |
| decoder.transpconvs.0 | 0.4M                 |
| decoder.transpconvs.1 | 0.4M                 |
| decoder.transpconvs.2 | 0.7M                 |
| decoder.transpconvs.3 | 0.3M                 |
| decoder.transpconvs.4 | 32.8K                |
| decoder.transpconvs.5 | 8.2K                 |
| decoder.seg_layers    | 3.4K                 |
| decoder.seg_layers.0  | 1.0K                 |
| decoder.seg_layers.1  | 1.0K                 |
| decoder.seg_layers.2  | 0.8K                 |
| decoder.seg_layers.3  | 0.4K                 |
| decoder.seg_layers.4  | 0.2K                 |
| decoder.seg_layers.5  | 99                   |