# RestoreNet
Testing codes for "One-Shot Image Classification by Learning to Restore Prototypes"

The repository contains the essential codes for RestoreNet and hope it will help to understand the idea.

1. You can do your own training, or alternatively, you may like to download the network parameters and pre-computed features at https://www.dropbox.com/sh/6jy0g8nfc97bvrm/AACORpPowNVnFXdwek7vUYjIa?dl=0

   In the link, there are three files: 1) FeatureExtractor.pth 2) Transformer.pth 3) GalleryPool.

   The first two .pth files are network parameters. Please put them in the folder ./Network Params/

   The GalleryPool file is pre-computed images features used for self-training. We use it to avoid unnecessary computation. Please put it in ./

2. Put all images in miniImageNet dataset in the folder /miniImageNet/images.

3. Do evaluation by run 'python run_test.py' 

