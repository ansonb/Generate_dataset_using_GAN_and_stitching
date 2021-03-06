# Generate_dataset_using_GAN_and_stitching
Generate a dataset of cracked phones using GANs (to generate the cracks) and stitching the cracks on normal phone images.

# Requirements
Pytorch </br>
Scipy </br>

# Training/Generating
To train the GAN on custom data run the below command </br>
`python dc_gan_pytorch.py --data_train_path='path/to/training/images' --model_save_path='path/to/save/model'`

To generate the images run the below command </br>
`python generate_images_dc_gan.py --data_save_path='path/to/training/images' --model_save_path='path/to/save/model'`

To stitch the generated patches on images run </br>
`python stitch_images.py --out_dir='path/to/save/images' --input_image_dir='path/to/input/images' --input_patch_dir='path/to/input/patches' --out_csv_file_path='path/to/save/labels'`
