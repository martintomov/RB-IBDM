# Generating object masks with SAM for RB-IBDM on a test dataset
### [Segment-Anything Issue #720](https://github.com/facebookresearch/segment-anything/issues/720)

<a target="_blank" href="https://colab.research.google.com/github/martintmv-git/RB-IBDM/blob/main/Experiments/Generating%20Masks%20with%20SAM/Success%20with%20test%20dataset/sam_test_dataset_generate_masks.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

After exploring various approaches, I've discovered a method that effectively addresses the issue I was encountering with generating masks for a large dataset in Colab using SAM.

The solution involves processing the images in batches, significantly improving efficiency when dealing with thousands of images.

> NOTE: I noticed that for my use case, the first mask was always the best in quality, and as a result, I save only the first mask generated for each image to conserve storage space and streamline the process.

```
import cv2
from PIL import Image
import numpy as np

def save_masks_to_drive(masks, save_path, image_name):
    if masks:  # Check if there is at least one mask
        try:
            img = Image.fromarray((masks[0] * 255).astype(np.uint8))  # Use only the first mask
            mask_file_path = os.path.join(save_path, f'mask_{image_name}_0.png')  # Name for the first mask
            img.save(mask_file_path)
            print(f"Successfully saved mask to drive: {mask_file_path}")
        except Exception as e:
            print(f"Failed saving mask to drive for {image_name}: {e}")

def process_images_in_batches(dataset_path, save_path, batch_size=100):
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        for path in batch_paths:
            try:
                print(f"Processing: {path}")
                image_bgr = cv2.imread(path)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                sam_result = mask_generator.generate(image_rgb)
                masks = [mask['segmentation'] for mask in sorted(sam_result, key=lambda x: x['area'], reverse=True)]
                
                # Saving only the first mask as it's the most valuable for later training
                save_masks_to_drive(masks, save_path, os.path.basename(path).replace('.jpg', '').replace('.png', ''))
                print(f"Successfully processed and saved masks for: {path}")
            except Exception as e:
                print(f"Error processing {path}: {e}")

# Process the images in batches
process_images_in_batches(dataset_path, save_path, batch_size=100)
```
### Mask Generation Results

The method described above has been successful in generating masks for a test dataset of 273 images and will be used to generate masks for the entire dataset of ~27,000 images.

<a href="" rel="noopener">
<img src="https://i.imgur.com/9WOXu3W.png alt="mask_result_1">
</a>
