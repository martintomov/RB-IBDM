Before running the action from within Photoshop you need to copy the `DIOPSIS-Flower-Screen.jpg` (or any other high-resolution image that may have been chosen as a new background) file in your clipboard (select image and ctrl + C or command + C or right-click and press `Copy`).

From within Photoshop:
Running a batch from `File -> Automate -> Batch...` is advisable:
 - Set: InsectBackground
 - Action: Insect_Background_Replacement
 - Select input and output folders.
 - Tick the `Override Action "Save As" Commands` option.
 - Press `OK`.

If Photoshop keeps images in memory after completing a batch, which quickly fills up the free hard drive space, we advise dividing the images equally between multiple folders and performing a batch on each one with a restart of Photoshop in between batches when less than enough free space is available.

Newly generated images will be of bigger size as all images are saved at maximum possible JPEG quality to ensure minimal quality loss.