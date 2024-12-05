
# Video Frame Extractor for YOLO Preparation

This Python script extracts frames from a video file and allows you to crop them into 640x640 pixel images. It is especially useful for preparing data for training YOLO (You Only Look Once) models.

## Features

- Load a video file and extract frames.
- Skip a specified number of frames.
- Display frames with an overlay grid.
- Allow user interaction to adjust the position of the cropping rectangle.
- Save cropped images to a specified folder.

## Requirements

Make sure you have the following dependencies installed:

- **Python 3.10+**: [Download Python](https://www.python.org/downloads/)
- **OpenCV**: You can install it via pip:
```bash
pip install opencv-python
```
- **tqdm**: You can install it via pip:
```bash
pip install tqdm
```
- **opencv**: You can install it via pip:
```bash
pip install opencv-python
```

## Usage

1. **Run the script** by executing the following command in your terminal:

```bash
python video_frame_extractor.py
```

2. **Provide the video path**: When prompted, enter the full path to the video file you want to process.

3. **Specify the save folder**: Enter the full path to the folder where the cropped images should be saved.

4. **Set the number of frames to skip**: Provide the number of frames to skip during extraction. (e.g., if you want to analyze every second frame, you can set it to 29 for a 30FPS video).

5. **Interact with the frames**:
   - Use the `W`, `A`, `S`, `D` keys to move the cropping rectangle:
     - `W`: Move up
     - `A`: Move left
     - `S`: Move down
     - `D`: Move right

   - Press the `K` key to save the current cropped image.
   - Press the spacebar to go to the next frame.
   - Press `Q` to quit the application.

## Code Structure

- **Position Class**: Handles the positioning and dimensions of the cropping rectangle.
- **get_video_path()**: Prompts the user to enter a valid video file path.
- **get_folder_to_save()**: Prompts the user to enter a valid folder path for saving cropped images.
- **get_count_to_skip()**: Prompts the user for the number of frames to skip before processing.
- **draw_grid()**: Draws a grid overlay on the current frame.

## Example

After running the script and providing the necessary inputs, the program will display the video frames one by one. You can crop the frames interactively, and the cropped images will be saved in the specified folder.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and create a pull request.

## License
```
This project is licensed under the MIT License.
```