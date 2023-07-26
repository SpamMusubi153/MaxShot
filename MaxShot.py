# MaxShot.py

# Max Tran
# A Python script to automatically screenshot an area of your screen when it changes.
# Originally intended to automatically capture presentation slides from a Zoom meeting.

from dataclasses import dataclass

import os
import time

import numpy as np
import cv2 as cv
from skimage.metrics import structural_similarity as ssim

from pyautogui import screenshot


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
#  ―――――――――――――――――――――――
# | Environment Variables |
#  ―――――――――――――――――――――――
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

# Images are stored by default in:
# ...\The Program Directory\Automatically Saved Images
# This path can be changed by changing the path below:
IMAGE_STORAGE_PATH = "Automatically Saved Images"

# By default, screenshots are taken and processed once every second.
# To change this behavior, adjust the CAPTURE_DELAY_TIME, in seconds, below.
CAPTURE_DELAY_TIME = 1

# Display additional messages for debugging.
DISPLAY_DEBUG_MESSAGES = False


# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
#  ―――――――――――
# | Functions |
#  ―――――――――――
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

# Create a dataclass to store data related to a cropping operation.
@dataclass
class crop_result_storage:

    # Store the state of the cropping operation.
    started_cropping: bool = False
    crop_complete: bool = False

    # Store the starting and ending x and y coordinates.
    x_start: int = None
    y_start: int = None
    x_end: int = None
    y_end: int = None

    # Store an original copy of the displayed image.
    img = None

# Create a global instance of the dataclass to be accessible by the "crop_with_mouse" function.
crop_result = crop_result_storage()
CROP_WINDOW_NAME = "Select an area to capture"

# A callback function to handle cropping through an OpenCV window.
def crop_with_mouse(event, x, y, flags, param):
    # Use the global instance of the crop_result dataclass to save data.
    global crop_result

    # Return early if x or y are None due to an out-of-bounds click or other errors.
    if (x is None) or (y is None):
        return

    # Create a rectangle from starting and ending x and y coordinates.
    def create_rectangle(x_start, y_start, x_end, y_end):
        x_diff = abs(x_end - x_start)
        y_diff = abs(y_end - y_start)

        rectangle = np.zeros(shape=(y_diff, x_diff, 3), dtype=np.uint8)

        return rectangle
    
    # Append and display a rectangle based on the user's selected crop zone.
    # Due to limitations in OpenCV, this function does not use clean encapsulation practices.
    def append_rectangle_to_image():
        global crop_result

        rectangle = create_rectangle(crop_result.x_start, crop_result.y_start, crop_result.x_end, crop_result.y_end)

        # Extract the portion of the original image selected by the user. Note how the slicing in the y direction occurs before slicing in the x direction.
        original_img_portion = crop_result.img[crop_result.y_start:crop_result.y_end, crop_result.x_start:crop_result.x_end]

        # Make sure the rectangle is the same size as the portion cropped from the original image.
        assert (rectangle.shape == original_img_portion.shape), f"The rectangle mask shape of {rectangle.shape} does not match the shape of the portion of the original image {original_img_portion.shape}."

        # Combine the portion of the original image with the selection rectangle.
        new_portion = cv.addWeighted(original_img_portion, 0.5, rectangle, 0.5, 1.0)
        
        # Add the altered portion back to a copy of the original image.
        new_img = crop_result.img
        new_img[crop_result.y_start:crop_result.y_end, crop_result.x_start:crop_result.x_end] = new_portion

        # Display the new image.
        cv.imshow(CROP_WINDOW_NAME, new_img)

    # When the user begins cropping, save the starting x and y coordinates.
    if event == cv.EVENT_LBUTTONDOWN:
        crop_result.x_start = x
        crop_result.y_start = y
        crop_result.x_end = x
        crop_result.y_end = y

        crop_result.started_cropping = True

    # When the user moves the mouse, update the selection rectangle.
    if (crop_result.started_cropping) and (event == cv.EVENT_MOUSEMOVE):
        crop_result.x_end = x
        crop_result.y_end = y

        append_rectangle_to_image()

    # When the user releases the mouse, save the ending x and y coordinates and signal that the cropping operation was completed successfully.
    if (crop_result.started_cropping) and (event == cv.EVENT_LBUTTONUP):
        crop_result.x_end = x
        crop_result.y_end = y

        append_rectangle_to_image()

        crop_result.crop_complete = True

        if DISPLAY_DEBUG_MESSAGES:
            print(f"The crop size has been set with the following parameters:\n\ty:\n\t\ty_start:{crop_result.y_start}\n\t\ty_end:{crop_result.y_end}\n\t\ty_diff:{crop_result.y_end-crop_result.y_start}\n\tx\n\t\tx_start:{crop_result.x_start}\n\t\tx_end:{crop_result.x_end}\n\t\tx_diff:{crop_result.y_end-crop_result.y_start}")

# Captures a screenshot and optionally resizes and/or crops the captured image to a pre-specified size.
# The "resize_to" parameter takes a tuple, integer, or float.
#   If a tuple, such as (100, 100) is provided, it is interpreted as the desired width and height of the resized image.
#   If an integer or float is provided, the image is scaled in both dimensions according to its value.
# The "crop_to" parameter takes a tuple with values in the order:
#   (x_start, y_start, x_end, y_end)
def capture_screenshot(resize_to=None, crop_to=None):

    # Capture a full-screen screenshot and convert it from a PIL format to an OpenCV BGR Image.
    img = cv.cvtColor(np.array(screenshot()), cv.COLOR_RGB2BGR)

    # Resize the image if the resize_to argument is provided.
    if resize_to is not None:

        # If image pixel values are provided in a tuple, pass it to the OpenCV resize function as the dsize.
        if type(resize_to) == tuple:
            img = cv.resize(img, resize_to)
        
        # Otherwise, pass the integer or float value to the OpenCV resize function as the fx and fy scaling parameters.
        elif (type(resize_to) == float) or (type(resize_to) == int):
            img = cv.resize(img, dsize=None, fx=resize_to, fy=resize_to)

        else:
            raise TypeError(f"An invalid type \"{type(resize_to)}\" was provided as a \"resize_to\" parameter.")
    
    # Crop the image if the crop_to argument is provided.
    if crop_to is not None:

        assert (type(crop_to) == tuple), f"The crop_to shape of {crop_to} is not a tuple containing the x and y coordinates of a sub-image."

        img = img[int(crop_to[1]):int(crop_to[3]), int(crop_to[0]):int(crop_to[2])]

    return img

# Compares two images and returns a boolean value
# if the images are arbitrarily similar enough to be identical.
def imgs_are_same(img_1, img_2) -> bool:

    # Convert the images to a grayscale format.
    monochrome_1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
    monochrome_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

    # Compute the structural similarity of the the two images.
    score = ssim(monochrome_1, monochrome_2)

    if DISPLAY_DEBUG_MESSAGES:
        print(f"Similarity score: {score}")

    if score > 0.85:
        return True
    else:
        return False
    

# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
#  ―――――――――――――――
# | Main Function |
#  ―――――――――――――――
# ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――

if __name__ == "__main__":

    # Display a welcome message.
    print("MaxShot | Automatic Screenshot Script")

    # Display the image storage path.
    print(f"Image storage path set to:\n{IMAGE_STORAGE_PATH}")

    # Create folders leading to the image storage path if they do not already exist.
    if not os.path.exists(IMAGE_STORAGE_PATH):
        os.makedirs(IMAGE_STORAGE_PATH)

    # Move into the image storage path.
    os.chdir(IMAGE_STORAGE_PATH)

    time.sleep(2)

    # Guide the user to select a capture area.
    RESIZE_FACTOR = 0.75
    img = capture_screenshot(resize_to=RESIZE_FACTOR)
    cv.namedWindow(CROP_WINDOW_NAME)
    cv.setMouseCallback(CROP_WINDOW_NAME, crop_with_mouse)

    # Save the image to the crop_result dataclass for future use.
    crop_result.img = img

    # Display the image.
    cv.imshow(CROP_WINDOW_NAME, img)

    # Wait for the user to select a capture area.
    while not crop_result.crop_complete:
         if cv.waitKey(1) == ord("q"):
            break
    
    # Close the crop selection window.
    time.sleep(0.5)
    cv.destroyWindow(CROP_WINDOW_NAME)

    # Adjust the crop result to account for the reduced image size.
    crop_result.x_start = crop_result.x_start / RESIZE_FACTOR
    crop_result.y_start = crop_result.y_start / RESIZE_FACTOR
    crop_result.x_end = crop_result.x_end / RESIZE_FACTOR
    crop_result.y_end = crop_result.y_end / RESIZE_FACTOR

    if DISPLAY_DEBUG_MESSAGES:
        print(f"Adjusted Crop Values:\n\ty:\n\t\ty_start:{crop_result.y_start}\n\t\ty_end:{crop_result.y_end}\n\t\ty_diff:{crop_result.y_end-crop_result.y_start}\n\tx\n\t\tx_start:{crop_result.x_start}\n\t\tx_end:{crop_result.x_end}\n\t\tx_diff:{crop_result.y_end-crop_result.y_start}")

    # Primary program loop
    previous_img = capture_screenshot(crop_to=(crop_result.x_start, crop_result.y_start, crop_result.x_end, crop_result.y_end))
    img_count = 1
    while not cv.waitKey(1) == ord("q"):

        # Capture the screen every pre-specified number of seconds.
        time.sleep(CAPTURE_DELAY_TIME)
        img = capture_screenshot(crop_to=(crop_result.x_start, crop_result.y_start, crop_result.x_end, crop_result.y_end))
        
        # If the images are the same, continue waiting.
        if imgs_are_same(img, previous_img):
            previous_img = img
            continue
        else:
            previous_img = img

        # Find a valid path to save the image at.
        valid_img_path_found = False
        while not valid_img_path_found:

            # If the current img_count is a single-digit number, prepend a 0.
            if img_count < 10:
                current_img_count_string = f"0{img_count}"
            else:
                current_img_count_string = f"{img_count}"
            
            full_path = os.path.join(IMAGE_STORAGE_PATH, f"{current_img_count_string}.png")

            # If the path already exists, increment the img_count number.
            if os.path.exists(full_path):
                img_count += 1
                continue

            else:
                # If OpenCV is unable to write the image, exit the current save process and wait for the next image.
                try:
                    cv.imwrite(f"{current_img_count_string}.png", img)
                except:
                    break

                print(f"Image {current_img_count_string} has been saved to:\n{full_path}")
                valid_img_path_found = True
                img_count += 1