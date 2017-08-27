#!/usr/bin/env python
"""
Extract labelled traffic light images and save each in separate file
in the specified folder.
Filename convention is 000000_classname.png
Image sizes are variable, no resizing happens.

Example usage:
    python save_tl_images input.yaml output_folder
"""
import sys
import os
import cv2
from read_label_file import get_all_labels
from show_label_images import ir


def save_tl_images(input_yaml, output_folder):
    """
    Extracts labelled pictures of traffic lights.
    Saves them as separate files in specified output_folder.

    :param input_yaml: Path to yaml file
    :param output_folder: path to folder. created if does not exist
    """
    images = get_all_labels(input_yaml)

    assert output_folder is not None

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    j = 1
    for i, image_dict in enumerate(images):
        image = cv2.imread(image_dict['path'])
        if image is None:
            raise IOError('Could not open image path', image_dict['path'])

        for box in image_dict['boxes']:
            xmin = ir(box['x_min'])
            ymin = ir(box['y_min'])
            xmax = ir(box['x_max'])
            ymax = ir(box['y_max'])
            if xmax-xmin<=0 or ymax-ymin<=0:
                continue
            label = box['label']
            roi = image[ymin:(ymax+1), xmin:(xmax+1)]
            filename = os.path.join(output_folder,
                                    str(j).zfill(6) + '_' + label.lower() + '.png')
            #print("{}: {}".format(i, filename))
            cv2.imwrite(filename, roi)
            if os.stat(filename).st_size==0:
              os.remove(filename)
              print("saved file is zero size, deleting: {} {}".format(i, filename))
            j += 1


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(-1)
    label_file = sys.argv[1]
    output_folder = sys.argv[2]
    save_tl_images(label_file, output_folder=output_folder)
