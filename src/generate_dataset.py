'''
Imported from https://github.com/tylerhutcherson/synthetic-images
'''

import argparse
import json
import os
import random
from pathlib import Path

import numpy
import numpy as np
from PIL import Image, ImageEnhance
# Entrypoint Args
from cv2 import cv2

parser = argparse.ArgumentParser(description='Create synthetic training data for object detection algorithms.')
parser.add_argument("-bkg", "--backgrounds", type=str, default="datasets/raw/background/", help="Path to background images folder.")
parser.add_argument("-obj", "--objects", type=str, default="datasets/raw/objects/", help="Path to object images folder.")
parser.add_argument("-o", "--output", type=str, default="datasets/", help="Path to output images folder.")
parser.add_argument("-ann", "--annotate", type=bool, default=True, help="Include annotations in the data augmentation steps?")
parser.add_argument("-s", "--sframe", type=bool, default=False, help="Convert dataset to an sframe?")
parser.add_argument("-g", "--groups", type=bool, default=True, help="Include groups of objects in training set?")
parser.add_argument("-mut", "--mutate", type=bool, default=False, help="Perform mutations to objects (rotation, brightness, sharpness, contrast)")
args = parser.parse_args()

# Prepare data creation pipeline
base_bkgs_path = args.backgrounds
bkg_images = [f for f in os.listdir(base_bkgs_path) if not f.startswith(".")]
objs_path = args.objects
obj_images = [f for f in os.listdir(objs_path) if not f.startswith(".")]


# Helper functions
def get_obj_positions(obj, bkg, count=1):
    obj_w, obj_h = [], []
    x_positions, y_positions = [], []
    bkg_w, bkg_h = bkg.size
    # Rescale our obj to have a couple different sizes
    obj_sizes = [tuple([int(random.randint(1, 4) * x) for x in obj.size]) for _ in range(count)]
    for w, h in obj_sizes:
        obj_w.extend([w])
        obj_h.extend([h])
        max_x, max_y = bkg_w - w, bkg_h - h
        x_positions.extend(list(np.random.randint(0, max_x, count)))
        y_positions.extend(list(np.random.randint(0, max_y, count)))
    return obj_h, obj_w, x_positions, y_positions


def get_box(obj_w, obj_h, max_x, max_y):
    x1, y1 = np.random.randint(0, max_x, 1), np.random.randint(0, max_y, 1)
    x2, y2 = x1 + obj_w, y1 + obj_h
    return [x1[0], y1[0], x2[0], y2[0]]


# check if two boxes intersect
def intersects(box, new_box):
    box_x1, box_y1, box_x2, box_y2 = box
    x1, y1, x2, y2 = new_box
    return not (box_x2 < x1 or box_x1 > x2 or box_y1 > y2 or box_y2 < y1)


def get_group_obj_positions(obj_group, bkg):
    bkg_w, bkg_h = bkg.size
    boxes = []
    objs = [Image.open(objs_path + obj_images[i]) for i in obj_group]
    obj_sizes = []
    for obj in objs:
        random_scale = random.randint(1, 4)
        obj_sizes.append(tuple([sz * random_scale for sz in obj.size]))
    for w, h in obj_sizes:
        # set background image boundaries
        max_x, max_y = bkg_w - w, bkg_h - h
        # get new box coordinates for the obj on the bkg
        while True:
            new_box = get_box(w, h, max_x, max_y)
            for box in boxes:
                res = intersects(box, new_box)
                if res:
                    break

            else:
                break  # only executed if the inner loop did NOT break
            # print("retrying a new obj box")
            continue  # only executed if the inner loop DID break
        # append our new box
        boxes.append(new_box)
    return obj_sizes, boxes


def mutate_image(img):
    # resize image for random value
    resize_rate = random.randint(1, 4)
    img = img.resize([int(img.width * resize_rate), int(img.height * resize_rate)], Image.BILINEAR)

    # rotate image for random andle and generate exclusion mask
    rotate_angle = random.randint(0, 360)
    mask = Image.new('L', img.size, 255)
    img = img.rotate(rotate_angle, expand=True)
    mask = mask.rotate(rotate_angle, expand=True)

    # perform some enhancements on image
    enhancers = [ImageEnhance.Brightness, ImageEnhance.Color, ImageEnhance.Contrast, ImageEnhance.Sharpness]
    enhancers_count = random.randint(0, 3)
    for i in range(0, enhancers_count):
        enhancer = random.choice(enhancers)
        enhancers.remove(enhancer)
        img = enhancer(img).enhance(random.uniform(0.5, 1.5))

    return img, mask


def generate(output_path, single_obj_count=1, multi_object_count=100):
    global bkg_images, base_bkgs_path, objs_path, obj_images
    n = 1
    annotations = []  # store annots here

    # Make synthetic training data
    print("\nMaking synthetic images.", flush=True)

    # if there are no background images then generate one black background
    if len(os.listdir(base_bkgs_path)) == 0:
        cv2.imwrite(filename=base_bkgs_path + 'black.png', img=numpy.zeros(shape=(512, 512, 3)))
        bkg_images = [f for f in os.listdir(base_bkgs_path) if not f.startswith(".")]

    for bkg in bkg_images:
        # Load the background image
        bkg_path = base_bkgs_path + bkg
        bkg_img = Image.open(bkg_path)
        bkg_x, bkg_y = bkg_img.size

        # Do single objs first
        for i in obj_images:
            # Load the single obj
            i_path = objs_path + i
            obj_img = Image.open(i_path)

            # Get an array of random obj positions (from top-left corner)
            obj_h, obj_w, x_pos, y_pos = get_obj_positions(obj=obj_img, bkg=bkg_img, count=single_obj_count)

            # Create synthetic images based on positions
            for h, w, x, y in zip(obj_h, obj_w, x_pos, y_pos):
                # Copy background
                bkg_w_obj = bkg_img.copy()

                if args.mutate:
                    new_obj, mask = mutate_image(obj_img)
                    # Paste on the obj
                    bkg_w_obj.paste(new_obj, (x, y), mask)
                else:
                    # Adjust obj size
                    new_obj = obj_img.resize(size=(w, h))
                    # Paste on the obj
                    bkg_w_obj.paste(new_obj, (x, y))
                output_fp = output_path + str(n) + ".png"
                # Save the image
                bkg_w_obj.save(fp=output_fp, format="png")

                if args.annotate:
                    # Make annotation
                    ann = [{'coordinates': {'height': h, 'width': w, 'x': x + (0.5 * w), 'y': y + (0.5 * h)}, 'label': i.split(".png")[0]}]
                    # Save the annotation data
                    annotations.append({
                        "path": output_fp,
                        "annotations": ann
                    })
                # print(n)
                n += 1

        if args.groups:
            # 24 Groupings of 2-4 objs together on a single background
            groups = [np.random.randint(0, len(obj_images) - 1, np.random.randint(2, 5, 1)) for r in range(multi_object_count)]
            # For each group of objs
            for group in groups:
                # Get sizes and positions
                ann = []
                obj_sizes, boxes = get_group_obj_positions(group, bkg_img)
                bkg_w_obj = bkg_img.copy()

                # For each obj in the group
                for i, size, box in zip(group, obj_sizes, boxes):
                    # Get the obj
                    obj = Image.open(objs_path + obj_images[i])
                    obj_w, obj_h = size
                    # Resize it as needed
                    new_obj = obj.resize((obj_w, obj_h))
                    x_pos, y_pos = box[:2]
                    if args.annotate:
                        # Add obj annotations
                        annot = {
                            'coordinates': {
                                'height': obj_h,
                                'width': obj_w,
                                'x': int(x_pos + (0.5 * obj_w)),
                                'y': int(y_pos + (0.5 * obj_h))
                            },
                            'label': obj_images[i].split(".png")[0]
                        }
                        ann.append(annot)
                    # Paste the obj to the background
                    bkg_w_obj.paste(new_obj, (x_pos, y_pos))

                output_fp = output_path + str(n) + ".png"
                # Save image
                bkg_w_obj.save(fp=output_fp, format="png")
                if args.annotate:
                    # Save annotation data
                    annotations.append({
                        "path": output_fp,
                        "annotations": ann
                    })
                # print(n)
                n += 1

    if args.annotate:
        print("Saving out Annotations", flush=True)
        # Save annotations
        with open(Path(output_path).joinpath('annotations.json'), 'w') as f:
            f.write(json.dumps(annotations, indent=2))

    if args.sframe:
        print("Saving out SFrame", flush=True)
        # Write out data to an sframe for turicreate training
        import turicreate as tc

        # Load images and annotations to sframes
        images = tc.load_images(output_path).sort("path")
        annots = tc.SArray(annotations).unpack(column_name_prefix=None).sort("path")
        # Join
        images = images.join(annots, how='left', on='path')
        # Save out sframe
        images[['image', 'path', 'annotations']].save("training_data.sframe")

    total_images = len([f for f in os.listdir(output_path) if not f.startswith(".")])
    print("Done! Created {} synthetic images in {}.".format(total_images, output_path), flush=True)


if __name__ == "__main__":
    generate(output_path=args.output + 'train/', single_obj_count=4, multi_object_count=10000)
    generate(output_path=args.output + 'test/', single_obj_count=1, multi_object_count=10)
