import argparse
import datetime
import json
import os
from PIL import Image

import pycococreatortools


def get_arguments():
    parser = argparse.ArgumentParser(description="transform mask annotation to coco annotation")
    parser.add_argument("--dataset", type=str, default='CIHP', help="name of dataset (CIHP, MHPv2 or VIP)")
    parser.add_argument("--json_save_dir", type=str, default='../data/CIHP/annotations',
                        help="path to save coco-style annotation json file")
    parser.add_argument("--test_img_dir", type=str, default='../data/CIHP/Testing/Images',
                        help="test image path")
    return parser.parse_args()

args = get_arguments()

INFO = {
    "description": args.dataset + "Dataset",
    "url": "",
    "version": "",
    "year": 2020,
    "contributor": "yunqiuxu",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "",
        "url": ""
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'person',
        'supercategory': 'person',
    },
]

def main(args):
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1

    for root, _, files in os.walk(args.test_img_dir):
        for image_name in files:
            if image_name.endswith('.png'):
                image_path = os.path.join(root, image_name)
                image = Image.open(image_path)
                image_info = pycococreatortools.create_image_info(
                    image_id, os.path.relpath(image_path, args.test_img_dir), image.size
                )
                coco_output["images"].append(image_info)
                image_id += 1
        if image_id >= 23:
            break

    if not os.path.exists(args.json_save_dir):
        os.makedirs(args.json_save_dir)

    with open(os.path.join(args.json_save_dir, f'{args.dataset}.json'), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main(args)
