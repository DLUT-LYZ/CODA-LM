import argparse
import torch
import os
import json
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import math
import os
import numpy as np


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img, width, height
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result, width, height
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result, width, height

def box_xyxy_expand2square(box, w, h):
    x1, y1, x2, y2 = box
    if w == h:
        return [round(x1 / w, 3), round(y1 / h, 3), round(x2 / w, 3), round(y2 / w, 3)]
    if w > h:
        x1, y1, x2, y2 = box
        y1 += (w - h) // 2
        y2 += (w - h) // 2
        box = [round(x1 / w, 3), round(y1 / h, 3), round(x2 / w, 3), round(y2 / w, 3)]
        return box
    assert w < h
    x1, y1, x2, y2 = box
    x1 += (h - w) // 2
    x2 += (h - w) // 2
    box = [round(x1 / w, 3), round(y1 / h, 3), round(x2 / w, 3), round(y2 / w, 3)]
    return box

# Custom dataset class
class CODALMDataset(Dataset):
    def __init__(self, 
                 data_root = 'CODA/CODA-LM',
                 version = 'Mini'):

        self.data_root = data_root
        self.version = version
        self.data_list = self.load_data_list()

    def load_data_list(self):
        data_path = os.path.join(self.data_root, self.version)
        json_list = [each for each in os.listdir(data_path) if each.endswith('.json')]
        json_list = sorted(json_list)  # test_0001, ....., val_0001, val_0002, ......

        data_list = []
        for json_name in json_list:
            raw_ann_info = dict()
            with open(os.path.join(data_path, json_name), "r") as f:
                json_info = json.load(f)
            
            img_root, img_id = json_name.replace(".json", "").split("_") # test, 0001

            raw_ann_info["img_path"] = os.path.join(self.data_root, "..", img_root, "images", img_id + ".jpg")
            raw_ann_info["text"] = json_info
            raw_ann_info["prompt"] = {
                "general_perception": "There is an image of traffic captured from the perspective of the ego car. Please focus on objects that have a great influence on ego car driving behavior in the scene, describe these objects and the reasons why they influence ego car driving.",
                "region_perception": "Please provide a description for this object <boxes> and explain why this object affect ego car driving.",
                "driving_suggestion": "There is an image of traffic captured from the perspective of the ego car. Please provide driving suggestions for the ego car based on the current scene.",
            }

            data_list.append(raw_ann_info)

        return data_list

    def parse_data_info(self, data_info, width, height):
        parse_data_info = {
            "round": []
        }

        
        # round1:
        res_info = []
        for key, value in data_info["text"]["general_perception"].items():
            if isinstance(value, list):
                for v in value:
                    res_info.append(v["description"])
                    res_info.append(v["explanation"])
            else:
                res_info.append(v["description"])
                res_info.append(v["explanation"])

        parse_data_info["round"].append({
            "question": data_info["prompt"]["general_perception"],
            "answer": " ".join(res_info)
        })

        # round2；
        parse_data_info["round"].append({
            "question": data_info["prompt"]["driving_suggestion"],
            "answer": data_info["text"]["driving_suggestion"]
        })
        
        # round3:
        for key, value in data_info["text"]["region_perception"].items():
            box = box_xyxy_expand2square(value["box"], width, height)
            box_str = f"<{box[0]}, {box[1]}, {box[2]}, {box[3]}>"
            # import pdb; pdb.set_trace()
            parse_data_info["round"].append({
                "question": data_info["prompt"]["region_perception"].replace("<boxes>", box_str),
                "answer": value["description and explanation"]
            })

        parse_data_info.update(data_info)

        return parse_data_info
            
    def __getitem__(self, index):
        data_info = self.data_list[index]

        # image open
        image = Image.open(data_info["img_path"]).convert('RGB')
        image, width, height = expand2square(image, 114)
        image_numpy = np.array(image)

        # text round
        parse_data_info = self.parse_data_info(data_info, width, height)

        parse_data_info.update({"image_numpy": image_numpy})

        return parse_data_info

    def __len__(self):
        return len(self.data_list)
     
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="CODA/CODA-LM")
    parser.add_argument("--version", type=str, default="Mini")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    assert args.batch_size == 1

    dataset = CODALMDataset(args.data_root, args.version)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    for data in tqdm(data_loader):
        image_tensor = data['image_numpy']
        rounds = data['round']
        print(rounds)
