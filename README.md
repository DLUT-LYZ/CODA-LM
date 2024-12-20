# CODA-LM
[![arXiv](https://img.shields.io/badge/arXiv-2404.10595-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2404.10595) [![Web](https://img.shields.io/badge/Web-CODA_LM-blue.svg?style=plastic)](https://coda-dataset.github.io/coda-lm/) [![HF](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow?style=plastic)](https://huggingface.co/collections/KaiChen1998/coda-lm-6726500ab7d88dbcf9dc3fd0)

This repository contains the implementation of the paper:

> Automated Evaluation of Large Vision-Language Models on Self-driving Corner Cases <br>
> [Kai Chen](https://kaichen1998.github.io)\*, [Yanze Li]()\*, [Wenhua Zhang]()\*, [Yanxin Liu](), [Pengxiang Li](https://scholar.google.com/citations?user=rUp_4RgAAAAJ&hl=en), [Ruiyuan Gao](https://gaoruiyuan.com/), [Lanqing Hong](https://scholar.google.com.sg/citations?user=2p7x6OUAAAAJ&hl=en)†, [Meng Tian](), [Xinhai Zhao](), [Zhenguo Li](https://scholar.google.com/citations?user=XboZC1AAAAAJ&hl=en&oi=ao), [Dit-Yan Yeung](https://sites.google.com/view/dyyeung), [Huchuan Lu](https://scholar.google.com/citations?user=D3nE0agAAAAJ&hl=en), [Xu Jia](https://stephenjia.github.io/)† <br>
> *Equal Contribution   †Corresponding Authors
> <br>*IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2025*

<p align="center">
  <img src="./images/overview.png" style="width: 65%; margin: 0 auto; text-align: center"/>
</p>

## Update

- **2024.11**: CODA-LM in the LLaVA data format has been released on [HuggingFace](https://huggingface.co/datasets/KaiChen1998/coda-lm-llava-format)! Check [here](https://github.com/DLUT-LYZ/CODA-LM?tab=readme-ov-file#-llava-format) for more details.
- **2024.10**: CODA-LM Test set annotations have been released on [HuggingFace](https://huggingface.co/datasets/KaiChen1998/coda-lm/tree/main/CODA-LM/Test)!
- **2024.10**: CODA-LM has been accepted by WACV 2025!

## Data Preparation

The instructions for downloading CODA-LM are listed as follows:

1. Download the image files following the CODA official instructions [here](https://coda-dataset.github.io/download.html#instructions)
2. Download the **original** CODA-LM annotation files and then decompress them in the same root directory.

| Split | Size | Image Source  |                       Original Format                        | LLaVA Format                                                 |
| :---: | :--: | :-----------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Train | 4884 | CODA2022 val  | [HF Hub](https://huggingface.co/datasets/KaiChen1998/coda-lm) | [HF Hub](https://huggingface.co/datasets/KaiChen1998/coda-lm-llava-format) |
|  Val  | 4384 | CODA2022 test | [HF Hub](https://huggingface.co/datasets/KaiChen1998/coda-lm) | [HF Hub](https://huggingface.co/datasets/KaiChen1998/coda-lm-llava-format) |
| Test  | 500  | CODA2022 test | [HF Hub](https://huggingface.co/datasets/KaiChen1998/coda-lm) | [HF Hub](https://huggingface.co/datasets/KaiChen1998/coda-lm-llava-format) |
| Mini  |  50  | CODA2022 test | [HF Hub](https://huggingface.co/datasets/KaiChen1998/coda-lm) | [HF Hub](https://huggingface.co/datasets/KaiChen1998/coda-lm-llava-format) |

Note that:

1. Images of **CODA-LM train** set come from **CODA2022 val** set, while images of **CODA-LM val and test** sets come from **CODA2022 test** set.
2. **CODA-LM mini** set is a 50-image subset of **CODA-LM val** set for demonstration.

After decompression, the data organization is listed as follows:

```
├── val                    -- CODA2022 val (we only use images)
│   │── images
│   │   │── *.jpg
├── test                   -- CODA2022 test (we only use images)
│   │── images
│   │   │── *.jpg
├── CODA-LM
│   │── Train              -- CODA-LM train (we use 4884 images from CODA2022 val)
│   │   │── val_*.json
│   │── Val                -- CODA-LM val (we use 4384 images from CODA2022 test)
│   │   │── test_*.json
│   │── Test               -- CODA-LM test (we use 500 images from CODA2022 test)
│   │   │── test_*.json
│   │── Mini               -- CODA-LM mini (a 50-image subset of CODA-LM val)
│   │   │── test_*.json
```

## Task Hierarchy

<p align="center">
  <img src="./images/tasks.PNG" style="width: 30%; margin: 0 auto; text-align: center"/>
</p>

CODA-LM contains three tasks to promote the development of more reliable and interpretable autonomous driving agents.
- **General Perception.**  The foundational aspect of the general perception task lies in the comprehensive understanding of key entities within driving scenes, including the appearance, location, and the reasons they influence driving. Given a first-view road driving scene, MLLMs are required to describe all the potential road obstacles in the traffic scenes and explain why they would affect the driving decisions. We primarily focus on seven categories of obstacles, including vehicles, vulnerable road users (VRUs), traffic cones, traffic lights, traffic signs, barriers, and miscellaneous.

- **Region Perception.**  This task measures the MLLMs' capability to understand corner objects when provided with specific bounding boxes, followed by describing these objects and explaining why they might affect self-driving behavior. Note that for region perception, there are no constraints on how to use bounding boxes in MLLMs. Any possible encoding leading to better performance is feasible. We provide an example by visualizing the bounding boxes on images with red rectangles here for reference.

- **Driving Suggestions.**  This task aims to evaluate MLLMs' capability in formulating driving advice in the autonomous driving domain. This task is closely related to the planning process of autonomous driving, requiring MLLMs to provide optimal driving suggestions for the ego car after correctly perceiving the general and regional aspects of the current driving environment.

## Data Format

The annotation files contains question-answering pairs for all three tasks as following,

```
{
    "general_perception":{
        "vehicles": [                             -- list containing information on all vehicles
            {
                "description": <str>,             -- description about a single vehicle
                "explanation": <str>"             -- explanation why it affects the ego car
            },
        "vulnerable_road_users": [...],           -- list containing information on all VRUs
        "traffic signs": [...],                   -- list containing information on all traffic signs
        "traffic lights": [...],                  -- list containing information on all traffic lights
        "traffic cones": [...],                   -- list containing information on all traffic cones
        "barriers": [...],                        -- list containing information on all barriers
        "other objects": [...],                   -- list containing information on all other objects
    	"description and explanation": <str>      -- summarization of information on all categories
    },
    "region_perception":{
        "1": {                                    -- region index
            "description and explanation": <str>, -- description of road users in the specific region with explanation on why it affects the ego car
            "box": <list of float>,               -- xywh coordinates
            "category_name": <str>                -- object category
        },
        "2": {...},
        "3": {...}
    },
    "driving_suggestion": <str>,
}
```

## Data Usage

### ✨ LLaVA Format

To better facilitate training LVLMs with CODA-LM, we further organize the CODA-LM data in the LLaVA data format, based on which, we can use CODA-LM simply by utilizing the HuggingFace APIs. 

1. Install the HuggingFace datasets dependency via pip.

   ```bash
   pip install datasets
   ```

2. Download and load the specified subsets and splits of CODA-LM. Note that by default, we adopt the **red rectangle prompt** for the regional perception task.

```python
from datasets import load_dataset

# name can be selected from ['English', 'Chinese']
# split can be selected from ['Mini', 'Train', 'Val', 'Test']
dataset = load_dataset("KaiChen1998/coda-lm-llava-format", name="English", split='Train')

# should be a dictionary containing
# {"id": sample identification, 'image': PIL Image, 'conversations': with <image> token}
for data in dataset:
    print(data)
```

### Original Format

To help users better understand the structure of CODA-LM, we provide a python script to convert our annotations to basic VQA formats, as follows:

1. Download the data and make sure the directory organization follows [Data Prepration](https://github.com/DLUT-LYZ/CODA-LM?tab=readme-ov-file#data-preparation).

2. Run `convert2vqa.py` as follows:

   ```bash
   # English
   python convert2vqa.py --coda_root $CODA_ROOT --codalm_ann_name CODA-LM
   
   # Chinese
   python convert2vqa.py --coda_root $CODA_ROOT --codalm_ann_name CODA-LM-chinese
   ```

3. After that, the resulting data organization will be like this:

   ```
   ├── val
   │   │── images
   │   │── images_w_bboxes                  -- Images with bboxes drawn for region perception
   │   │   │── *.jpg
   ├── test
   │   │── images
   │   │── images_w_bboxes                  -- Images with bboxes drawn for region perception
   │   │   │── *.jpg
   ├── CODA-LM
   │   │── Train
   │   │   │── vqa_anno
   │   │   │   │── general_perception.jsonl  -- VQA annotations for general perception
   │   │   │   │── region_perception.jsonl   -- VQA annotations for region perception
   │   │   │   │── driving_suggestion.jsonl  -- VQA annotations for driving suggestion
   │   │── Val
   │   │   │── vqa_anno
   │   │── Test
   │   │   │── vqa_anno
   │   │── Mini
   │   │   │── vqa_anno
   ```

4. The basic VQA format saves data sample simply with a dictionary containing `question_id`, `image`, `question`, and `answer`, as follows:

   ```
   {"question_id": 0, "image": val/images/0001.jpg, "question": <str>, "answer": <str>}
   {"question_id": 1, "image": val/images/0002.jpg, "question": <str>, "answer": <str>}
   {"question_id": 2, "image": val/images/0003.jpg, "question": <str>, "answer": <str>}
   ...
   ```

5. Note that for regional perception, there are various possible manners to utilize the bbox annotations. Here we provide a simple implementation by drawing the bboxes with red rectangles on images, which are saved in the `images_w_bboxes` directory.

   ![img](images/visual.png)


## Data Annotation

Check [CODA-LM Annotation Tool](./labeling) for more details.


# Evaluation

Check [CODA-LM Evaluation](./evaluation) for more details.



## Citation

```bibtex
@article{li2024automated,
  title={Automated Evaluation of Large Vision-Language Models on Self-driving Corner Cases},
  author={Li, Yanze and Zhang, Wenhua and Chen, Kai and Liu, Yanxin and Li, Pengxiang and Gao, Ruiyuan and Hong, Lanqing and Tian, Meng and Zhao, Xinhai and Li, Zhenguo and others},
  journal={arXiv preprint arXiv:2404.10595},
  year={2024}
}
```

