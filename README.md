# CODA-LM
[![arXiv](https://img.shields.io/badge/arXiv-2404.10595-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2404.10595) [![arXiv](https://img.shields.io/badge/Web-CODA_LM-blue.svg?style=plastic)](https://coda-dataset.github.io/coda-lm/)

This repository contains the implementation of the paper:

> Automated Evaluation of Large Vision-Language Models on Self-driving Corner Cases <br>
> [Yanze Li]()\*, [Wenhua Zhang]()\*, [Kai Chen](https://kaichen1998.github.io)\*, [Yanxin Liu](), [Pengxiang Li](https://scholar.google.com/citations?user=rUp_4RgAAAAJ&hl=en), [Ruiyuan Gao](https://gaoruiyuan.com/), [Lanqing Hong](https://scholar.google.com.sg/citations?user=2p7x6OUAAAAJ&hl=en), [Meng Tian](), [Xinhai Zhao](), [Zhenguo Li](https://scholar.google.com/citations?user=XboZC1AAAAAJ&hl=en&oi=ao), [Dit-Yan Yeung](https://sites.google.com/view/dyyeung), [Huchuan Lu](https://scholar.google.com/citations?user=D3nE0agAAAAJ&hl=en), [Xu Jia](https://stephenjia.github.io/)† <br>
> *Equal Contribution   †Corresponding Author

<img src="./images/overview.png" style="width: 65%; margin: 0 auto; text-align: center"/>



## Data Preparation

The instructions for downloading CODA-LM is listed as follows:

1. Download the image files following the CODA official instructions [here](https://coda-dataset.github.io/download.html#instructions)

2. Download the CODA-LM annotation files and then decompress in the same root directory.

After decompression, the data organization is listed as follows:

```
├── val
│   │── annotations.json
│   │── images
│   │   │── *_*.jpg
├── test
│   │── annotations.json
│   │── images
│   │   │── *_*.jpg
├── CODA-LM
│   │── Train
│   │   │── *_*.json
│   │── Val
│   │   │── *_*.json
│   │── Test
│   │   │── *_*.json
│   │── Mini
│   │   │── *_*.json
```



## Data Format

The annotation files contains question-answering pairs for all three tasks as following,

```
{
	"general_perception":{
	},
	"region_perception":{
	},
	"driving_suggestion": <str>,
}
```



## Check Yourself!





## Citation

```bibtex
@article{li2024automated,
  title={Automated Evaluation of Large Vision-Language Models on Self-driving Corner Cases},
  author={Li, Yanze and Zhang, Wenhua and Chen, Kai and Liu, Yanxin and Li, Pengxiang and Gao, Ruiyuan and Hong, Lanqing and Tian, Meng and Zhao, Xinhai and Li, Zhenguo and others},
  journal={arXiv preprint arXiv:2404.10595},
  year={2024}
}
```

