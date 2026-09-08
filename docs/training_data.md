## LLaVA dataset
The LLaVA dataset dataset is used for training the different models. The details for data preparation are provided in the following link of TinyLLaVA: 
[Training Data Preparation](https://tinyllava-factory.readthedocs.io/en/latest/Prepare%20Datasets.html).

For housekeeping purposes we provide these details also locally in the following sections.

## Pretraining images
The pretraining images of LLaVA is from the 558K subset of the LAION-CC-SBU dataset. Download as follows.

LAION-CC-SBU-558K [images.zip](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/images.zip)

## Pretraining annotations
The pretraining annotations of LLaVA. Download as follows.

Pretraining annotations: [blip_laion_cc_sbu_558k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)

## SFT images
The SFT images of LLaVA. Download as follows.
* LAION-CC-SBU-558K: Already downloaded as “LAION-CC-SBU-558K” in Pretraining Images.
* COCO: This dataset is from the [COCO2017_challenge](https://cocodataset.org/). Download: [train2017](http://images.cocodataset.org/zips/train2017.zip)
* GQA: [GQA_project_page](https://cs.stanford.edu/people/dorarad/gqa/about.html). Download: [gqa_images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
* OCR-VQA: [OCR-VQA_project_page](https://ocr-vqa.github.io/). Download: [download_script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_). All files are saved as `.jpg`
* TextVQA: [TextVQA_project_page](https://textvqa.org/). Download: [trainval_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
* VisualGenome: [VisualGenome_project_page](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html). Download: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

## SFT annotations
The SFT annotations of LLaVA. Download as follows.

SFT annotations: [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)

##  Data organization

Organize the image files and annotation files as follows in `path/to/your/dataset` :

```
dataset
├── llava
│   ├── llava_pretrain
│   │   ├── images
├── coco
│   ├── train2017
├── sam
│   ├── images
├── gqa
│   ├── images
├── ocr_vqa
│   ├── images
├── textvqa
│   ├── train_images
├── vg
│   ├── VG_100K
│   ├── VG_100K_2
├── share_textvqa
│   ├── images
├── text_files
│   ├── blip_laion_cc_sbu_558k.json
│   ├── llava_v1_5_mix665k.json
```
