# MOTIOD

## Conda

``` 
conda create -n MOTIOD
conda activate MPTIOD
```


## Installation

``` 
pip install -r requirements.txt
```
- Linux, CUDA>=9.2, GCC>=5.4
- Python>=3.7
- PyTorch ≥ 1.5 and torchvision that matches the PyTorch installation. You can install them together at pytorch.org to make sure of this
- OpenCV is optional and needed by demo and visualization



## Train
``` 
python main_track.py  --output_dir /path/to/folder/ --dataset_file mot --batch_size 2 --with_box_refine --num_queries 500 --resume
/path/to/checkpoint.pth
```

## Demo
``` 
python demo_new.py  --with_box_refine
```


## Acknowledgement
We thank for the part of code of [TransTrack](https://github.com/PeizeSun/TransTrack). We thank the authors for releasing their code.




