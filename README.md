# TimeSformer
This is a test demo for Timesformer the detail information is in [timesformer]('https://arxiv.org/abs/2102.05095') for paper and origin code. The model is as follows:
![screenshot](https://github.com/lucidrains/TimeSformer-pytorch/blob/main/diagram.png?raw=true 'model')
we do a test demo on kinectics 600 use T+S model
# example
The test of our video is 
![screenshot](data1.gif)
the output of this video is bowling

# usage
Test with you own test videos by use `test_demo.py`
`model_file` to your model path and `video_path` to your own video path



# Acknowledgements

TimeSformer is built on top of [PySlowFast](https://github.com/facebookresearch/SlowFast) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) by [Ross Wightman](https://github.com/rwightman). We thank the authors for releasing their code. If you use our model, please consider citing these works as well:

```BibTeX
@misc{fan2020pyslowfast,
  author =       {Haoqi Fan and Yanghao Li and Bo Xiong and Wan-Yen Lo and
                  Christoph Feichtenhofer},
  title =        {PySlowFast},
  howpublished = {\url{https://github.com/facebookresearch/slowfast}},
  year =         {2020}
}
```

```BibTeX
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```