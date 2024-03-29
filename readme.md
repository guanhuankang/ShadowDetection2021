![image-20220814170427728](other/title.png)

## Delving into Dark Regions for Robust Shadow Detection
**Keywords** Shadow detection *·* Global/local shadow understanding *·* Region-based shadow analysis

> This work was done in 2021.

[<a href="https://arxiv.org/abs/2402.13631">paper</a>] [<a href="https://github.com/guanhuankang/ShadowDetection2021">github</a>]

![visual](other/more_visual.jpg)

## Resources

Download model weights and put into folder "models"

SBU: https://drive.google.com/file/d/1TY8O5F9GPB0Zv7CxQoAUaqQTbJFLdb6a/view?usp=sharing

ISTD: https://drive.google.com/file/d/1tc6WpmlAwZTg7aJL18moz2bBn9Rxves0/view?usp=sharing

![](other/quantitative.jpg)

## Demo

We provide a demo in jupyter notebook format "demo.ipynb". Start the jupyter-lab and run all cells for a quick start!

![image-20220814171349944](other/demo.png)



## Inference

1. configurate "config.py"
2. run "infer.py"

```
BER Score:
SBU: 3.04
UCF: 7.75
ISTD: 1.33
```



If you want to train our model, please refer to other/train_script.rar for details (todo: make it better for revision).



## Evaluation

1. After the inference, set the tool/config.py
2. Run "cd tool; python start.py"

## Cite
If you find our work is helpful, cite and star it! Thanks!

```html
@misc{guan2024delving,
      title={Delving into Dark Regions for Robust Shadow Detection}, 
      author={Huankang Guan and Ke Xu and Rynson W. H. Lau},
      year={2024},
      eprint={2402.13631},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

