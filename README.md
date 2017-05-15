## Deep Co-occurrence Feature Learning for Visual Object Recognition ##

[Ya-Fang Shih](https://yafangshih.github.io)\*, Yang-Ming Yeh\*, [Yen-Yu Lin](https://www.citi.sinica.edu.tw/pages/yylin/), Ming-Fang Weng, Yi-Chang Lu, [Yung-Yu Chuang](https://www.csie.ntu.edu.tw/~cyy/) 
<br />
(CVPR 2017)

    .
    ├── src/                  
    ├── data/                   
    │   ├── cub/          
    │   └── models/         
    ├── exp/                   
    │   ├── cub-imdb.mat
    │   ├── deep_cooc_models/         
    │   └── feature_maps/        
    ├── from-bcnn-package/                  
    ├── matconvnet/               
    └── vlfeat/                 

### Installation  
1) clone the Deep-COOC repository
2) follow the [intructions](http://www.vlfeat.org/matconvnet/install/) to install matconvnet
3) download [vlfeat toolbox](http://www.vlfeat.org/download.html), and put the files into `vlfeat/` 
4) download [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), and put the files into `data/cub/`
5) download [ImageNet-pretrained CNN models](http://www.vlfeat.org/matconvnet/pretrained/) (in our experiments, we used the **imagenet-resnet-152-dag model**), and put them into `data/models/`
6) run `src/main.m`

