This is a PyTorch example implementation of our paper: 
**Mining Negative Samples on Contrastive Learning via Curricular Weighting Strategy**

1.The codes of CIFAR-10,CIFAR-100,STL-10 and SVHN are in *image directory*.  
2.You can download the datasets online and store them in *data directory*.  
3.Then you can change the directory to *image*: 
>train: python main_cur.py --dataset_name cifar10/stl10/svhn/cifar100  
test: python linear.py --dataset_name cifar10/stl10/svhn/cifar100 --model_path _your model path_
### Acknowledgements

Part of this code is inspired by [joshr17/HCL](https://github.com/joshr17/HCL), by [HuangYG123/CurricularFace](https://github.com/HuangYG123/CurricularFace).
