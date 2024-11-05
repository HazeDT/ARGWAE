# Adversarially regularized graph wavelet autoencoder  (ARGWAE)
This code is about the implementation of [A Novel Unsupervised Graph Wavelet Autoencoder for Mechanical System Fault Detection]().

![ARGWAE](https://github.com/HazeDT/ARGWAE/blob/main/ARGWAE.jpg)

# Note
The ARGWAE consists of a [SGWConv enocder](https://ieeexplore.ieee.org/abstract/document/10079151), a MLP decoder and an adversarial regularizer, and the framework of this code is based on [Learning graph embedding with adversarial training methods](https://ieeexplore.ieee.org/abstract/document/8822591).


# Implementation
python ./main.py --model_name ARGWAE  --checkpoint_dir ./results/   --data_name yourdata_name --data_dir ./data/your data --per_node 10


# Citation
SGWN: 
@ARTICLE{10079151,
  author={Li, Tianfu and Sun, Chuang and Fink, Olga and Yang, Yuangui and Chen, Xuefeng and Yan, Ruqiang},
  journal={IEEE Transactions on Cybernetics}, 
  title={Filter-Informed Spectral Graph Wavelet Networks for Multiscale Feature Extraction and Intelligent Fault Diagnosis}, 
  year={2024},
  volume={54},
  number={1},
  pages={506-518},
  doi={10.1109/TCYB.2023.3256080}}


ARGWAE:
@ARTICLE{ARGWAE,
  author={Li, Tianfu and Sun, Chuang and Yan, Ruqiang and Chen, Xuefeng},
  journal={Journal of Intelligent Manufacturing}, 
  title={A novel unsupervised graph wavelet autoencoder for mechanical system fault detection}, 
  year={2024},
  volume={},
  number={},
  pages={},
  doi={10.1007/s10845-024-02511-2}}



