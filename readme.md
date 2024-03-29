# ACSA-HGCN
Aspect-Category based Sentiment Analysis with Hierarchical Graph Convolutional Network [[COLING 2020]](https://www.aclweb.org/anthology/2020.coling-main.72.pdf).

## Requirements
* Python 3.7
* Pytorch 1.5.1

## Running
Modify the corresponding BERT_BASE_DIR, DATA_DIR and output_dir to run the script.
```
sh run.sh
```

## Note
The statistical numbers of the Explicit and Implicit Aspects test set in the first paragraph of section 4.5 were put in the wrong order. The Explicit Aspects test set contains 360 samples and the Implicit Aspects test set contains 212 samples.


## Citation
If the code is used in your research, please cite our paper as follows:
```
@inproceedings{cai2020aspect,
  title={Aspect-Category based Sentiment Analysis with Hierarchical Graph Convolutional Network},
  author={Cai, Hongjie and Tu, Yaofeng and Zhou, Xiangsheng and Yu, Jianfei and Xia, Rui},
  booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
  pages={833--843},
  year={2020}
}
```
