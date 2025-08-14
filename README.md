# SimSTC
The source code for "A Simple Graph Contrastive Learning Framework for Short Text Classification"

Here we provide two datasets, snippets and mr, for reproducibility.
## Usage
You can run the following the command.

```
python train.py --dataset 'snippets'
```

You can change the ```--dataset``` to ```mr``` to train another dataset.

We provide the `original datasets` used in our work, along with the `preprocess.py` script to convert the raw data into the required target format.

## Related Work
Our related work can be found here: [GIFT](https://github.com/KEAML-JLU/GIFT) and [MI-DELIGHT](https://github.com/KEAML-JLU/MI-DELIGHT).
## Cite
If you find our work can help your research, please cite our work! <br>
```
@inproceedings{liu2022few,
  title={Improved Graph Contrastive Learning for Short Text Classification},
  author={Liu, Yonghao and Huang, Lan and Giunchiglia, Fausto and Feng, Xiaoyue and Guan, Renchu},
  booktitle={Proceedings of the Thirty-Eighth Conference on Association for the Advancement of Artificial Intelligence (AAAI)},
  year={2024}
}
```

```
@inproceedings{liu2025boosting,
  title={Boosting Short Text Classification with Multi-Source Information Exploration and Dual-Level Contrastive Learning},
  author={Liu, Yonghao and Li, Mengyu and Pang, Wei and Giunchiglia, Fausto and Huang, Lan and Feng, Xiaoyue and Guan, Renchu},
  booktitle={Proceedings of the Thirty-Nineth Conference on Association for the Advancement of Artificial Intelligence (AAAI)},
  year={2025}
}
```
```
@inproceedings{liu2025simple,
  title={A Simple Graph Contrastive Learning Framework for Short Text Classification},
  author={Liu, Yonghao and Giunchiglia, Fausto and Huang, Lan and Li, Ximing and Feng, Xiaoyue and Guan, Renchu},
  booktitle={Proceedings of the Thirty-Nineth Conference on Association for the Advancement of Artificial Intelligence (AAAI)},
  year={2025}
}
```

## Contact
If you have any question, feel free to contact via [email](mailto:yonghao20@mails.jlu.edu.cn).
