<div align="center"><img src="https://user-images.githubusercontent.com/29671696/126465498-1dfe6db1-8725-4b35-95d0-428ea8777b7f.png" width=200></div>

# CATEX: Crf-based Asr Toolkit with EXtensions

**CATEX is a CRF-based data-efficient end-to-end speech processing toolkit , includes speech recognition, pronunciation evaluation, text-to-speech, speech enhacement, speaker diarization, and so on. CATEX uses [pytorh](https://pytorch.org) and [tensorflow](https://www.tensorflow.org) as deep learning engine, and provide recipes for various speech processing experiments.**

* [Overview](#Overview)
* [Key Features](#Key-Features)
* [Installation](#Installation)
* [Quick start](#Quick-start)
* [DevLog](#DevLog)

## Overview
[ ] rewrite as catex
Deep neural networks (DNNs) of various architectures have become dominantly used in automatic speech recognition (ASR), which roughly can be classified into two approaches - the DNN-HMM hybrid and the end-to-end (E2E) approaches. DNN-HMM hybrid systems like [Kaldi](http://kaldi-asr.org/) and [RASR](http://www-i6.informatik.rwth-aachen.de/rwth-asr/) achieve the state-of-the-art performance in terms of recognition accuracy, usually measured by word error rate (WER) or character error rate (CER). End-to-end systems[^e2e](like [Eesen](https://github.com/yajiemiao/eesen) and [Espnet](https://github.com/espnet/espnet)) put simplicity of the training pipeline at a higher priority and usually are data-hungry. When comparing the hybrid and E2E approaches (modularity versus a single neural network, separate optimization versus joint optimization), it is worthwhile to note the pros and cons of each approach, as described in [2].

CAT aims at combining the advantages of the two kinds of ASR systems. CAT advocates discriminative training in the framework of [conditional random field](https://en.wikipedia.org/wiki/Conditional_random_field) (CRF), particularly with but not limited to [connectionist temporal classification]()(CTC) inspired state topology.

The recently developed [CTC-CRF](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ctc-crf.pdf) (namely CRF with CTC topology)  has achieved superior benchmarking performance with training data ranging from ~100 to ~1000 hours, while being end-to-end with simplified pipeline and being data-efficient in the sense that cheaply available language models (LMs) can be leveraged effectively with or without a pronunciation lexicon.

[^e2e]: End-to-end is in the sense that flat-start training of a single DNN in one stage, without using any previously trained models, forced alignments, or building state-tying decision trees, with or without a pronunciation lexicon.

Please cite CAT using:

[1] Hongyu Xiang, Zhijian Ou. CRF-based Single-stage Acoustic Modeling with CTC Topology. ICASSP, 2019. [pdf](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ctc-crf.pdf)

[2] Keyu An, Hongyu Xiang. Zhijian Ou. CRF-based ASR Toolkit. arXiv, 2019. [pdf](https://arxiv.org/abs/1911.08747) (More descriptions about the toolkit implementation)

[3] Keyu An, Hongyu Xiang. Zhijian Ou. CAT: A CTC-CRF based ASR Toolkit Bridging the Hybrid and the End-to-end Approaches towards Data Efficiency and Low Latency. INTERSPEECH, 2020. [pdf](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/is2020_CAT.pdf)

## Key Features
[ ] rewrite as catex
1. **CAT contains a full-fledged implementation of CTC-CRF.** 
   * A non-trivial issue is that the gradient in training CRFs is the difference between *empirical expectation* and *model expectation*, which both can be efficiently calculated by the forward-backward algorithm.
   * CAT modifies [warp-ctc](https://github.com/baidu-research/warp-ctc) for fast parallel calculation of the *empirical expectation*, which resembles the CTC forward-backward calculation.
   * CAT calculates the *model expectation* using CUDA C/C++ interface, drawing inspiration from Kaldi's implementation of denominator forward-backward calculation.

2. **CAT adopts PyTorch to build DNNs and do automatic gradient computation, and so inherits the power of PyTorch in handling DNNs.**

3. **CAT provides a complete workflow for CRF-based end-to-end speech recognition.**
   * CAT provides complete training and testing scripts for a number of Chinese and English benchmarks and all the experimental results reported in this paper can be readily reproduced. 
   * Detailed documentation and code comments are also provided in CAT, making it easy to get start and obtain state-of-the-art baseline results even for beginners of ASR.

4. **Evaluation results on major benchmarks such as Switchboard and Aishell show that CAT obtains the state-of-the-art results among existing end-to-end models with less parameters, and is competitive compared with the hybrid DNN-HMM models.**

5. **We add the support of streaming ASR**. To this end, we propose a new method called contextualized soft forgetting (CSF), which combines soft forgetting and context-sensitive-chunk in bidirectional LSTM (BLSTM). With contextualized soft forgetting, the chunk BLSTM based CTC-CRF with a latency of 300ms outperforms the whole-utterance BLSTM based CTC-CRF. See [pdf](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/is2020_CAT.pdf) for details.

## Citations

```
@inproceedings{xianghongyu2019ctccrf,
   author = {Hongyu Xiang, Zhijian Ou},
   booktitle = {Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
   organization = {IEEE},
   pages = {5676-5680},
   title = {CRF-based single-stage acoustic modeling with CTC topology},
   year = {2019}}
@misc{1911.08747,
   Author = {Keyu An and Hongyu Xiang and Zhijian Ou},
   Title = {CAT: CRF-based ASR Toolkit},
   Year = {2019},
   Eprint = {arXiv:1911.08747},
}
@article{an2020cat,
   title={CAT: A CTC-CRF based ASR toolkit bridging the hybrid and the end-to-end approaches towards data efficiency and low latency},
   author={An, Keyu and Xiang, Hongyu and Ou, Zhijian},
   journal={arXiv preprint arXiv:2005.13326},
   year={2020}
}
```

