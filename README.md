# Deep Segmentor - Sequence Segmentation using Joint RNN and Structured Prediction Models

We describe and analyze a simple and effective algorithm for sequence segmentation applied to speech processing tasks. We propose a neural architecture that is composed of two modules trained jointly: a recurrent neural network (RNN) module and a structured prediction model. The RNN outputs are considered as feature functions to the structured model. The overall model is trained with a structured loss function which can be designed to the given segmentation task. We demonstrate the effectiveness of our method by applying it to two simple tasks commonly used in phonetic studies: word segmentation and voice onset time segmentation.

If you find our work useful please cite: 
[Sequence Segmentation using Joint RNN and Structured Prediction Models] (https://arxiv.org/pdf/1610.07918v1.pdf)

```
@article{adi2016sequence,
  title={Sequence Segmentation Using Joint RNN and Structured Prediction Models},
  author={Adi, Yossi and Keshet, Joseph and Cibelli, Emily and Goldrick, Matthew},
  journal={arXiv preprint arXiv:1610.07918},
  year={2016}
}
```
