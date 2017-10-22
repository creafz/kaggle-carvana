The solution for the Carvana Image Masking Challenge on Kaggle. It uses a custom version of RefineNet with Squeeze-and-Excitation modules implemented in PyTorch. It was a part of the final ensemble that was ranked 23 out of 735 teams (top 4%).

### About the Competition
The goal of the [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge) was to develop an algorithm that removes a background from a wide variety of car photos. Here you can see predictions from a trained neural network for 16 images of a single car.

![Neural network predictions](https://raw.githubusercontent.com/creafz/kaggle-carvana/master/img/example_predictions.gif)

### About the model
RefineNet is a convolutional neural network developed for semantic segmentation. For more details see ["RefineNet: Multi-Path Refinement Networks with Identity Mappings for High-Resolution Semantic Segmentation"](https://arxiv.org/abs/1611.06612) by Guosheng Lin, Anton Milan, Chunhua Shen and Ian Reid.

As described in the original paper, Squeeze-and-Excitation blocks "adaptively recalibrate channel-wise feature responses by explicitly modelling interdependencies between channels. These blocks were foundation of ILSVRC 2017 classification submission which
won first place and achieved a ∼25% relative improvement over the
winning entry of 2016." For more details see ["Squeeze-and-Excitation Networks"](https://arxiv.org/abs/1709.01507) by Jie Hu, Li Shen and Gang Sun.

[My version of RefineNet](https://github.com/creafz/kaggle-carvana/blob/master/models/se_refinenet.py) based on [this PyTorch implementation](https://github.com/thomasjpfan/pytorch_refinenet) of the original paper by Thomas Fan.

### Requirements
- Python 3.6
- PyTorch 0.2
- TensorBoard or Crayon server
- Libraries from requirements.txt

### To run the code
- Adjust config variables in config.py
- Execute run.sh file
