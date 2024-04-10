

# MNIST Classification
For Assignment 1 in deep learning class, I attempted to classify the MNIST dataset using both the LeNet-5 and MLP (Multilayer Perceptron) models.

## 1. How to use
```python 
python main.py
```
Using the provided code and command, you can train both the LeNet-5 and MLP models sequentially. Once the LeNet-5 training concludes, the training for the MLP model begins right away.

## 2. Requirement
### 2.1. The number of the models
The instructions specified that the LeNet-5 and MLP models should have an equal number of parameters. Below are the details regarding the number of parameters for each model.

###### [NOTE] The input shape of the models is `(1, 32, 32)`. To align the LeNet-5 architecture from the lecture notes, I transformed the shape of all input images using `torchvision.transforms`

- LeNet-5

  - `self.c1`: (1 * 5 * 5 + 1) * 6 = 156
  - `self.s2`: 0
  - `self.c3`: (6 * 5 * 5 + 1) * 16 = 2,416
  - `self.s4`: 0
  - `self.c5`: (16 * 5 * 5 + 1) * 120 = 48,120
  - `self.fc6`: 120 * 84 = 10,080
  - `self.output`: 84 * 10 = 840

  -> Total parameters: **61,612**

- MLP

  - `self.fc1`: 1024 * 50 + 1= 51,200
  - `self.fc2`: 50 * 64 + 1 = 3,200
  - `self.fc3`: 64 * 32 + 1 = 2,048
  - `self.fc4`: 32 * 32 + 1 = 1,024
  - `self.fc5`: 32 * 32 + 1 = 1,024
  - `self.fc6`: 32 * 32 + 1 = 1,024
  - `self.fc7`: 32 * 32 + 1 = 1,024
  - `self.fc8`: 32 * 32 + 1 = 1,024
  - `self.fc9`: 32 * 10 + 1 = 320

  -> Total parameters: **61,897**


### 2.2. Training & Testing Statistics
The next step involves plotting the training and testing statistics. As detailed in the `main.py` file, I recorded the loss values and accuracies for both training and testing phases at the end of each epoch.

<table style="margin-left: auto; margin-right: auto;">
    <tr>
        <td>
            <div style="text-align: center;">
                Average Loss value
            </div>
            <img src='https://github.com/drizzle0171/Greenyyy/assets/90444862/41482015-f24b-4952-a2a3-d3cef741cb8c' width=400>
        </td>
        <td>
            <div style="text-align: center;">
                Accuracay
            </div>
            <img src='https://github.com/drizzle0171/Greenyyy/assets/90444862/f2e88ca3-3acf-49a3-9ae8-04589026c07a' width=400>
        </td>
        </tr>   
</table>

As observed, LeNet-5 outperforms the MLP in handling image data, demonstrated by its significantly lower initial and overall loss values. This superiority indicates that convolution layers, which calculate using adjacent information and share parameters, are more suited for image data compared to MLP, where all the parameters are independent. Additionally, the training loss curve of the LeNet-5 is smoother, highlightning the advantage of convolution layers in preserving the spatial information of the image data. In contrast, the fully-connected layers perceive images as flat vectors, losing out on capturing spatial details.

### 2.3. Regularization technique

To enhance LeNet-5's performance, I implemented **early stopping** and **L2 regularization using AdamW** to regularize the model's weight. 

<table style="margin-left: auto; margin-right: auto;">
    <tr>
        <td>
            <div style="text-align: center;">
                Average Loss value w/o regularization
            </div>
            <img src='https://github.com/drizzle0171/Greenyyy/assets/90444862/41482015-f24b-4952-a2a3-d3cef741cb8c' width=400>
        </td>
        <td>
            <div style="text-align: center;">
                Average Loss value w/ regularization
            </div>
            <img src='https://github.com/tml-epfl/llm-adaptive-attacks/assets/90444862/fbe00603-4592-4384-9239-cbd5c6bff763' width=400>
        </td>
        </tr>   
</table>


<table style="margin-left: auto; margin-right: auto;">
    <tr>
        <td>
            <div style="text-align: center;">
                Accuracy w/o regularization
            </div>
            <img src='https://github.com/drizzle0171/Greenyyy/assets/90444862/f2e88ca3-3acf-49a3-9ae8-04589026c07a' width=400>
        </td>
        <td>
            <div style="text-align: center;">
                Accuracy w/ regularization
            </div>
            <img src='https://github.com/tml-epfl/llm-adaptive-attacks/assets/90444862/bed1722b-e59b-4d4b-9095-c590d934f259' width=400>
        </td>
        </tr>   
</table>

As you can see, LeNet-5's test loss was getting higher without regularization. However, with the introduction of early stopping and AdamW, the increase in test loss was effectively arrested before it could climb higher. This demonstrates that the use of regularization techniques significantly contributed to the model's enhancements.