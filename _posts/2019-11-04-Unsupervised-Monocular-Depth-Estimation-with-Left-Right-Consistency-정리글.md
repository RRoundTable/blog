---
title: Unsupervised Monocular Depth Estimation with Left-Right Consistency 정리글
toc: true
badges: true
branch: master
categories: ['deeplearning', 'computer vision']
---

#  Unsupervised Monocular Depth Estimation with Left-Right Consistency 정리글



## Abstract

많은 방법론들이 depth estimation부분에서 성과를 보여주었다. 하지만,  대부분 지도학습이라는 한계를 가지고 있으며, 이는 결국 많은 수의 ground-truth data가 필요하다는 것을 의미한다. 하지만, depth를 기록하는 것은 매우 어려운 문제이다. 따라서 이 연구에서는 얻기 쉬운 binocular stereo footage를 이용하여 문제를 해결한다.

epipolar geometry constraints를 활용해서 reconstruction loss로 학습을 시킬 수 있다. 하지만 이 결과물은 depth image의 질이 낮아진다. 이러한 문제를 해결하기 위해서,  consistency를 유지할 수 있게하는 loss를 제안한다.



> **Epipolar geometry**
>
> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7c/Aufnahme_mit_zwei_Kameras.svg/250px-Aufnahme_mit_zwei_Kameras.svg.png">
>
> stereo vision의 geometry, 
>
> 2개의 각기 다른 위치에서 3D 이미지의 정보를 얻을 때. 3D points와 2D points간의 많은 기하학적 관계가 있다.

## Introduction

이전의 많은 연구들이 multiple observation이 가능하다는 가정아래에서 진행되었다. multiple viewpoint와 다른 조명조건에서의 data도 필요하다. 이러한 한계를 극복하기 위해서, supervised learning방식으로 mono depth estimation 방법론들이 연구되었다. 이러한 방법론들은 많은 수의 data를 기반으로 depth를 직접적으로 추정하고자 하였다. 많은 성과를 이루었지만, 데이터의 수가 많아야된다는 한계를 가지고 있다.(depth 관련 데이터를 수집하기는 힘들다)



image의 외관과 관계없이 장면의 모양을 이해할 수 있는 것은 machine perception분야에서 매우 중요한 이슈이다. 



사람은 monocular depth estimation을 잘하는데 이를 위해서 다음과 같은 힌트를 사용한다.

- 원근법
- 이미 알려진 물체크기로 상대적인 추정
- 조명 혹은 그림자 가려진 상태에서의 모양
- etc

위와 같은 top-down, bottom-up 힌트들을 조합해서 정확하게 depth를 추정할 수 있다. 이 연구에서는 depth data가 필요하지 않다. 학습과정에서는 synthesize depth를 이용한다.  두 이미지(left view, right view) 사이에서 해당 모델은 pixel level의 예측을 한다. (regression) 다른 연구에서도 이와 같은 방법론을 사용했지만 아래와 같은 한계가 존재한다.

- memory issue
- not fully diferentiable



## Related Work

### Learning-Based Stereo: between two images

대게 stereo estimation은 첫번째 이미지의 특정 픽셀과 두번째 이미지의 모든 pixel간의 similarity를 계산하는 알고리즘이다. 대부분의 stereo pairs는 정제되어 있으며, disparity estimation은 각 pixel의 1D search 문제이다. 

하지만, 최근 연구에 따르면,  head defined similarity measure방법론보다 matching function을 일종의 supervised learning problem으로 두고 문제를 해결하는 것이 더 좋은 성능을 보였다. 특히 Mayer et al.fully convolutional deep network를 활용하여 DispNet을 고안하였다.  이 방법론은 매우 많은 수의 ground-truth data가 필요했다.



### Supervised Single Image Depth Estimation

Saxena et al. patch based model([Make3D](http://make3d.cs.cornell.edu/))을 제안하였다. 이는 laser scan data를 이용하여 학습되었으며 prediction은 MRF를 활용하여 결합하였다. 이 방법론의 단점은 얇은 구조물을 modeling하는데 적절하지 않았다. 이는 현실적인 이미지를 만드는데 적절치 않다는 것을 의미한다.



위의 방법론은 hand tuning이 필요하다. 이런 방법론과 다르게, Liu et al은 CNN을 이용하여 이를 학습하고자 했다. 

Karsch et al. consistent한 image output을 가질려고 노력했다. 이는 training set으로부터 depth image를 복사하여 이루어졌다. 따라서 이 방법론은 test할 때, 모든 training set이 필요하다는 한계를 가진다.

Eigen et al.은 두개의 scale deep network를 활용해서 depth estimation이 가능하다는 것을 보였다. 이들은 hand craft feature를 사용하지 않았고, initial over segmentation을 이용하지 않았다. 대신에 raw pixel value를 활용하여, representation을 하였다.

많은 연구들이 CRF기반으로 accuracy향상을 이루었다.



### Unsupervised Depth Estimation



Flynne et al.의 DeepStereo는 새로운 view의 이미지를 만들어낸다. 학습하는동안,  다양한 카메라의 상대적인 pose가 근처 이미지의 모습을 만들어내는데 사용된다. test할 때, image synthesis는 겹치는 작은 patch에서 작동한다. 이 모델은 근처의 다른 posed image의 view가 필요하므로,  monocular depth estimation에 부적절하다.

Xie et al의 Deep3D의 목표는 left image 기반으로 right view이미지를 만들어내는 것이다. reconstruction loss를 활용하며,  각 픽셀에 대해서 가능한 disparities에 대해서 분포를 만들어낸다. 이 방법론의 단점은 scalable하지 않다는 것이다. 가능한 disparities가 많아질수록 많은 memory가 필요하다.

해당연구와 비슷한 연구는 Garg et al이 제안하였다. 이 nework는 monocular depth estimation방법론이긴 하지만, fully differentiable하지 않다는 한계를 가진다. 이런 점을 극복하기 위해서 taylor approximation을 사용하였다.





## Method

### 1. Depth Estimation as Image Reconstruction

해당 연구에서는 직접 depth를 추정하는 것이 아니라 image reconstruction을 이용하여 추정한다. 기존의 연구에서는 depth estimation문제를 supervised task로 인식해서 해결했지만, 앞서 설명했듯이 depth ground truth data를 구하는 것은 매우 힘든 일이다. 비싼 하드웨어를 사용하더라고 실제환경에서는 정확하지 않을 수 있다.

이 연구에서는 training과정에서 image reconstruction을 활용하여 해결한다. 메인 아이디어는 left-view image에서 right-view image를 만들 수 있는function을 구할 수 있다면, 3D shape에 대한 지식을 알고 있는 것이라고 볼 수 있다.(right view -> left-view도 마찬가지)

left-view image로부터 right-view image를 만든다음에 각 픽셀이 가지는 depth value를 추정한다.

> Image disparity
>
>  https://www.quora.com/What-are-disparity-maps-and-How-are-they-created 
>
> - baseline distance between the camera and the camera focal length $f$: $b$
> - image disparity: $d$
> - depth $\hat{d} = bf/d$



### 2.  Depth Estimation Network 

> Disparity map
>
> <img src="https://i.stack.imgur.com/tECoA.png">

이 연구에서는 left-to-right , right-to-left disparities를 모두 구할 수 있으며, 서로 consistency를 유지하게 함으로써 더 좋은 성능을 가져온다.  아래는 기존의 연구의 architecture와 해당 연구의 architecture를 보여주고 있다.

<img src="https://user-images.githubusercontent.com/27891090/68211370-c6a3e180-001a-11ea-944e-c70cf85c7cad.png">

오른쪽 이미지를 보면, left image를 가지고 right image를 생성해낸다. 하지만, 우리는 right-view image에서 sampling된 left-view image가 필요하다.  No LR을 보면 right-view image에서 left view image를 생성한다. 하지만 이는 texture-copy artifact라는 현상을 보이며, detph가 연속적이지 않은 부분에서 error가 많이 발생한다. 아래의 이미지를 보면 알 수 있다.

<img src="https://user-images.githubusercontent.com/27891090/68212304-9bba8d00-001c-11ea-8cb0-9753fca7f09a.png">



이 연구에서는 이 문제를 model이 두가지 disparity map를 생성해내도록 만들어서 해결하였다. (sampling from the opposite input images) 이 방법은 여전히 left-view image하나만 필요하며, right image는 training과정에서만 사용된다. left-right consistency loss를 이용하여, 더 정확한 정확도를 가질 수 있다.



### 3. Training Loss


$$
C = \Sigma_{s=1}^4C_s
$$

$$
C_{s} = \alpha_{ap}(C_{ap}^l + C_{ap}^r) + \alpha_{ds}(C_{ds}^l + C_{ds}^r) + \alpha_{lr}(C_{lr}^l + C_{lr}^r)
$$


- Apperance Matching Loss: $C_{ap}$
- Disparity Smoothness Loss: $C_{ds}$
- Left-Right Disparity Consistency: $C_{lr}$

<img src="https://user-images.githubusercontent.com/27891090/68212839-b6413600-001d-11ea-89cf-2e89ac8071a2.png">

**Apperance Matching Loss**는 oposite setero이미지로부터 image를 생성하도록 학습시킨다. 여기서는 spatial transformer network를 사용하여 disparity map을 만든다. 여기서 bilinear sampler를 사용하는데 locally fully differentiable하며 fully convolutional architecture를 가진다.


$$
C_{ap}^l = \frac{1}{N}\Sigma_{i, j}\alpha\frac{1 -SSIM(I_{ij}^l, \hat{I}_{i,j}^l)}{2} + (1 - \alpha)\rVert I_{ij}^l - \hat{I}_{i,j}^l \rVert
$$
이 연구에서는 $\alpha$ 는 0.85 SSIMd에서는 3 x 3 block filter를 사용했다.

> SSIM: structural similarity
>
> 사람의 시각 시스템은 이미지에서 구조 정보를 도출하는데 특화되어 있기 때문에 구조 정보의 왜곡정도가 지각 품질에 가장 큰 영향을 미친다. 이것이 SSIM의 기본이 되는 핵심가설이다. 구체적으로는 원본 이미지 x와 왜곡 이미지 y의 brightness, contrast, structure를 비교한다.
>
> - brightness
>
>   $u_x$: x 이미지의 평균밝기
>
>   $u_y$: y 이미지의 평균 밝기
>
>   $I(x, y) = \frac{2u_xu_y + C_1}{u_x^2 + u_y^2 + C_2}$
>
> - contrast
>
>   $C(x, y) = \frac{2\sigma_x\sigma_y+C_2}{\sigma_x^2 + \sigma_y^2 + C_2}$
>
> - structure
>
>   structure = $\frac{x-u_x}{\sigma_x}$
>
>   $S(x, y) = \frac{\sigma_{xy} + C_3}{\sigma_x\sigma_y + C_3}$  
>
> - SSIM
>
>   $SSIM(x,y) = I(x,y)C(c,y)S(x,y)$ 
>
>   $SSIM(x, y)= \frac{(2u_xu_y + C_1) (2\sigma_{xy} +C_2)}{(u_x^2 + u_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 +C_2)}$

**Disparity Smoothness Loss**의 경우에는 아래와 같이 수식으로 표현된다. 이는 disparities가 locally smooth하게 하는 효과를 가져온다.
$$
D_{ds}^l = \frac{1}{N}\Sigma_{i,j}\rvert \partial_{x}d_{i,j}^l\rvert e ^{\rVert \partial_{x}I_{i,j}\rVert} + \rvert \partial_y d_{i,j}^l \rvert e ^{\rVert\partial_y I_{i,j}\rVert}
$$

**Left-Right Disparity Consistency Loss** 는 더 정확한 disparity map을 만들기 위한 term이다.  이 term은 left-view disparity map을 projected right-view disparity map과 동일하게 만들어주는 역할을 한다.
$$
C_{lr} ^l = \frac{1}{N}\Sigma_{i,j}\rvert d_{ij}^l - d_{ij+d_{ij}^l} ^r \rvert
$$
