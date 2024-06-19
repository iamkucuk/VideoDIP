# VIDEO DECOMPOSITION PRIOR: EDITING VIDEOS LAYER BY LAYER

Paper Authors: 

| Name | Shcool | Mail |
| ---- | ------ | ---- |
| Gaurav Shrivastava | University of Maryland, College Park | gauravsh@umd.edu|
| Abhinav Shrivastava | University of Maryland, College Park | abhinav@cs.umd.edu
| Ser-Nam Lim | University of Central Florida | sernam@ucf.edu |


Project Authors: Alper Bahçekapılı, Furkan Küçük


Paper Summary: Paper is a deep learning framework to edit videos without supervision. Namely following three points are adressed in the paper:

* Video Relighting
* Video Dehazing
* Unsupervised Video Object Segmentation



## Overall Logic of the Paper:

Paper approaches the problem with the intuition from video editing programs. As in these programs, they treat the videos as they are composed of multiple layers. For relighting problem, one layer is relight-map and the other is dark frame. In dehazing, again similar to reliht one layer is t-map etc. For segmentation, layers are foreground objects layer and the other is background layer. 

All optimization is done in the inference time. So for each of the video, we train models from the ground up. Paper realize given solutions with two main modules. RGB-net and $\alpha$-net models. For each of the problem type, these models quantity(1 RGB-net for relight, 2 RGB-net for segmentation) and purpose change. 

These models harness the information that is obtained by flow between the frames. Inclusion of optical flow captures motion effectively and makes the model significantly moer resilient to variations in lighting.

## Modules Overview

**RGBnet:** Given that we only optimize the weights over a single video, a shallow convolutional U-Net is sufficent for the task. This model takes $X_t$ of the video seq. and outputs RGB layer. 

**$\alpha$ Layer:** Similar to RGBNet arcitecture is again shallow U-Net for predicting the t-maps or opacity layer. This layer takes RGB representation of the forward optical flow($F^{RGB}_{t\rightarrow t-1}$) 

## Video Relighting

By manipulating the shading and lighting layers, the method allows for changing the illumination conditions of a video. This can be used to simulate different times of day, weather conditions, or artificial lighting effects.
The relighting process involves adjusting the lighting layer to achieve the desired illumination effect while maintaining the natural appearance and coherence of the video.

The goal of the video relighting task is to recover video ${\{(X_t^{out})\}^T_1}$ with good lighting conditions from its poorly lit pair $\{X_t^{in}\}^T_1$. Authors model transmittion map ($1 / A_t$) with the $\alpha$-net model. They model $\gamma^{-1}$ as a learnable parameter where it can take values from the range (0,1). Pipeline is given by the Figure 1.

<center>
    <figure>
        <img src="figures/figure-1.png" alt="Video Relighting" title="Figure 1" width="1000">
        <figcaption>Figure 1: Video Relighting</figcaption>
    </figure>
</center>

$F^{(1)}_{RGB}$, $F^{(1)}_{\alpha}$, $\gamma^{-1}$ are optimized with the following loss objectives(below are general definition of the losses. Each module updates these a little)

**Overall Loss Objective:** $L_{final}$ = $\lambda_{rec}$ $L_{rec}$ + $\lambda_{warp}$ $L_{warp}$ (1)

**Reconstruction Loss:** $\sum_t ||X_t - \hat{X_t}||_1 + || \phi (X_t) - \phi (\hat{X_t})||_1$ (2)

**Optical Flow Warp Loss** $\sum_t || F_{t-1 \rightarrow t} (X_{t-1}^o) -  X_{t}^o  || $ (3)



Relit video is reconstructed with the following equation.

$X_t^{out} = A_t \odot  (X_t^{in})^{\gamma}$,  $\forall t \in (1,T] $ (4)


For the VDP framework authors update eq. 4 as follows

$log(X_t^{in}) = \gamma^{-1}(log(1/A_t)+log(x_t^{out}))$, $\forall t \in (1,T] $ (5)




Relighting task is evaluated on SDSD dataset where the video has relit and dark version of these. SSIM and PSNR metrics are utilized in order to evaluate quantatively. You can see an example from the SDSD dataset in the Figure 2


<center>
    <figure>
        <img src="figures/figure-2_1.png" alt="Dark Counterpart" title="Figure 2_1" width="40%">
        <img src="figures/figure-2_2.png" alt="Relit Counterpart" title="Figure 2_1" width="40%">
        <figcaption>Figure 2: SDSD Dataset Example: Relit version of the image is on the left. Right part is the darker counterpart</figcaption>
    </figure>
</center>


Eventough paper did not explain following metrics in detail, we believe it is important to exmplain them a little. They are used in dehazing task as well: 



**PSNR Metric:** The PSNR (Peak Signal-to-Noise Ratio) metric is a widely used quantitative measure for evaluating the quality of reconstructed or processed images and videos compared to their original versions. PSNR is expressed in decibels (dB). Higher PSNR values indicate better quality of the reconstructed or processed image/video compared to the original. 

**SSIM Metric:** The SSIM (Structural Similarity Index Measure) is a metric used to measure the similarity between two images. Unlike PSNR, which focuses on pixel-level differences, SSIM considers changes in structural information, luminance, and contrast, providing a more comprehensive assessment of perceived image quality. The SSIM index can range from -1 to 1, where 1 indicates perfect similarity.



## Unsupervised Video Object Segmentation

Given the input video, target of the unsupervised video object segmentation is to segment out the main object in the video. Note that in any stage any human annotations are not needed. They start by $\alpha$-blending equation to write the reconstruction of the input video.
 
$X_t = \sum_{i=1}^{L} M_t^i \odot f_{\text{RGB}}^i(X_t) \quad \forall t \in (1, T]$ (6)

Here L is the number of layers(number of masks generated by $\alpha$-net) Here $M_t^i$ denotes the alpha map for the ith object layer and tth frame. And it is obrained as follows:

$M_t^i = f_{\alpha}^i(F_{t-1 \to t}^{\text{RGB}}) \quad \forall t \in (1, n]$ (7)

where $F_{t-1 \to t}^{\text{RGB}}$ is the flow estimate from t-1 to t. Additionally followin constraint should also satisfy:

$J_{h,w} = \sum_{i=1}^{L} M_t^i$ (8)

where $J_{h,w}$ denotes all-ones matrix of size [h,w]


However video decomposition problem is ill-posed and loss equation (1) is not strong enough to find a visually plausible layered decomposition of an input video.


$ L_{\text{sim}} = \frac{\phi(M \circ F^{\text{RGB}}) \cdot \phi((1 - M) \circ F^{\text{RGB}})}{\|\phi(M \circ F^{\text{RGB}})\| \| \phi((1 - M) \circ F^{\text{RGB}})\|} $ (9)


$\mathcal{L}_{\text{Mask}} = \sum \left( M_t^i - 0.5 \right)^{-1}$ (10)


$L_{\text{layer}} = \| M_i \circ X_t - M_i \circ X_{t+1} \|$ (11)


$ L_{\text{UVOS}} = \lambda_{\text{rec}}L_{\text{rec}} + \lambda_{\text{sim}}L_{\text{sim}} + \lambda_{\text{layer}}L_{\text{layer}} + \lambda_{\text{warp}}L_{\text{warp}} + \lambda_{\text{mask}}L_{\text{mask}} $ (12)


$X_i = M_{t} \odot  f_{\text{RGB}}^1(X_t) + (1 - M_{t}) \odot  f_{\text{RGB}}^2(X_t) \forall i \in [1,n]. $ (13)

$ X_t = \alpha \odot Clr(X_t) + (1-\alpha) \odot A_t $ (14)












## Video Dehazing


## Conclusion


