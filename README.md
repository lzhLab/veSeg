# veSeg
Compensation of small data with large filters for accurate liver vessel segmentation from contrast-enhanced CT images.

## Introduction
Segmenting liver vessels from contrast-enhanced computed tomography images is essential for diagnosing liver diseases, planning surgeries, and delivering radiotherapy. Nevertheless, identifying vessels is a challenging task due to the tiny cross-sectional areas occupied by vessels, which has posed great challenges for vessel segmentation, such as limited features to be learned and difficulty in constructing high-quality as well as large-volume data.
    We present an approach that only requires a few labeled vessels but delivers significantly improved results. Our model starts with vessel enhancement by fading out liver intensity and generates candidate vessels by a classifier fed with a large number of image filters. Afterwards, the initial segmentation is refined using Markov random fields.
    In experiments on the well-known dataset 3D-IRCADb, the averaged Dice coefficient is lifted to 0.63, and the mean sensitivity is increased to 0.71. These results are significantly better than those obtained from existing machine-learning approaches and comparable to those generated from deep-learning models.
    Sophisticated integration of large number filters is able to pinpoint effective features from liver images that are sufficient to distinguish vessels from other liver tissues under a scarcity of large-volume labeled data. The study can shed light on medical image segmentation, especially for those without sufficient data.

The filters used in this study include:
- CLAHE [1]
- Gabor [2]
- Gamma [3]
- Gaussian [4]
- Hessian [5]
- Laplacian [6]
- Median [7]
- Mean [8]
- Minimum [9]
- Bilateral [10]
- Sobel [11]
- Canny [12]
- Pillow [13]
  
# Reference
1. Kuran, U., Kuran, E.C.: Parameter selection for clahe using multi-objective cuckoo search algorithm for image contrast enhancement. Intelligent Systems with Applications 12, 200051 (2021)
2. Mehrotra, R., Namuduri, K.R., Ranganathan, N.: Gabor filter-based edge detection. Pattern recognition 25(12), 1479–1494 (1992)
3. Rahman, S., Rahman, M.M., Abdullah-Al-Wadud, M., Al-Quaderi, G.D., Shoyaib, M.: An adaptive gamma correction for image enhancement. EURASIP Journal on Image and Video Processing 2016(1), 1–13 (2016)
4. Reddy, K.S., Jaya, T.: De-noising and enhancement of MRI medical images using gaussian filter and histogram equalization. Materials Today: Proceedings (2021)
5. Frangi, A.F., Niessen, W.J., Vincken, K.L., Viergever, M.A.: Multiscale vessel enhancement filtering. In: International Conference on Medical Image Computing and Computer-assisted Intervention, pp. 130–137 (1998). Springer
6. Zunair, H., Ben Hamza, A.: Sharp U-Net: Depthwise convolutional network for biomedical image segmentation. Computers in Biology and Medicine 136, 104699 (2021)
7. Wu, C.H., Shi, Z.X., Govindaraju, V.: Fingerprint image enhancement method using directional median filter. In: Biometric Technology for Human Identification, vol. 5404, pp. 66–75 (2004). International Society for Optics and Photonics
8. Janani, P., Premaladha, J., Ravichandran, K.: Image enhancement techniques: A study. Indian Journal of Science and Technology 8(22), 1–12 (2015)
9. Chen, H., Li, A., Kaufman, L., Hale, J.: A fast filtering algorithm for image enhancement. IEEE transactions on medical imaging 13(3), 557–564 (1994)
10. Geng, J., Jiang, W., Deng, X.: Multi-scale deep feature learning network with bilateral filtering for SAR image classification. ISPRS Journal of Photogrammetry and Remote Sensing 167, 201–213 (2020)
11. Nguyen, T.P., Chae, D.-S., Park, S.-J., Yoon, J.: A novel approach for evaluating bone mineral density of hips based on sobel gradient-based map of radiographs utilizing convolutional neural network. Computers in Biology and Medicine 132, 104298 (2021)
12. Shokhan, M.: An efficient approach for improving canny edge detection algorithm. International journal of advances in engineering & technology 7(1), 59 (2014)
13. Clark, A.: Pillow (pil fork) documentation. Readthedocs. https://buildmedia.readthedocs.org/media/pdf/pillow/latest/pillow.pdf (2015)
