# veSeg
Compensation of small data with large filters for accurate liver vessel segmentation from contrast-enhanced CT images.

The filters used in this study include:
		CLAHE <a href="ref1">1</a>
		Gabor \cite{mehrotra1992gabor} 
		Gamma \cite{rahman2016adaptive} 
		Gaussian \cite{SRINIVASAREDDY2021} 
		Hessian \cite{frangi1998multiscale} 
		Laplacian \cite{ZUNAIR2021104699} 
		Median \cite{wu2004fingerprint} 
		Mean \cite{janani2015image} 
		Minimum \cite{chen1994fast}
		Bilateral \cite{GENG2020201} 
		Sobel \cite{NGUYEN2021104298} 
		Canny \cite{shokhan2014efficient} 
		Pillow (imageFilter) \cite{clark2015pillow}
# Reference
1. <a name="ref1">Kuran, U., Kuran, E.C.: Parameter selection for clahe using multi-objective cuckoo search algorithm for image contrast enhancement. Intelligent Systems with Applications 12, 200051 (2021)</a>
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
13. Clark, A.: Pillow (pil fork) documentation. Readthedocs. https://buildmedia. readthedocs. org/media/pdf/pillow/latest/pillow. pdf (2015)
