# A Collaborative Microphone Clustering Framework for Multi-Task Distributed Microphone Arrays

## Abstract
In large-scale distributed microphone array (DMA), microphone clustering is a key step to save resource usage for downstream tasks like speech separation (SS), speech enhancement (SE) and speaker verification (SV). This was often done in literature by gathering raw audio data from all microphones, which causes a high resource consumption. In this work, we propose an end-cloud collaborative microphone clustering framework. First, we exploit a convolution recurrent neural network (CRNN) to extract speaker embedding, which is computed at the microphone side. We design a microphone counting module for each device to estimate the number of active sources, which was usually assumed to be known. The counting solution and speech feature vector instead of full waveform are then sent to the fusion center, which applies fuzzy C-means algorithm for microphone clustering. Besides, for each source-dominant cluster we propose to use generalized cross-correlation with phase transform (GCC-PHAT) to select a reference microphone, which is required by cluster-based SE and SS. 
Results show the superiority of CRNN over other features, the efficacy of individual modules as well as the applicability of the proposed method to DMA-based SS, SE and SV tasks.

## our LABNet
### train
```bash
python train.py
```
### test
```bash
python test.py
```

## Contact
If you have any questions, please feel free to contact us at: **hyyan@mail.ustc.edu.cn** or **cqjiang@mail.ustc.edu.cn**
