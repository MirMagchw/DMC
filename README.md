# A Collaborative Microphone Clustering Framework for Multi-Task Distributed Microphone Arrays

## Abstract
In large-scale distributed microphone arrays (DMAs), microphone clustering (MC) is a key step to save resource usage for downstream tasks like speech separation (SS), speech enhancement (SE) and speaker verification (SV). 
In this work, we propose an end-cloud collaborative MC framework. First, we exploit a convolution recurrent neural network (CRNN) to extract speaker embeddings at the end side. Several other extractors are also discussed. We design a speaker counting module for each device to estimate the number of active speakers, which was usually assumed to be known. The counting solution and speech feature vector instead of full waveform are then sent to the cloud side, which applies fuzzy C-means algorithm for MC. Besides, for each speaker-dominant cluster we propose to use generalized cross-correlation with phase transform (GCC-PHAT) to select a reference microphone, which is required by cluster-based SE and SS. 
Results show the superiority of CRNN over other models, the efficacy of individual modules as well as the applicability of the proposed method to DMA-based SS, SE and SV tasks.


