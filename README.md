# LInKs-3D-Human-Pose-Estimation
This repository contains the code and weights of the lifting models used within the LInKs “Lifting Independent Keypoints” - Partial Pose Lifting for Occlusion
Handling with Improved Accuracy in 2D-3D Human Pose Estimation paper.

First, we want to share our appreciation for Bastian Wandt and his help in answering questions during this research.
This work was heavily inspired by Elepose and our code was based upon his obtainable at https://github.com/bastianwandt/ElePose we recommend checking it out.

To train the model or evaluate our weights on the Human3.6M dataset it requires first downloading the dataset with H36m-fetch (https://github.com/anibali/h36m-fetch).
Once you have downloaded the dataset, extracted and preprocessed it you can use our pre_process_h36m_fetch.py file to make it useable with our data loader.
