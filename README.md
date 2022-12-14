# A deep recursive multi-scale feature fusion network for image super-resolution
This repository is Pytorch code for our proposed DRMSFFN.
![figure2](https://user-images.githubusercontent.com/42378133/206623801-3c82333f-b307-42e8-8800-bed3e71dbf3d.png)
The structure of the proposed Deep Recursive Multi-Scale Feature Fusion Network (DRMSFFN).
![figure3](https://user-images.githubusercontent.com/42378133/206623805-4143d8c8-a1aa-41af-bcac-9ac7ee5d4a6f.png)
Structure of the Recursive Multi-Scale Feature Fusion Block (RMSFFB).

The details about our proposed DRMSFFN can be found in our main paper:https://www.sciencedirect.com/science/article/pii/S1047320322002504

If you find our work useful in your research or publications, please star the code and consider citing:

    @article{liu2022deep,
      title={A deep recursive multi-scale feature fusion network for image super-resolution},
      author={Liu, Feiqiang and Yang, Xiaomin and De Baets, Bernard},
      journal={Journal of Visual Communication and Image Representation},
      pages={103730},
      year={2022},
      publisher={Elsevier}
    }

## Requirements:

    1. Python==3.6 (Anaconda is recommended)
    2. skimage
    3. imageio
    4. Pytorch==1.2
    5. tqdm
    6. pandas
    7. cv2 (pip install opencv-python)

## Test:

    python test.py -opt options/test/test_DRMSFFN_x2.json
    python test.py -opt options/test/test_DRMSFFN_x3.json
    python test.py -opt options/test/test_DRMSFFN_x4.json

    Finally, PSNR/SSIM values for Set5 are shown on your screen, you can find the reconstruction images in ./results. Other standard SR benchmark dadasets, you need to
    change the datasets storage path in the test_DRMSFFN_x2.json, test_DRMSFFN_x3.json and test_DRMSFFN_x4.json files.
    
## Results:
Quantitative Results:![Quantitative Results](https://user-images.githubusercontent.com/42378133/207537886-e01230e1-a428-464f-a82f-21e7395d1bad.png)
Quantitative comparison for scale factors ×2, ×3 and ×4 of the proposed method DRMSFFN with state-of-the-art methods. The best and the second best results are indicated in bold and underlined, respectively.

Some Qualitative Results:
![figure6](https://user-images.githubusercontent.com/42378133/207536546-8ae5d527-2ca0-4904-ae52-1ea6cc5db028.png)
Visual comparison of the results of the proposed method DRMSFFN with those of other state-of-the-art methods on some images from the Set14 and B100 datasets for ×4
SR. The best results are indicated in bold.

![figure7](https://user-images.githubusercontent.com/42378133/207536598-b7e39d48-b38a-4ffe-8f87-a9691278c2ce.png)
Visual comparison of the results of our DRMSFFN with those of other state-of-the-art methods on some images from the Urban100 and Manga109 datasets for ×4 SR. The
best results are indicated in bold.

Running time comparison
![figure8](https://user-images.githubusercontent.com/42378133/207536640-f77e62af-fd6e-4739-a989-8f4cfec30401.png)
The running time, number of parameters and performance of different SISR methods. The results denote the mean PSNR and running time for scale factor x4 on the Set5 dataset.
