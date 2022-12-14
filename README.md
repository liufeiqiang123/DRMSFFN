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
Quantitative Results:![Quantitative Results](https://user-images.githubusercontent.com/42378133/207483168-2570d986-637d-42b1-888a-572775799ca0.png)
Quantitative comparison for scale factors ×2, ×3 and ×4 of the proposed method DRMSFFN with state-of-the-art methods. The best and the second best results are indicated in bold and underlined, respectively.

Some Qualitative Results:
![FIGURE6](https://user-images.githubusercontent.com/42378133/207484288-5fcfd44d-9948-4ce7-a6cc-d049d098d5b6.png)
Visual comparison of the results of the proposed method DRMSFFN with those of other state-of-the-art methods on some images from the Set14 and B100 datasets for ×4
SR. The best results are indicated in bold.
![FIGURE7](https://user-images.githubusercontent.com/42378133/207484774-abcde8ec-6867-4728-bab0-7913ed2e0baf.png)
Visual comparison of the results of our DRMSFFN with those of other state-of-the-art methods on some images from the Urban100 and Manga109 datasets for ×4 SR. The
best results are indicated in bold.

Running time comparison
![FIGURE8](https://user-images.githubusercontent.com/42378133/207485430-ba745663-8cba-401c-9481-7a3762fc36f2.png)
The running time, number of parameters and performance of different SISR methods. The results denote the mean PSNR and running time for scale factor x4 on the Set5 dataset.
