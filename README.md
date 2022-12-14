# A deep recursive multi-scale feature fusion network for image super-resolution
This repository is Pytorch code for our proposed DRMSFFN.
![figure2](https://user-images.githubusercontent.com/42378133/206623801-3c82333f-b307-42e8-8800-bed3e71dbf3d.png)
The structure of the proposed Deep Recursive Multi-Scale Feature Fusion Network (DRMSFFN).
![figure3](https://user-images.githubusercontent.com/42378133/206623805-4143d8c8-a1aa-41af-bcac-9ac7ee5d4a6f.png)
Structure of the Recursive Multi-Scale Feature Fusion Block (RMSFFB).

The details about our proposed FRN can be found in our main paper:https://www.sciencedirect.com/science/article/pii/S1047320322002504

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

    python test.py -opt options/test/test_FRN_x2.json
    python test.py -opt options/test/test_FRN_x3.json
    python test.py -opt options/test/test_FRN_x4.json

    Finally, PSNR/SSIM values for Set5 are shown on your screen, you can find the reconstruction images in ./results. Other standard SR benchmark dadasets, you need to
    change the datasets storage path in the test_FRN_x2.json, test_FRN_x3.json and test_FRN_x4.json files.
    
## Results:








