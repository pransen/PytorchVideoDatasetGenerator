# PytorchVideoDatasetGenerator
Implementation of Video Frame Generator in PyTorch. This can be used to train CNN LSTM models, 3D-CNN models and other architectures where both spatial and temporal features are required to be captured.

## Usage
The expected directory structure of the dataset is as follows:

    ROOT DIR
    ├── label1                   
    │   ├── video1             
    │   ├── video2              
    │   ├── video3
    │   ├── ...
    │                
    ├── label2  
    │   ├── video1             
    │   ├── video2              
    │   ├── ...
    │
    ├── label3
        ├── video1
        ├── ...
    
       
* First, run python3 utils/dataset_util.py --root_dir <path to the ROOT_DIR above>. This would create a data.csv file containing the paths to the individual videos, the associated label and the number of frames in each video

* For using video_dataset.py, refer to the file for an example.
