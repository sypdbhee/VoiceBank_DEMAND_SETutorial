# VoiceBank_DEMAND_SETutorial
Signal channel speech enhancement tutorial on VoiceBank+DEMAND database 

In this tutorial, each DNN, FCN, LSTM, or BLSTM model is implemented to perform a speech enhancement system. It is worth noting that the DNN, LSTM, and BLSTM model structures used for the VoiceBank-DEMAND dataset were not optimized. Instead, the FCN model is reimplemented based on [research](https://ieeexplore.ieee.org/document/8281993):

S.-W. Fu, Y. Tsao, X. Lu, and H. Kawai, "Raw waveform-based speech enhancement via fully convolutional networks," Proc. APSIPA ASC, 2017.

***
### Usage:

**1. List preparation:**
```
python VCTK_list_prepared.py --mode train (train_noisy_folder_path) (train_clean_folder_path) .;
```
```
python VCTK_list_prepared.py --mode test (test_noisy_folder_path) (test_clean_folder_path) .;
```
   
**2. Training:**
```
CUDA_VISIBLE_DEVICES=0 python Main.py --model_save_path ./model --model_name (Model) --num_workers 8 --sampling_rate 16000 --mode train --batch_size 16 --num_epochs (100 or 500) --test_interval 10 --frame_size 400 --n_fft 512 --hop_length 256 vctk_training.list vctk_valid.list;
```   
**3. Testing:**
```
CUDA_VISIBLE_DEVICES=0 python Main.py --model_save_path ./model --model_name (Model) --num_workers 8 --sampling_rate 16000 --mode test --frame_size 400 --n_fft 512 --hop_length 256 vctk_testing.list ./Enhance/(Model);
```	
**3. Evaluation:**
```
python E_ScoreEval.py --num_workers 8 --output_csv (Model).csv vctk_testing.list ./Enhance/(Model);
```	

### Results

|  | PESQ | STOI | ESTOI | 
| -------- | -------- | -------- | -------- |
| Noisy | 1.9627 | 0.9210 | 0.7867 |
| DNN | 1.8333 | 0.8667 | 0.7147 |
| LSTM | 1.7315 | 0.8611 | 0.6982 |
| FCN | 2.1755 | 0.9143 | 0.7970 |
| BLSTM | 2.5088 | 0.9078 | 0.7826 |
