(video-tennis-analysis-py3.12) PS D:\Mestrado\redes_neurais\video_tennis_analysis\video_tennis_analysis> python src\train_model.py
2025-11-16 02:40:13.294355: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-11-16 02:40:14.170922: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
======================================================================
GPU CONFIGURATION
======================================================================

⚠️  No GPU found. Training will use CPU (slower)
   To use GPU, install CUDA and cuDNN, then reinstall tensorflow with GPU support

TensorFlow version: 2.18.1
Built with CUDA: False
======================================================================

Enabling mixed precision training for faster GPU performance...
✅ Mixed precision enabled (float16)

======================================================================
TENNIS STROKE RECOGNITION - POSE + LSTM
FRAME-BASED ANNOTATIONS
======================================================================

Found 21 annotation files

======================================================================
Processing: backhand_1.json
======================================================================
Video: Bhand_1.MP4
Annotations: 15
  - backhand: frames 622-635 (13 frames)
  - backhand: frames 674-688 (14 frames)
  - backhand: frames 718-729 (11 frames)
  - backhand: frames 761-775 (14 frames)
  - backhand: frames 806-818 (12 frames)
  - backhand: frames 850-863 (13 frames)
  - backhand: frames 891-906 (15 frames)
  - backhand: frames 941-953 (12 frames)
  - backhand: frames 984-996 (12 frames)
  - backhand: frames 1051-1062 (11 frames)
  - backhand: frames 1094-1108 (14 frames)
  - backhand: frames 1356-1372 (16 frames)
  - backhand: frames 1411-1426 (15 frames)
  - backhand: frames 1451-1467 (16 frames)
  - backhand: frames 1500-1514 (14 frames)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Processing video: D:\Mestrado\redes_neurais\dados_filtrados\videos\Bhand_1.MP4
FPS: 59.94005994005994, Total frames: 4596
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1763271617.149631   24632 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1763271617.174112    7680 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1763271619.393021   18324 landmark_projection_calculator.cc:186] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
Processed 100/4596 frames
Processed 200/4596 frames
Processed 300/4596 frames
Processed 400/4596 frames
Processed 500/4596 frames
Processed 600/4596 frames
Processed 700/4596 frames
Processed 800/4596 frames
Processed 900/4596 frames
Processed 1000/4596 frames
Processed 1100/4596 frames
Processed 1200/4596 frames
Processed 1300/4596 frames
Processed 1400/4596 frames
Processed 1500/4596 frames
Processed 1600/4596 frames
Processed 1700/4596 frames
Processed 1800/4596 frames
Processed 1900/4596 frames
Processed 2000/4596 frames
Processed 2100/4596 frames
Processed 2200/4596 frames
Processed 2300/4596 frames
Processed 2400/4596 frames
Processed 2500/4596 frames
Processed 2600/4596 frames
Processed 2700/4596 frames
Processed 2800/4596 frames
Processed 2900/4596 frames
Processed 3000/4596 frames
Processed 3100/4596 frames
Processed 3200/4596 frames
Processed 3300/4596 frames
Processed 3400/4596 frames
Processed 3500/4596 frames
Processed 3600/4596 frames
Processed 3700/4596 frames
Processed 3800/4596 frames
Processed 3900/4596 frames
Processed 4000/4596 frames
Processed 4100/4596 frames
Processed 4200/4596 frames
Processed 4300/4596 frames
Processed 4400/4596 frames
Processed 4500/4596 frames
Extracted 4596 pose sequences
Creating sequences from 4596 frames
Window size: 30, Overlap: 15
Created 303 sequences
Label distribution:
  backhand: 2
  neutral: 301

======================================================================
Processing: backhand_2.json
======================================================================
Video: Bhand_2.MP4
Annotations: 20
  - backhand: frames 452-473 (21 frames)
  - backhand: frames 501-514 (13 frames)
  - backhand: frames 546-559 (13 frames)
  - backhand: frames 595-607 (12 frames)
  - backhand: frames 635-648 (13 frames)
  - backhand: frames 672-687 (15 frames)
  - backhand: frames 711-725 (14 frames)
  - backhand: frames 754-767 (13 frames)
  - backhand: frames 792-809 (17 frames)
  - backhand: frames 847-860 (13 frames)
  - backhand: frames 891-902 (11 frames)
  - backhand: frames 969-984 (15 frames)
  - backhand: frames 1023-1039 (16 frames)
  - backhand: frames 1072-1085 (13 frames)
  - backhand: frames 1114-1128 (14 frames)
  - backhand: frames 1158-1172 (14 frames)
  - backhand: frames 1202-1216 (14 frames)
  - backhand: frames 1247-1258 (11 frames)
  - backhand: frames 1536-1549 (13 frames)
  - backhand: frames 1643-1655 (12 frames)
Processing video: D:\Mestrado\redes_neurais\dados_filtrados\videos\Bhand_2.MP4
FPS: 59.94005994005994, Total frames: 4879
Processed 100/4879 frames
Processed 200/4879 frames
Processed 300/4879 frames
Processed 400/4879 frames
Processed 500/4879 frames
Processed 600/4879 frames
Processed 700/4879 frames
Processed 800/4879 frames
Processed 900/4879 frames
Processed 1000/4879 frames
Processed 1100/4879 frames
Processed 1200/4879 frames
Processed 1300/4879 frames
Processed 1400/4879 frames
Processed 1500/4879 frames
Processed 1600/4879 frames
Processed 1700/4879 frames
Processed 1800/4879 frames
Processed 1900/4879 frames
Processed 2000/4879 frames
Processed 2100/4879 frames
Processed 2200/4879 frames
Processed 2300/4879 frames
Processed 2400/4879 frames
Processed 2500/4879 frames
Processed 2600/4879 frames
Processed 2700/4879 frames
Processed 2800/4879 frames
Processed 2900/4879 frames
Processed 3000/4879 frames
Processed 3100/4879 frames
Processed 3200/4879 frames
Processed 3300/4879 frames
Processed 3400/4879 frames
Processed 3500/4879 frames
Processed 3600/4879 frames
Processed 3700/4879 frames
Processed 3800/4879 frames
Processed 3900/4879 frames
Processed 4000/4879 frames
Processed 4100/4879 frames
Processed 4200/4879 frames
Processed 4300/4879 frames
Processed 4400/4879 frames
Processed 4500/4879 frames
Processed 4600/4879 frames
Processed 4700/4879 frames
Processed 4800/4879 frames
Extracted 4879 pose sequences
Creating sequences from 4879 frames
Window size: 30, Overlap: 15
Created 322 sequences
Label distribution:
  backhand: 3
  neutral: 319

======================================================================
Processing: forehand_1.json
======================================================================
Video: Forehand_1.mp4
Annotations: 27
  - fronthand: frames 377-393 (16 frames)
  - fronthand: frames 414-430 (16 frames)
  - fronthand: frames 458-473 (15 frames)
  - fronthand: frames 503-518 (15 frames)
  - fronthand: frames 550-562 (12 frames)
  - fronthand: frames 592-602 (10 frames)
  - fronthand: frames 632-647 (15 frames)
  - fronthand: frames 678-694 (16 frames)
  - fronthand: frames 725-737 (12 frames)
  - fronthand: frames 770-784 (14 frames)
  - fronthand: frames 812-824 (12 frames)
  - fronthand: frames 852-866 (14 frames)
  - fronthand: frames 897-909 (12 frames)
  - fronthand: frames 943-960 (17 frames)
  - fronthand: frames 985-999 (14 frames)
  - fronthand: frames 1028-1045 (17 frames)
  - fronthand: frames 1073-1086 (13 frames)
  - fronthand: frames 1115-1132 (17 frames)
  - fronthand: frames 1163-1177 (14 frames)
  - fronthand: frames 1198-1214 (16 frames)
  - fronthand: frames 1246-1260 (14 frames)
  - fronthand: frames 1294-1306 (12 frames)
  - fronthand: frames 1336-1351 (15 frames)
  - fronthand: frames 1385-1400 (15 frames)
  - fronthand: frames 1429-1444 (15 frames)
  - fronthand: frames 1477-1496 (19 frames)
  - fronthand: frames 1520-1531 (11 frames)
Processing video: D:\Mestrado\redes_neurais\dados_filtrados\videos\Forehand_1.mp4
FPS: 59.94005994005994, Total frames: 4492
Processed 100/4492 frames
Processed 200/4492 frames
Processed 300/4492 frames
Processed 400/4492 frames
Processed 500/4492 frames
Processed 600/4492 frames
Processed 700/4492 frames
Processed 800/4492 frames
Processed 900/4492 frames
Processed 1000/4492 frames
Processed 1100/4492 frames
Processed 1200/4492 frames
Processed 1300/4492 frames
Processed 1400/4492 frames
Processed 1500/4492 frames
Processed 1600/4492 frames
Processed 1700/4492 frames
Processed 1800/4492 frames
Processed 1900/4492 frames
Processed 2000/4492 frames
Processed 2100/4492 frames
Processed 2200/4492 frames
Processed 2300/4492 frames
Processed 2400/4492 frames
Processed 2500/4492 frames
Processed 2600/4492 frames
Processed 2700/4492 frames
Processed 2800/4492 frames
Processed 2900/4492 frames
Processed 3000/4492 frames
Processed 3100/4492 frames
Processed 3200/4492 frames
Processed 3300/4492 frames
Processed 3400/4492 frames
Processed 3500/4492 frames
Processed 3600/4492 frames
Processed 3700/4492 frames
Processed 3800/4492 frames
Processed 3900/4492 frames
Processed 4000/4492 frames
Processed 4100/4492 frames
Processed 4200/4492 frames
Processed 4300/4492 frames
Processed 4400/4492 frames
Extracted 4492 pose sequences
Creating sequences from 4492 frames
Window size: 30, Overlap: 15
Created 291 sequences
Label distribution:
  fronthand: 8
  neutral: 283

======================================================================
Processing: forehand_2.json
======================================================================
Video: Forehand_2.mp4
Annotations: 23
  - fronthand: frames 249-264 (15 frames)
  - fronthand: frames 466-481 (15 frames)
  - fronthand: frames 508-522 (14 frames)
  - fronthand: frames 548-561 (13 frames)
  - fronthand: frames 591-604 (13 frames)
  - fronthand: frames 640-656 (16 frames)
  - fronthand: frames 677-690 (13 frames)
  - fronthand: frames 750-762 (12 frames)
  - fronthand: frames 797-810 (13 frames)
  - fronthand: frames 841-858 (17 frames)
  - fronthand: frames 886-900 (14 frames)
  - fronthand: frames 932-947 (15 frames)
  - fronthand: frames 977-991 (14 frames)
  - fronthand: frames 1020-1032 (12 frames)
  - fronthand: frames 1069-1085 (16 frames)
  - fronthand: frames 1114-1128 (14 frames)
  - fronthand: frames 1160-1174 (14 frames)
  - fronthand: frames 1208-1219 (11 frames)
  - fronthand: frames 1252-1266 (14 frames)
  - fronthand: frames 1305-1320 (15 frames)
  - fronthand: frames 1347-1363 (16 frames)
  - fronthand: frames 1394-1410 (16 frames)
  - fronthand: frames 1441-1449 (8 frames)
Processing video: D:\Mestrado\redes_neurais\dados_filtrados\videos\Forehand_2.mp4
FPS: 59.94005994005994, Total frames: 4369
Processed 100/4369 frames
Processed 200/4369 frames
Processed 300/4369 frames
Processed 400/4369 frames
Processed 500/4369 frames
Processed 600/4369 frames
Processed 700/4369 frames
Processed 800/4369 frames
Processed 900/4369 frames
Processed 1000/4369 frames
Processed 1100/4369 frames
Processed 1200/4369 frames
Processed 1300/4369 frames
Processed 1400/4369 frames
Processed 1500/4369 frames
Processed 1600/4369 frames
Processed 1700/4369 frames
Processed 1800/4369 frames
Processed 1900/4369 frames
Processed 2000/4369 frames
Processed 2100/4369 frames
Processed 2200/4369 frames
Processed 2300/4369 frames
Processed 2400/4369 frames
Processed 2500/4369 frames
Processed 2600/4369 frames
Processed 2700/4369 frames
Processed 2800/4369 frames
Processed 2900/4369 frames
Processed 3000/4369 frames
Processed 3100/4369 frames
Processed 3200/4369 frames
Processed 3300/4369 frames
Processed 3400/4369 frames
Processed 3500/4369 frames
Processed 3600/4369 frames
Processed 3700/4369 frames
Processed 3800/4369 frames
Processed 3900/4369 frames
Processed 4000/4369 frames
Processed 4100/4369 frames
Processed 4200/4369 frames
Processed 4300/4369 frames
Extracted 4369 pose sequences
Creating sequences from 4369 frames
Window size: 30, Overlap: 15
Created 284 sequences
Label distribution:
  fronthand: 5
  neutral: 279

======================================================================
Processing: Jogo_real_1.json
======================================================================
Video: GOPR1058.MP4
Annotations: 15
  - saque: frames 664-715 (51 frames)
  - saque: frames 828-869 (41 frames)
  - saque: frames 1316-1366 (50 frames)
  - saque: frames 1471-1520 (49 frames)
  - slice direita: frames 1556-1575 (19 frames)
  - saque: frames 2039-2089 (50 frames)
  - saque: frames 2176-2217 (41 frames)
  - backhand: frames 2252-2273 (21 frames)
  - saque: frames 2793-2834 (41 frames)
  - saque: frames 3014-3053 (39 frames)
  - saque: frames 3250-3299 (49 frames)
  - saque: frames 3409-3460 (51 frames)
  - saque: frames 3803-3855 (52 frames)
  - saque: frames 3980-4038 (58 frames)
  - backhand: frames 4079-4101 (22 frames)
Processing video: D:\Mestrado\redes_neurais\dados_filtrados\videos\GOPR1058.MP4
FPS: 59.94005994005994, Total frames: 11652
Processed 100/11652 frames
Processed 200/11652 frames
Processed 300/11652 frames
Processed 400/11652 frames
Processed 500/11652 frames
Processed 600/11652 frames
Processed 700/11652 frames
Processed 800/11652 frames
Processed 900/11652 frames
Processed 1000/11652 frames
Processed 1100/11652 frames
Processed 1200/11652 frames
Processed 1300/11652 frames
Processed 1400/11652 frames
Processed 1500/11652 frames
Processed 1600/11652 frames
Processed 1700/11652 frames
Processed 1800/11652 frames
Processed 1900/11652 frames
Processed 2000/11652 frames
Processed 2100/11652 frames
Processed 2200/11652 frames
Processed 2300/11652 frames
Processed 2400/11652 frames
Processed 2500/11652 frames
Processed 2600/11652 frames
Processed 2700/11652 frames
Processed 2800/11652 frames
Processed 2900/11652 frames
Processed 3000/11652 frames
Processed 3100/11652 frames
Processed 3200/11652 frames
Processed 3300/11652 frames
Processed 3400/11652 frames
Processed 3500/11652 frames
Processed 3600/11652 frames
Processed 3700/11652 frames
Processed 3800/11652 frames
Processed 3900/11652 frames
Processed 4000/11652 frames
Processed 4100/11652 frames
Processed 4200/11652 frames
Processed 4300/11652 frames
Processed 4400/11652 frames
Processed 4500/11652 frames
Processed 4600/11652 frames
Processed 4700/11652 frames
Processed 4800/11652 frames
Processed 4900/11652 frames
Processed 5000/11652 frames
Processed 5100/11652 frames
Processed 5200/11652 frames
Processed 5300/11652 frames
Processed 5400/11652 frames
Processed 5500/11652 frames
Processed 5600/11652 frames
Processed 5700/11652 frames
Processed 5800/11652 frames
Processed 5900/11652 frames
Processed 6000/11652 frames
Processed 6100/11652 frames
Processed 6200/11652 frames
Processed 6300/11652 frames
Processed 6400/11652 frames
Processed 6500/11652 frames
Processed 6600/11652 frames
Processed 6700/11652 frames
Processed 6800/11652 frames
Processed 6900/11652 frames
Processed 7000/11652 frames
Processed 7100/11652 frames
Processed 7200/11652 frames
Processed 7300/11652 frames
Processed 7400/11652 frames
Processed 7500/11652 frames
Processed 7600/11652 frames
Processed 7700/11652 frames
Processed 7800/11652 frames
Processed 7900/11652 frames
Processed 8000/11652 frames
Processed 8100/11652 frames
Processed 8200/11652 frames
Processed 8300/11652 frames
Processed 8400/11652 frames
Processed 8500/11652 frames
Processed 8600/11652 frames
Processed 8700/11652 frames
Processed 8800/11652 frames
Processed 8900/11652 frames
Processed 9000/11652 frames
Processed 9100/11652 frames
Processed 9200/11652 frames
Processed 9300/11652 frames
Processed 9400/11652 frames
Processed 9500/11652 frames
Processed 9600/11652 frames
Processed 9700/11652 frames
Processed 9800/11652 frames
Processed 9900/11652 frames
Processed 10000/11652 frames
Processed 10100/11652 frames
Processed 10200/11652 frames
Processed 10300/11652 frames
Processed 10400/11652 frames
Processed 10500/11652 frames
Processed 10600/11652 frames
Processed 10700/11652 frames
Processed 10800/11652 frames
Processed 10900/11652 frames
Processed 11000/11652 frames
Processed 11100/11652 frames
Processed 11200/11652 frames
Processed 11300/11652 frames
Processed 11400/11652 frames
Processed 11500/11652 frames
Processed 11600/11652 frames
Extracted 11652 pose sequences
Creating sequences from 11652 frames
Window size: 30, Overlap: 15
Created 773 sequences
Label distribution:
  backhand: 3
  neutral: 733
  saque: 36
  slice direita: 1

======================================================================
Processing: Jogo_real_2.json
======================================================================
Video: GOPR1059.MP4
Annotations: 26
  - fronthand: frames 857-875 (18 frames)
  - fronthand: frames 2408-2432 (24 frames)
  - fronthand: frames 3217-3241 (24 frames)
  - backhand: frames 4159-4183 (24 frames)
  - fronthand: frames 5164-5185 (21 frames)
  - fronthand: frames 5743-5765 (22 frames)
  - slice direita: frames 5817-5843 (26 frames)
  - fronthand: frames 6656-6677 (21 frames)
  - saque: frames 7684-7737 (53 frames)
  - slice direita: frames 7777-7800 (23 frames)
  - slice esquerda: frames 7843-7861 (18 frames)
  - fronthand: frames 7896-7924 (28 frames)
  - saque: frames 8457-8507 (50 frames)
  - saque: frames 8591-8650 (59 frames)
  - backhand: frames 8691-8716 (25 frames)
  - saque: frames 9208-9255 (47 frames)
  - saque: frames 9360-9409 (49 frames)
  - saque: frames 9862-9907 (45 frames)
  - saque: frames 10080-10127 (47 frames)
  - slice esquerda: frames 10167-10193 (26 frames)
  - saque: frames 10640-10692 (52 frames)
  - saque: frames 10926-10979 (53 frames)
  - slice esquerda: frames 11020-11042 (22 frames)
  - saque: frames 11739-11792 (53 frames)
  - saque: frames 11928-11987 (59 frames)
  - slice direita: frames 12026-12057 (31 frames)
Processing video: D:\Mestrado\redes_neurais\dados_filtrados\videos\GOPR1059.MP4
FPS: 59.94005994005994, Total frames: 31310
Processed 100/31310 frames
Processed 200/31310 frames
Processed 300/31310 frames
Processed 400/31310 frames
Processed 500/31310 frames
Processed 600/31310 frames
Processed 700/31310 frames
Processed 800/31310 frames
Processed 900/31310 frames
Processed 1000/31310 frames
Processed 1100/31310 frames
Processed 1200/31310 frames
Processed 1300/31310 frames
Processed 1400/31310 frames
Processed 1500/31310 frames
Processed 1600/31310 frames
Processed 1700/31310 frames
Processed 1800/31310 frames
Processed 1900/31310 frames
Processed 2000/31310 frames
Processed 2100/31310 frames
Processed 2200/31310 frames
Processed 2300/31310 frames
Processed 2400/31310 frames
Processed 2500/31310 frames
Processed 2600/31310 frames
Processed 2700/31310 frames
Processed 2800/31310 frames
Processed 2900/31310 frames
Processed 3000/31310 frames
Processed 3100/31310 frames
Processed 3200/31310 frames
Processed 3300/31310 frames
Processed 3400/31310 frames
Processed 3500/31310 frames
Processed 3600/31310 frames
Processed 3700/31310 frames
Processed 3800/31310 frames
Processed 3900/31310 frames
Processed 4000/31310 frames
Processed 4100/31310 frames
Processed 4200/31310 frames
Processed 4300/31310 frames
Processed 4400/31310 frames
Processed 4500/31310 frames
Processed 4600/31310 frames
Processed 4700/31310 frames
Processed 4800/31310 frames
Processed 4900/31310 frames
Processed 5000/31310 frames
Processed 5100/31310 frames
Processed 5200/31310 frames
Processed 5300/31310 frames
Processed 5400/31310 frames
Processed 5500/31310 frames
Processed 5600/31310 frames
Processed 5700/31310 frames
Processed 5800/31310 frames
Processed 5900/31310 frames
Processed 6000/31310 frames
Processed 6100/31310 frames
Processed 6200/31310 frames
Processed 6300/31310 frames
Processed 6400/31310 frames
Processed 6500/31310 frames
Processed 6600/31310 frames
Processed 6700/31310 frames
Processed 6800/31310 frames
Processed 6900/31310 frames
Processed 7000/31310 frames
Processed 7100/31310 frames
Processed 7200/31310 frames
Processed 7300/31310 frames
Processed 7400/31310 frames
Processed 7500/31310 frames
Processed 7600/31310 frames
Processed 7700/31310 frames
Processed 7800/31310 frames
Processed 7900/31310 frames
Processed 8000/31310 frames
Processed 8100/31310 frames
Processed 8200/31310 frames
Processed 8300/31310 frames
Processed 8400/31310 frames
Processed 8500/31310 frames
Processed 8600/31310 frames
Processed 8700/31310 frames
Processed 8800/31310 frames
Processed 8900/31310 frames
Processed 9000/31310 frames
Processed 9100/31310 frames
Processed 9200/31310 frames
Processed 9300/31310 frames
Processed 9400/31310 frames
Processed 9500/31310 frames
Processed 9600/31310 frames
Processed 9700/31310 frames
Processed 9800/31310 frames
Processed 9900/31310 frames
Processed 10000/31310 frames
Processed 10100/31310 frames
Processed 10200/31310 frames
Processed 10300/31310 frames
Processed 10400/31310 frames
Processed 10500/31310 frames
Processed 10600/31310 frames
Processed 10700/31310 frames
Processed 10800/31310 frames
Processed 10900/31310 frames
Processed 11000/31310 frames
Processed 11100/31310 frames
Processed 11200/31310 frames
Processed 11300/31310 frames
Processed 11400/31310 frames
Processed 11500/31310 frames
Processed 11600/31310 frames
Processed 11700/31310 frames
Processed 11800/31310 frames
Processed 11900/31310 frames
Processed 12000/31310 frames
Processed 12100/31310 frames
Processed 12200/31310 frames
Processed 12300/31310 frames
Processed 12400/31310 frames
Processed 12500/31310 frames
Processed 12600/31310 frames
Processed 12700/31310 frames
Processed 12800/31310 frames
Processed 12900/31310 frames
Processed 13000/31310 frames
Processed 13100/31310 frames
Processed 13200/31310 frames
Processed 13300/31310 frames
Processed 13400/31310 frames
Processed 13500/31310 frames
Processed 13600/31310 frames
Processed 13700/31310 frames
Processed 13800/31310 frames
Processed 13900/31310 frames
Processed 14000/31310 frames
Processed 14100/31310 frames
Processed 14200/31310 frames
Processed 14300/31310 frames
Processed 14400/31310 frames
Processed 14500/31310 frames
Processed 14600/31310 frames
Processed 14700/31310 frames
Processed 14800/31310 frames
Processed 14900/31310 frames
Processed 15000/31310 frames
Processed 15100/31310 frames
Processed 15200/31310 frames
Processed 15300/31310 frames
Processed 15400/31310 frames
Processed 15500/31310 frames
Processed 15600/31310 frames
Processed 15700/31310 frames
Processed 15800/31310 frames
Processed 15900/31310 frames
Processed 16000/31310 frames
Processed 16100/31310 frames
Processed 16200/31310 frames
Processed 16300/31310 frames
Processed 16400/31310 frames
Processed 16500/31310 frames
Processed 16600/31310 frames
Processed 16700/31310 frames
Processed 16800/31310 frames
Processed 16900/31310 frames
Processed 17000/31310 frames
Processed 17100/31310 frames
Processed 17200/31310 frames
Processed 17300/31310 frames
Processed 17400/31310 frames
Processed 17500/31310 frames
Processed 17600/31310 frames
Processed 17700/31310 frames
Processed 17800/31310 frames
Processed 17900/31310 frames
Processed 18000/31310 frames
Processed 18100/31310 frames
Processed 18200/31310 frames
Processed 18300/31310 frames
Processed 18400/31310 frames
Processed 18500/31310 frames
Processed 18600/31310 frames
Processed 18700/31310 frames
Processed 18800/31310 frames
Processed 18900/31310 frames
Processed 19000/31310 frames
Processed 19100/31310 frames
Processed 19200/31310 frames
Processed 19300/31310 frames
Processed 19400/31310 frames
Processed 19500/31310 frames
Processed 19600/31310 frames
Processed 19700/31310 frames
Processed 19800/31310 frames
Processed 19900/31310 frames
Processed 20000/31310 frames
Processed 20100/31310 frames
Processed 20200/31310 frames
Processed 20300/31310 frames
Processed 20400/31310 frames
Processed 20500/31310 frames
Processed 20600/31310 frames
Processed 20700/31310 frames
Processed 20800/31310 frames
Processed 20900/31310 frames
Processed 21000/31310 frames
Processed 21100/31310 frames
Processed 21200/31310 frames
Processed 21300/31310 frames
Processed 21400/31310 frames
Processed 21500/31310 frames
Processed 21600/31310 frames
Processed 21700/31310 frames
Processed 21800/31310 frames
Processed 21900/31310 frames
Processed 22000/31310 frames
Processed 22100/31310 frames
Processed 22200/31310 frames
Processed 22300/31310 frames
Processed 22400/31310 frames
Processed 22500/31310 frames
Processed 22600/31310 frames
Processed 22700/31310 frames
Processed 22800/31310 frames
Processed 22900/31310 frames
Processed 23000/31310 frames
Processed 23100/31310 frames
Processed 23200/31310 frames
Processed 23300/31310 frames
Processed 23400/31310 frames
Processed 23500/31310 frames
Processed 23600/31310 frames
Processed 23700/31310 frames
Processed 23800/31310 frames
Processed 23900/31310 frames
Processed 24000/31310 frames
Processed 24100/31310 frames
Processed 24200/31310 frames
Processed 24300/31310 frames
Processed 24400/31310 frames
Processed 24500/31310 frames
Processed 24600/31310 frames
Processed 24700/31310 frames
Processed 24800/31310 frames
Processed 24900/31310 frames
Processed 25000/31310 frames
Processed 25100/31310 frames
Processed 25200/31310 frames
Processed 25300/31310 frames
Processed 25400/31310 frames
Processed 25500/31310 frames
Processed 25600/31310 frames
Processed 25700/31310 frames
Processed 25800/31310 frames
Processed 25900/31310 frames
Processed 26000/31310 frames
Processed 26100/31310 frames
Processed 26200/31310 frames
Processed 26300/31310 frames
Processed 26400/31310 frames
Processed 26500/31310 frames
Processed 26600/31310 frames
Processed 26700/31310 frames
Processed 26800/31310 frames
Processed 26900/31310 frames
Processed 27000/31310 frames
Processed 27100/31310 frames
Processed 27200/31310 frames
Processed 27300/31310 frames
Processed 27400/31310 frames
Processed 27500/31310 frames
Processed 27600/31310 frames
Processed 27700/31310 frames
Processed 27800/31310 frames
Processed 27900/31310 frames
Processed 28000/31310 frames
Processed 28100/31310 frames
Processed 28200/31310 frames
Processed 28300/31310 frames
Processed 28400/31310 frames
Processed 28500/31310 frames
Processed 28600/31310 frames
Processed 28700/31310 frames
Processed 28800/31310 frames
Processed 28900/31310 frames
Processed 29000/31310 frames
Processed 29100/31310 frames
Processed 29200/31310 frames
Processed 29300/31310 frames
Processed 29400/31310 frames
Processed 29500/31310 frames
Processed 29600/31310 frames
Processed 29700/31310 frames
Processed 29800/31310 frames
Processed 29900/31310 frames
Processed 30000/31310 frames
Processed 30100/31310 frames
Processed 30200/31310 frames
Processed 30300/31310 frames
Processed 30400/31310 frames
Processed 30500/31310 frames
Processed 30600/31310 frames
Processed 30700/31310 frames
Processed 30800/31310 frames
Processed 30900/31310 frames
Processed 31000/31310 frames
Processed 31100/31310 frames
Processed 31200/31310 frames
Processed 31300/31310 frames
Extracted 31310 pose sequences
Creating sequences from 31310 frames
Window size: 30, Overlap: 15
Created 2082 sequences
Label distribution:
  backhand: 3
  fronthand: 12
  neutral: 2019
  saque: 37
  slice direita: 5
  slice esquerda: 6

======================================================================
Processing: secao_1.json
======================================================================
Video: secao_1.MP4
Annotations: 28
  - fronthand: frames 778-806 (28 frames)
  - fronthand: frames 829-853 (24 frames)
  - backhand: frames 872-888 (16 frames)
  - fronthand: frames 958-978 (20 frames)
  - backhand: frames 995-1012 (17 frames)
  - backhand: frames 1031-1051 (20 frames)
  - backhand: frames 1068-1086 (18 frames)
  - backhand: frames 1107-1126 (19 frames)
  - backhand: frames 1147-1162 (15 frames)
  - fronthand: frames 1260-1280 (20 frames)
  - fronthand: frames 1303-1318 (15 frames)
  - fronthand: frames 1340-1355 (15 frames)
  - fronthand: frames 1375-1394 (19 frames)
  - fronthand: frames 1426-1441 (15 frames)
  - fronthand: frames 1465-1480 (15 frames)
  - fronthand: frames 1501-1521 (20 frames)
  - backhand: frames 1544-1560 (16 frames)
  - backhand: frames 1580-1594 (14 frames)
  - backhand: frames 1615-1628 (13 frames)
  - backhand: frames 1652-1665 (13 frames)
  - fronthand: frames 1871-1889 (18 frames)
  - fronthand: frames 1915-1928 (13 frames)
  - fronthand: frames 1952-1968 (16 frames)
  - fronthand: frames 1993-2004 (11 frames)
  - fronthand: frames 2029-2050 (21 frames)
  - fronthand: frames 2076-2091 (15 frames)
  - backhand: frames 2114-2132 (18 frames)
  - backhand: frames 2157-2174 (17 frames)
Processing video: D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_1.MP4
FPS: 59.94005994005994, Total frames: 6055
Processed 100/6055 frames
Processed 200/6055 frames
Processed 300/6055 frames
Processed 400/6055 frames
Processed 500/6055 frames
Processed 600/6055 frames
Processed 700/6055 frames
Processed 800/6055 frames
Processed 900/6055 frames
Processed 1000/6055 frames
Processed 1100/6055 frames
Processed 1200/6055 frames
Processed 1300/6055 frames
Processed 1400/6055 frames
Processed 1500/6055 frames
Processed 1600/6055 frames
Processed 1700/6055 frames
Processed 1800/6055 frames
Processed 1900/6055 frames
Processed 2000/6055 frames
Processed 2100/6055 frames
Processed 2200/6055 frames
Processed 2300/6055 frames
Processed 2400/6055 frames
Processed 2500/6055 frames
Processed 2600/6055 frames
Processed 2700/6055 frames
Processed 2800/6055 frames
Processed 2900/6055 frames
Processed 3000/6055 frames
Processed 3100/6055 frames
Processed 3200/6055 frames
Processed 3300/6055 frames
Processed 3400/6055 frames
Processed 3500/6055 frames
Processed 3600/6055 frames
Processed 3700/6055 frames
Processed 3800/6055 frames
Processed 3900/6055 frames
Processed 4000/6055 frames
Processed 4100/6055 frames
Processed 4200/6055 frames
Processed 4300/6055 frames
Processed 4400/6055 frames
Processed 4500/6055 frames
Processed 4600/6055 frames
Processed 4700/6055 frames
Processed 4800/6055 frames
Processed 4900/6055 frames
Processed 5000/6055 frames
Processed 5100/6055 frames
Processed 5200/6055 frames
Processed 5300/6055 frames
Processed 5400/6055 frames
Processed 5500/6055 frames
Processed 5600/6055 frames
Processed 5700/6055 frames
Processed 5800/6055 frames
Processed 5900/6055 frames
Processed 6000/6055 frames
Extracted 6055 pose sequences
Creating sequences from 6055 frames
Window size: 30, Overlap: 15
Created 394 sequences
Label distribution:
  backhand: 11
  fronthand: 11
  neutral: 372

======================================================================
Processing: secao_10.json
======================================================================
Video: secao_10.MP4
Annotations: 30
  - fronthand: frames 419-438 (19 frames)
  - fronthand: frames 464-483 (19 frames)
  - backhand: frames 517-536 (19 frames)
  - fronthand: frames 602-626 (24 frames)
  - fronthand: frames 649-672 (23 frames)
  - fronthand: frames 696-711 (15 frames)
  - backhand: frames 736-760 (24 frames)
  - fronthand: frames 929-949 (20 frames)
  - fronthand: frames 971-989 (18 frames)
  - fronthand: frames 1011-1029 (18 frames)
  - fronthand: frames 1050-1069 (19 frames)
  - fronthand: frames 1099-1119 (20 frames)
  - fronthand: frames 1145-1160 (15 frames)
  - fronthand: frames 1196-1215 (19 frames)
  - fronthand: frames 1241-1259 (18 frames)
  - fronthand: frames 1291-1306 (15 frames)
  - fronthand: frames 1435-1456 (21 frames)
  - fronthand: frames 1488-1505 (17 frames)
  - fronthand: frames 1527-1545 (18 frames)
  - backhand: frames 1573-1591 (18 frames)
  - fronthand: frames 1612-1628 (16 frames)
  - backhand: frames 1650-1669 (19 frames)
  - fronthand: frames 1701-1717 (16 frames)
  - fronthand: frames 1753-1775 (22 frames)
  - fronthand: frames 1816-1831 (15 frames)
  - fronthand: frames 1865-1883 (18 frames)
  - fronthand: frames 1910-1926 (16 frames)
  - fronthand: frames 1951-1972 (21 frames)
  - fronthand: frames 2000-2021 (21 frames)
  - fronthand: frames 2045-2062 (17 frames)
Processing video: D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_10.MP4
FPS: 29.97002997002997, Total frames: 3079
Processed 100/3079 frames
Processed 200/3079 frames
Processed 300/3079 frames
Processed 400/3079 frames
Processed 500/3079 frames
Processed 600/3079 frames
Processed 700/3079 frames
Processed 800/3079 frames
Processed 900/3079 frames
Processed 1000/3079 frames
Processed 1100/3079 frames
Processed 1200/3079 frames
Processed 1300/3079 frames
Processed 1400/3079 frames
Processed 1500/3079 frames
Processed 1600/3079 frames
Processed 1700/3079 frames
Processed 1800/3079 frames
Processed 1900/3079 frames
Processed 2000/3079 frames
Processed 2100/3079 frames
Processed 2200/3079 frames
Processed 2300/3079 frames
Processed 2400/3079 frames
Processed 2500/3079 frames
Processed 2600/3079 frames
Processed 2700/3079 frames
Processed 2800/3079 frames
Processed 2900/3079 frames
Processed 3000/3079 frames
Extracted 3079 pose sequences
Creating sequences from 3079 frames
Window size: 30, Overlap: 15
Created 196 sequences
Label distribution:
  backhand: 5
  fronthand: 27
  neutral: 164

======================================================================
Processing: secao_11.json
======================================================================
Video: secao_11.MP4
Annotations: 62
  - fronthand: frames 254-268 (14 frames)
  - fronthand: frames 293-307 (14 frames)
  - fronthand: frames 371-393 (22 frames)
  - fronthand: frames 417-439 (22 frames)
  - backhand: frames 601-623 (22 frames)
  - backhand: frames 654-671 (17 frames)
  - backhand: frames 693-713 (20 frames)
  - fronthand: frames 729-747 (18 frames)
  - fronthand: frames 766-781 (15 frames)
  - fronthand: frames 802-820 (18 frames)
  - fronthand: frames 853-872 (19 frames)
  - fronthand: frames 894-918 (24 frames)
  - fronthand: frames 961-979 (18 frames)
  - fronthand: frames 1006-1026 (20 frames)
  - fronthand: frames 1070-1090 (20 frames)
  - backhand: frames 1133-1150 (17 frames)
  - fronthand: frames 1181-1197 (16 frames)
  - backhand: frames 1229-1250 (21 frames)
  - fronthand: frames 1277-1300 (23 frames)
  - backhand: frames 1334-1354 (20 frames)
  - backhand: frames 1379-1398 (19 frames)
  - fronthand: frames 1428-1446 (18 frames)
  - backhand: frames 1474-1494 (20 frames)
  - fronthand: frames 1522-1540 (18 frames)
  - fronthand: frames 1566-1590 (24 frames)
  - backhand: frames 1680-1697 (17 frames)
  - backhand: frames 1726-1748 (22 frames)
  - fronthand: frames 1776-1793 (17 frames)
  - fronthand: frames 1821-1839 (18 frames)
  - fronthand: frames 1871-1893 (22 frames)
  - fronthand: frames 1920-1942 (22 frames)
  - slice esquerda: frames 1982-2003 (21 frames)
  - fronthand: frames 2064-2086 (22 frames)
  - backhand: frames 2127-2144 (17 frames)
  - backhand: frames 2171-2192 (21 frames)
  - backhand: frames 2660-2679 (19 frames)
  - backhand: frames 2724-2741 (17 frames)
  - fronthand: frames 2777-2798 (21 frames)
  - backhand: frames 2839-2858 (19 frames)
  - backhand: frames 2898-2916 (18 frames)
  - fronthand: frames 2949-2968 (19 frames)
  - backhand: frames 2996-3015 (19 frames)
  - fronthand: frames 3043-3066 (23 frames)
  - fronthand: frames 3093-3118 (25 frames)
  - backhand: frames 3151-3170 (19 frames)
  - fronthand: frames 3204-3222 (18 frames)
  - fronthand: frames 3251-3272 (21 frames)
  - fronthand: frames 3317-3334 (17 frames)
  - fronthand: frames 3366-3386 (20 frames)
  - backhand: frames 3484-3504 (20 frames)
  - backhand: frames 3544-3562 (18 frames)
  - backhand: frames 3640-3663 (23 frames)
  - fronthand: frames 3691-3714 (23 frames)
  - fronthand: frames 3740-3762 (22 frames)
  - backhand: frames 3808-3829 (21 frames)
  - fronthand: frames 3968-3988 (20 frames)
  - fronthand: frames 4017-4040 (23 frames)
  - backhand: frames 4071-4093 (22 frames)
  - backhand: frames 4122-4139 (17 frames)
  - fronthand: frames 4246-4269 (23 frames)
  - fronthand: frames 4308-4329 (21 frames)
  - fronthand: frames 4360-4380 (20 frames)
Processing video: D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_11.MP4
FPS: 29.97002997002997, Total frames: 6374
Processed 100/6374 frames
Processed 200/6374 frames
Processed 300/6374 frames
Processed 400/6374 frames
Processed 500/6374 frames
Processed 600/6374 frames
Processed 700/6374 frames
Processed 800/6374 frames
Processed 900/6374 frames
Processed 1000/6374 frames
Processed 1100/6374 frames
Processed 1200/6374 frames
Processed 1300/6374 frames
Processed 1400/6374 frames
Processed 1500/6374 frames
Processed 1600/6374 frames
Processed 1700/6374 frames
Processed 1800/6374 frames
Processed 1900/6374 frames
Processed 2000/6374 frames
Processed 2100/6374 frames
Processed 2200/6374 frames
Processed 2300/6374 frames
Processed 2400/6374 frames
Processed 2500/6374 frames
Processed 2600/6374 frames
Processed 2700/6374 frames
Processed 2800/6374 frames
Processed 2900/6374 frames
Processed 3000/6374 frames
Processed 3100/6374 frames
Processed 3200/6374 frames
Processed 3300/6374 frames
Processed 3400/6374 frames
Processed 3500/6374 frames
Processed 3600/6374 frames
Processed 3700/6374 frames
Processed 3800/6374 frames
Processed 3900/6374 frames
Processed 4000/6374 frames
Processed 4100/6374 frames
Processed 4200/6374 frames
Processed 4300/6374 frames
Processed 4400/6374 frames
Processed 4500/6374 frames
Processed 4600/6374 frames
Processed 4700/6374 frames
Processed 4800/6374 frames
Processed 4900/6374 frames
Processed 5000/6374 frames
Processed 5100/6374 frames
Processed 5200/6374 frames
Processed 5300/6374 frames
Processed 5400/6374 frames
Processed 5500/6374 frames
Processed 5600/6374 frames
Processed 5700/6374 frames
Processed 5800/6374 frames
Processed 5900/6374 frames
Processed 6000/6374 frames
Processed 6100/6374 frames
Processed 6200/6374 frames
Processed 6300/6374 frames
Extracted 6374 pose sequences
Creating sequences from 6374 frames
Window size: 30, Overlap: 15
Created 417 sequences
Label distribution:
  backhand: 30
  fronthand: 43
  neutral: 343
  slice esquerda: 1

======================================================================
Processing: secao_12.json
======================================================================
Video: secao_12.MP4
Annotations: 53
  - fronthand: frames 216-237 (21 frames)
  - fronthand: frames 262-280 (18 frames)
  - fronthand: frames 300-320 (20 frames)
  - fronthand: frames 338-359 (21 frames)
  - fronthand: frames 388-407 (19 frames)
  - backhand: frames 438-458 (20 frames)
  - backhand: frames 490-509 (19 frames)
  - backhand: frames 536-555 (19 frames)
  - backhand: frames 580-601 (21 frames)
  - fronthand: frames 622-641 (19 frames)
  - fronthand: frames 667-688 (21 frames)
  - backhand: frames 724-742 (18 frames)
  - fronthand: frames 764-786 (22 frames)
  - backhand: frames 878-901 (23 frames)
  - fronthand: frames 937-955 (18 frames)
  - fronthand: frames 992-1011 (19 frames)
  - backhand: frames 1217-1236 (19 frames)
  - backhand: frames 1395-1418 (23 frames)
  - backhand: frames 1451-1473 (22 frames)
  - fronthand: frames 1511-1536 (25 frames)
  - fronthand: frames 1678-1701 (23 frames)
  - fronthand: frames 1735-1763 (28 frames)
  - fronthand: frames 1797-1821 (24 frames)
  - fronthand: frames 1863-1882 (19 frames)
  - backhand: frames 1924-1942 (18 frames)
  - backhand: frames 1977-1995 (18 frames)
  - backhand: frames 2025-2042 (17 frames)
  - fronthand: frames 2073-2092 (19 frames)
  - fronthand: frames 2134-2153 (19 frames)
  - fronthand: frames 2195-2212 (17 frames)
  - fronthand: frames 2260-2288 (28 frames)
  - fronthand: frames 2332-2356 (24 frames)
  - backhand: frames 2403-2423 (20 frames)
  - backhand: frames 2466-2488 (22 frames)
  - fronthand: frames 2529-2552 (23 frames)
  - fronthand: frames 2572-2599 (27 frames)
  - fronthand: frames 3103-3122 (19 frames)
  - fronthand: frames 3150-3170 (20 frames)
  - fronthand: frames 3256-3283 (27 frames)
  - backhand: frames 3309-3333 (24 frames)
  - backhand: frames 3364-3386 (22 frames)
  - backhand: frames 3414-3437 (23 frames)
  - backhand: frames 3468-3492 (24 frames)
  - backhand: frames 3519-3544 (25 frames)
  - backhand: frames 3585-3606 (21 frames)
  - fronthand: frames 3637-3658 (21 frames)
  - backhand: frames 3689-3709 (20 frames)
  - backhand: frames 3736-3757 (21 frames)
  - fronthand: frames 3789-3811 (22 frames)
  - slice direita: frames 3852-3871 (19 frames)
  - slice esquerda: frames 3896-3917 (21 frames)
  - slice esquerda: frames 3944-3967 (23 frames)
  - slice direita: frames 3991-4011 (20 frames)
Processing video: D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_12.MP4
FPS: 29.97002997002997, Total frames: 5232
Processed 100/5232 frames
Processed 200/5232 frames
Processed 300/5232 frames
Processed 400/5232 frames
Processed 500/5232 frames
Processed 600/5232 frames
Processed 700/5232 frames
Processed 800/5232 frames
Processed 900/5232 frames
Processed 1000/5232 frames
Processed 1100/5232 frames
Processed 1200/5232 frames
Processed 1300/5232 frames
Processed 1400/5232 frames
Processed 1500/5232 frames
Processed 1600/5232 frames
Processed 1700/5232 frames
Processed 1800/5232 frames
Processed 1900/5232 frames
Processed 2000/5232 frames
Processed 2100/5232 frames
Processed 2200/5232 frames
Processed 2300/5232 frames
Processed 2400/5232 frames
Processed 2500/5232 frames
Processed 2600/5232 frames
Processed 2700/5232 frames
Processed 2800/5232 frames
Processed 2900/5232 frames
Processed 3000/5232 frames
Processed 3100/5232 frames
Processed 3200/5232 frames
Processed 3300/5232 frames
Processed 3400/5232 frames
Processed 3500/5232 frames
Processed 3600/5232 frames
Processed 3700/5232 frames
Processed 3800/5232 frames
Processed 3900/5232 frames
Processed 4000/5232 frames
Processed 4100/5232 frames
Processed 4200/5232 frames
Processed 4300/5232 frames
Processed 4400/5232 frames
Processed 4500/5232 frames
Processed 4600/5232 frames
Processed 4700/5232 frames
Processed 4800/5232 frames
Processed 4900/5232 frames
Processed 5000/5232 frames
Processed 5100/5232 frames
Processed 5200/5232 frames
Extracted 5232 pose sequences
Creating sequences from 5232 frames
Window size: 30, Overlap: 15
Created 340 sequences
Label distribution:
  backhand: 29
  fronthand: 39
  neutral: 265
  slice direita: 3
  slice esquerda: 4

======================================================================
Processing: secao_13.json
======================================================================
Video: secao_13.MP4
Annotations: 66
  - fronthand: frames 254-280 (26 frames)
  - fronthand: frames 309-331 (22 frames)
  - backhand: frames 348-370 (22 frames)
  - backhand: frames 399-428 (29 frames)
  - slice direita: frames 456-476 (20 frames)
  - fronthand: frames 512-533 (21 frames)
  - fronthand: frames 561-576 (15 frames)
  - fronthand: frames 602-623 (21 frames)
  - fronthand: frames 654-675 (21 frames)
  - fronthand: frames 697-717 (20 frames)
  - fronthand: frames 739-759 (20 frames)
  - backhand: frames 795-820 (25 frames)
  - fronthand: frames 860-882 (22 frames)
  - backhand: frames 914-938 (24 frames)
  - backhand: frames 972-992 (20 frames)
  - fronthand: frames 1018-1039 (21 frames)
  - backhand: frames 1071-1089 (18 frames)
  - backhand: frames 1113-1134 (21 frames)
  - backhand: frames 1165-1191 (26 frames)
  - fronthand: frames 1215-1230 (15 frames)
  - fronthand: frames 1256-1277 (21 frames)
  - backhand: frames 1308-1331 (23 frames)
  - backhand: frames 1355-1375 (20 frames)
  - slice direita: frames 1537-1564 (27 frames)
  - backhand: frames 1595-1621 (26 frames)
  - backhand: frames 1649-1678 (29 frames)
  - backhand: frames 1708-1733 (25 frames)
  - fronthand: frames 1763-1790 (27 frames)
  - fronthand: frames 1815-1837 (22 frames)
  - slice esquerda: frames 1871-1896 (25 frames)
  - backhand: frames 1952-1980 (28 frames)
  - slice esquerda: frames 2083-2112 (29 frames)
  - fronthand: frames 2256-2291 (35 frames)
  - fronthand: frames 2312-2339 (27 frames)
  - fronthand: frames 2367-2388 (21 frames)
  - backhand: frames 2428-2451 (23 frames)
  - backhand: frames 2585-2610 (25 frames)
  - fronthand: frames 2892-2915 (23 frames)
  - fronthand: frames 2947-2969 (22 frames)
  - fronthand: frames 2994-3021 (27 frames)
  - backhand: frames 3052-3077 (25 frames)
  - fronthand: frames 3200-3222 (22 frames)
  - fronthand: frames 3245-3270 (25 frames)
  - backhand: frames 3306-3332 (26 frames)
  - fronthand: frames 3369-3392 (23 frames)
  - fronthand: frames 3411-3435 (24 frames)
  - fronthand: frames 3462-3489 (27 frames)
  - backhand: frames 3522-3545 (23 frames)
  - backhand: frames 3579-3613 (34 frames)
  - backhand: frames 3737-3763 (26 frames)
  - backhand: frames 3793-3814 (21 frames)
  - backhand: frames 3867-3898 (31 frames)
  - backhand: frames 3966-3986 (20 frames)
  - fronthand: frames 4026-4050 (24 frames)
  - backhand: frames 4092-4118 (26 frames)
  - fronthand: frames 4248-4284 (36 frames)
  - fronthand: frames 4316-4344 (28 frames)
  - fronthand: frames 4378-4401 (23 frames)
  - fronthand: frames 4430-4453 (23 frames)
  - fronthand: frames 4472-4499 (27 frames)
  - fronthand: frames 4528-4554 (26 frames)
  - fronthand: frames 4590-4615 (25 frames)
  - fronthand: frames 4640-4660 (20 frames)
  - fronthand: frames 4692-4719 (27 frames)
  - fronthand: frames 4749-4776 (27 frames)
  - fronthand: frames 4812-4841 (29 frames)
Processing video: D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_13.MP4
FPS: 29.97002997002997, Total frames: 7217
Processed 100/7217 frames
Processed 200/7217 frames
Processed 300/7217 frames
Processed 400/7217 frames
Processed 500/7217 frames
Processed 600/7217 frames
Processed 700/7217 frames
Processed 800/7217 frames
Processed 900/7217 frames
Processed 1000/7217 frames
Processed 1100/7217 frames
Processed 1200/7217 frames
Processed 1300/7217 frames
Processed 1400/7217 frames
Processed 1500/7217 frames
Processed 1600/7217 frames
Processed 1700/7217 frames
Processed 1800/7217 frames
Processed 1900/7217 frames
Processed 2000/7217 frames
Processed 2100/7217 frames
Processed 2200/7217 frames
Processed 2300/7217 frames
Processed 2400/7217 frames
Processed 2500/7217 frames
Processed 2600/7217 frames
Processed 2700/7217 frames
Processed 2800/7217 frames
Processed 2900/7217 frames
Processed 3000/7217 frames
Processed 3100/7217 frames
Processed 3200/7217 frames
Processed 3300/7217 frames
Processed 3400/7217 frames
Processed 3500/7217 frames
Processed 3600/7217 frames
Processed 3700/7217 frames
Processed 3800/7217 frames
Processed 3900/7217 frames
Processed 4000/7217 frames
Processed 4100/7217 frames
Processed 4200/7217 frames
Processed 4300/7217 frames
Processed 4400/7217 frames
Processed 4500/7217 frames
Processed 4600/7217 frames
Processed 4700/7217 frames
Processed 4800/7217 frames
Processed 4900/7217 frames
Processed 5000/7217 frames
Processed 5100/7217 frames
Processed 5200/7217 frames
Processed 5300/7217 frames
Processed 5400/7217 frames
Processed 5500/7217 frames
Processed 5600/7217 frames
Processed 5700/7217 frames
Processed 5800/7217 frames
Processed 5900/7217 frames
Processed 6000/7217 frames
Processed 6100/7217 frames
Processed 6200/7217 frames
Processed 6300/7217 frames
Processed 6400/7217 frames
Processed 6500/7217 frames
Processed 6600/7217 frames
Processed 6700/7217 frames
Processed 6800/7217 frames
Processed 6900/7217 frames
Processed 7000/7217 frames
Processed 7100/7217 frames
Processed 7200/7217 frames
Extracted 7217 pose sequences
Creating sequences from 7217 frames
Window size: 30, Overlap: 15
Created 468 sequences
Label distribution:
  backhand: 40
  fronthand: 53
  neutral: 368
  slice direita: 3
  slice esquerda: 4

======================================================================
Processing: secao_14.json
======================================================================
Video: secao_14.MP4
Annotations: 56
  - fronthand: frames 222-253 (31 frames)
  - fronthand: frames 274-294 (20 frames)
  - fronthand: frames 320-345 (25 frames)
  - backhand: frames 377-397 (20 frames)
  - fronthand: frames 435-456 (21 frames)
  - fronthand: frames 485-509 (24 frames)
  - fronthand: frames 533-554 (21 frames)
  - fronthand: frames 581-602 (21 frames)
  - fronthand: frames 633-661 (28 frames)
  - fronthand: frames 690-712 (22 frames)
  - fronthand: frames 744-766 (22 frames)
  - fronthand: frames 802-825 (23 frames)
  - backhand: frames 997-1024 (27 frames)
  - backhand: frames 1051-1077 (26 frames)
  - backhand: frames 1099-1125 (26 frames)
  - backhand: frames 1145-1161 (16 frames)
  - backhand: frames 1185-1204 (19 frames)
  - fronthand: frames 1227-1247 (20 frames)
  - backhand: frames 1285-1316 (31 frames)
  - backhand: frames 1342-1363 (21 frames)
  - fronthand: frames 1412-1432 (20 frames)
  - backhand: frames 1568-1594 (26 frames)
  - backhand: frames 1628-1650 (22 frames)
  - fronthand: frames 1685-1704 (19 frames)
  - fronthand: frames 1736-1759 (23 frames)
  - backhand: frames 1785-1801 (16 frames)
  - fronthand: frames 1831-1854 (23 frames)
  - backhand: frames 1877-1898 (21 frames)
  - backhand: frames 1926-1948 (22 frames)
  - backhand: frames 1980-1996 (16 frames)
  - backhand: frames 2088-2110 (22 frames)
  - backhand: frames 2136-2162 (26 frames)
  - fronthand: frames 2186-2205 (19 frames)
  - backhand: frames 2241-2259 (18 frames)
  - backhand: frames 2291-2310 (19 frames)
  - backhand: frames 2336-2355 (19 frames)
  - backhand: frames 2574-2597 (23 frames)
  - backhand: frames 2634-2652 (18 frames)
  - fronthand: frames 2684-2704 (20 frames)
  - backhand: frames 2739-2763 (24 frames)
  - fronthand: frames 2790-2818 (28 frames)
  - backhand: frames 2853-2875 (22 frames)
  - backhand: frames 2901-2923 (22 frames)
  - backhand: frames 3671-3696 (25 frames)
  - backhand: frames 3728-3748 (20 frames)
  - fronthand: frames 3783-3803 (20 frames)
  - fronthand: frames 3825-3848 (23 frames)
  - backhand: frames 3873-3892 (19 frames)
  - fronthand: frames 3924-3947 (23 frames)
  - backhand: frames 3971-4001 (30 frames)
  - backhand: frames 4125-4147 (22 frames)
  - backhand: frames 4173-4192 (19 frames)
  - fronthand: frames 4216-4241 (25 frames)
  - backhand: frames 4269-4292 (23 frames)
  - backhand: frames 4319-4341 (22 frames)
  - fronthand: frames 4462-4484 (22 frames)
Processing video: D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_14.MP4
FPS: 29.97002997002997, Total frames: 5980
Processed 100/5980 frames
Processed 200/5980 frames
Processed 300/5980 frames
Processed 400/5980 frames
Processed 500/5980 frames
Processed 600/5980 frames
Processed 700/5980 frames
Processed 800/5980 frames
Processed 900/5980 frames
Processed 1000/5980 frames
Processed 1100/5980 frames
Processed 1200/5980 frames
Processed 1300/5980 frames
Processed 1400/5980 frames
Processed 1500/5980 frames
Processed 1600/5980 frames
Processed 1700/5980 frames
Processed 1800/5980 frames
Processed 1900/5980 frames
Processed 2000/5980 frames
Processed 2100/5980 frames
Processed 2200/5980 frames
Processed 2300/5980 frames
Processed 2400/5980 frames
Processed 2500/5980 frames
Processed 2600/5980 frames
Processed 2700/5980 frames
Processed 2800/5980 frames
Processed 2900/5980 frames
Processed 3000/5980 frames
Processed 3100/5980 frames
Processed 3200/5980 frames
Processed 3300/5980 frames
Processed 3400/5980 frames
Processed 3500/5980 frames
Processed 3600/5980 frames
Processed 3700/5980 frames
Processed 3800/5980 frames
Processed 3900/5980 frames
Processed 4000/5980 frames
Processed 4100/5980 frames
Processed 4200/5980 frames
Processed 4300/5980 frames
Processed 4400/5980 frames
Processed 4500/5980 frames
Processed 4600/5980 frames
Processed 4700/5980 frames
Processed 4800/5980 frames
Processed 4900/5980 frames
Processed 5000/5980 frames
Processed 5100/5980 frames
Processed 5200/5980 frames
Processed 5300/5980 frames
Processed 5400/5980 frames
Processed 5500/5980 frames
Processed 5600/5980 frames
Processed 5700/5980 frames
Processed 5800/5980 frames
Processed 5900/5980 frames
Extracted 5980 pose sequences
Creating sequences from 5980 frames
Window size: 30, Overlap: 15
Created 382 sequences
Label distribution:
  backhand: 42
  fronthand: 32
  neutral: 308

======================================================================
Processing: secao_15.json
======================================================================
Video: secao_15.MP4
Annotations: 41
  - backhand: frames 267-289 (22 frames)
  - backhand: frames 315-336 (21 frames)
  - fronthand: frames 358-370 (12 frames)
  - backhand: frames 400-419 (19 frames)
  - fronthand: frames 447-470 (23 frames)
  - fronthand: frames 585-600 (15 frames)
  - fronthand: frames 630-650 (20 frames)
  - fronthand: frames 747-771 (24 frames)
  - backhand: frames 794-818 (24 frames)
  - backhand: frames 840-857 (17 frames)
  - fronthand: frames 887-906 (19 frames)
  - backhand: frames 935-955 (20 frames)
  - fronthand: frames 1030-1050 (20 frames)
  - backhand: frames 1078-1097 (19 frames)
  - backhand: frames 1125-1147 (22 frames)
  - backhand: frames 1174-1193 (19 frames)
  - backhand: frames 1225-1246 (21 frames)
  - backhand: frames 1272-1297 (25 frames)
  - backhand: frames 1324-1340 (16 frames)
  - backhand: frames 1365-1383 (18 frames)
  - backhand: frames 1408-1426 (18 frames)
  - backhand: frames 1456-1478 (22 frames)
  - fronthand: frames 1500-1521 (21 frames)
  - fronthand: frames 1542-1562 (20 frames)
  - fronthand: frames 1589-1614 (25 frames)
  - backhand: frames 1631-1651 (20 frames)
  - fronthand: frames 2802-2824 (22 frames)
  - backhand: frames 2848-2867 (19 frames)
  - backhand: frames 2892-2917 (25 frames)
  - backhand: frames 2945-2963 (18 frames)
  - fronthand: frames 3036-3057 (21 frames)
  - slice direita: frames 3087-3117 (30 frames)
  - slice direita: frames 3150-3176 (26 frames)
  - backhand: frames 3221-3241 (20 frames)
  - backhand: frames 3270-3292 (22 frames)
  - fronthand: frames 3322-3343 (21 frames)
  - backhand: frames 3370-3387 (17 frames)
  - fronthand: frames 3413-3433 (20 frames)
  - backhand: frames 3489-3509 (20 frames)
  - slice direita: frames 3615-3645 (30 frames)
  - backhand: frames 3677-3706 (29 frames)
Processing video: D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_15.MP4
FPS: 29.97002997002997, Total frames: 5381
Processed 100/5381 frames
Processed 200/5381 frames
Processed 300/5381 frames
Processed 400/5381 frames
Processed 500/5381 frames
Processed 600/5381 frames
Processed 700/5381 frames
Processed 800/5381 frames
Processed 900/5381 frames
Processed 1000/5381 frames
Processed 1100/5381 frames
Processed 1200/5381 frames
Processed 1300/5381 frames
Processed 1400/5381 frames
Processed 1500/5381 frames
Processed 1600/5381 frames
Processed 1700/5381 frames
Processed 1800/5381 frames
Processed 1900/5381 frames
Processed 2000/5381 frames
Processed 2100/5381 frames
Processed 2200/5381 frames
Processed 2300/5381 frames
Processed 2400/5381 frames
Processed 2500/5381 frames
Processed 2600/5381 frames
Processed 2700/5381 frames
Processed 2800/5381 frames
Processed 2900/5381 frames
Processed 3000/5381 frames
Processed 3100/5381 frames
Processed 3200/5381 frames
Processed 3300/5381 frames
Processed 3400/5381 frames
Processed 3500/5381 frames
Processed 3600/5381 frames
Processed 3700/5381 frames
Processed 3800/5381 frames
Processed 3900/5381 frames
Processed 4000/5381 frames
Processed 4100/5381 frames
Processed 4200/5381 frames
Processed 4300/5381 frames
Processed 4400/5381 frames
Processed 4500/5381 frames
Processed 4600/5381 frames
Processed 4700/5381 frames
Processed 4800/5381 frames
Processed 4900/5381 frames
Processed 5000/5381 frames
Processed 5100/5381 frames
Processed 5200/5381 frames
Processed 5300/5381 frames
Extracted 5381 pose sequences
Creating sequences from 5381 frames
Window size: 30, Overlap: 15
Created 344 sequences
Label distribution:
  backhand: 35
  fronthand: 17
  neutral: 288
  slice direita: 4

======================================================================
Processing: secao_16.json
======================================================================
Video: secao_16.MP4
Annotations: 60
  - fronthand: frames 288-314 (26 frames)
  - fronthand: frames 348-367 (19 frames)
  - backhand: frames 393-413 (20 frames)
  - fronthand: frames 447-470 (23 frames)
  - fronthand: frames 495-514 (19 frames)
  - backhand: frames 590-620 (30 frames)
  - fronthand: frames 642-669 (27 frames)
  - backhand: frames 697-722 (25 frames)
  - fronthand: frames 755-783 (28 frames)
  - backhand: frames 808-829 (21 frames)
  - backhand: frames 869-890 (21 frames)
  - backhand: frames 922-942 (20 frames)
  - backhand: frames 962-983 (21 frames)
  - backhand: frames 1005-1029 (24 frames)
  - backhand: frames 1057-1085 (28 frames)
  - slice esquerda: frames 1119-1147 (28 frames)
  - fronthand: frames 1171-1194 (23 frames)
  - fronthand: frames 2133-2168 (35 frames)
  - fronthand: frames 2192-2213 (21 frames)
  - fronthand: frames 2247-2272 (25 frames)
  - fronthand: frames 2304-2330 (26 frames)
  - backhand: frames 2351-2379 (28 frames)
  - fronthand: frames 2407-2431 (24 frames)
  - fronthand: frames 2454-2481 (27 frames)
  - backhand: frames 2520-2545 (25 frames)
  - fronthand: frames 2580-2604 (24 frames)
  - fronthand: frames 2817-2839 (22 frames)
  - fronthand: frames 2863-2888 (25 frames)
  - slice esquerda: frames 2913-2947 (34 frames)
  - fronthand: frames 2994-3018 (24 frames)
  - backhand: frames 3074-3093 (19 frames)
  - fronthand: frames 3722-3743 (21 frames)
  - fronthand: frames 3765-3783 (18 frames)
  - fronthand: frames 3809-3825 (16 frames)
  - fronthand: frames 3850-3874 (24 frames)
  - fronthand: frames 3895-3926 (31 frames)
  - fronthand: frames 3952-3978 (26 frames)
  - backhand: frames 4006-4032 (26 frames)
  - backhand: frames 4063-4086 (23 frames)
  - fronthand: frames 4129-4150 (21 frames)
  - fronthand: frames 4180-4202 (22 frames)
  - backhand: frames 4248-4266 (18 frames)
  - backhand: frames 4311-4332 (21 frames)
  - fronthand: frames 4365-4389 (24 frames)
  - fronthand: frames 4415-4436 (21 frames)
  - backhand: frames 4465-4493 (28 frames)
  - fronthand: frames 4621-4646 (25 frames)
  - fronthand: frames 4667-4694 (27 frames)
  - fronthand: frames 4720-4745 (25 frames)
  - backhand: frames 4773-4799 (26 frames)
  - backhand: frames 4970-4995 (25 frames)
  - fronthand: frames 5024-5045 (21 frames)
  - fronthand: frames 5100-5121 (21 frames)
  - backhand: frames 5149-5174 (25 frames)
  - backhand: frames 5205-5232 (27 frames)
  - backhand: frames 5266-5285 (19 frames)
  - backhand: frames 5317-5340 (23 frames)
  - backhand: frames 5367-5395 (28 frames)
  - backhand: frames 5424-5450 (26 frames)
  - backhand: frames 5478-5497 (19 frames)
Processing video: D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_16.MP4
FPS: 29.97002997002997, Total frames: 7463
Processed 100/7463 frames
Processed 200/7463 frames
Processed 300/7463 frames
Processed 400/7463 frames
Processed 500/7463 frames
Processed 600/7463 frames
Processed 700/7463 frames
Processed 800/7463 frames
Processed 900/7463 frames
Processed 1000/7463 frames
Processed 1100/7463 frames
Processed 1200/7463 frames
Processed 1300/7463 frames
Processed 1400/7463 frames
Processed 1500/7463 frames
Processed 1600/7463 frames
Processed 1700/7463 frames
Processed 1800/7463 frames
Processed 1900/7463 frames
Processed 2000/7463 frames
Processed 2100/7463 frames
Processed 2200/7463 frames
Processed 2300/7463 frames
Processed 2400/7463 frames
Processed 2500/7463 frames
Processed 2600/7463 frames
Processed 2700/7463 frames
Processed 2800/7463 frames
Processed 2900/7463 frames
Processed 3000/7463 frames
Processed 3100/7463 frames
Processed 3200/7463 frames
Processed 3300/7463 frames
Processed 3400/7463 frames
Processed 3500/7463 frames
Processed 3600/7463 frames
Processed 3700/7463 frames
Processed 3800/7463 frames
Processed 3900/7463 frames
Processed 4000/7463 frames
Processed 4100/7463 frames
Processed 4200/7463 frames
Processed 4300/7463 frames
Processed 4400/7463 frames
Processed 4500/7463 frames
Processed 4600/7463 frames
Processed 4700/7463 frames
Processed 4800/7463 frames
Processed 4900/7463 frames
Processed 5000/7463 frames
Processed 5100/7463 frames
Processed 5200/7463 frames
Processed 5300/7463 frames
Processed 5400/7463 frames
Processed 5500/7463 frames
Processed 5600/7463 frames
Processed 5700/7463 frames
Processed 5800/7463 frames
Processed 5900/7463 frames
Processed 6000/7463 frames
Processed 6100/7463 frames
Processed 6200/7463 frames
Processed 6300/7463 frames
Processed 6400/7463 frames
Processed 6500/7463 frames
Processed 6600/7463 frames
Processed 6700/7463 frames
Processed 6800/7463 frames
Processed 6900/7463 frames
Processed 7000/7463 frames
Processed 7100/7463 frames
Processed 7200/7463 frames
Processed 7300/7463 frames
Processed 7400/7463 frames
Extracted 7463 pose sequences
Creating sequences from 7463 frames
Window size: 30, Overlap: 15
Created 485 sequences
Label distribution:
  backhand: 37
  fronthand: 49
  neutral: 395
  slice esquerda: 4

======================================================================
Processing: secao_17.json
======================================================================
Video: secao_17.MP4
Annotations: 37
  - fronthand: frames 183-202 (19 frames)
  - fronthand: frames 223-238 (15 frames)
  - fronthand: frames 261-278 (17 frames)
  - fronthand: frames 297-320 (23 frames)
  - fronthand: frames 355-372 (17 frames)
  - fronthand: frames 393-415 (22 frames)
  - fronthand: frames 435-452 (17 frames)
  - fronthand: frames 475-494 (19 frames)
  - fronthand: frames 514-538 (24 frames)
  - fronthand: frames 557-577 (20 frames)
  - fronthand: frames 597-625 (28 frames)
  - backhand: frames 650-676 (26 frames)
  - backhand: frames 700-723 (23 frames)
  - backhand: frames 746-775 (29 frames)
  - fronthand: frames 798-826 (28 frames)
  - backhand: frames 857-882 (25 frames)
  - backhand: frames 906-930 (24 frames)
  - backhand: frames 954-979 (25 frames)
  - fronthand: frames 1004-1030 (26 frames)
  - fronthand: frames 1205-1227 (22 frames)
  - backhand: frames 1345-1367 (22 frames)
  - backhand: frames 1390-1413 (23 frames)
  - backhand: frames 1436-1459 (23 frames)
  - backhand: frames 1476-1502 (26 frames)
  - backhand: frames 1554-1588 (34 frames)
  - backhand: frames 1604-1632 (28 frames)
  - backhand: frames 1657-1675 (18 frames)
  - backhand: frames 1705-1724 (19 frames)
  - backhand: frames 1747-1769 (22 frames)
  - fronthand: frames 1795-1813 (18 frames)
  - backhand: frames 1839-1864 (25 frames)
  - backhand: frames 1890-1910 (20 frames)
  - backhand: frames 1935-1956 (21 frames)
  - fronthand: frames 1980-2006 (26 frames)
  - fronthand: frames 2046-2064 (18 frames)
  - fronthand: frames 2092-2118 (26 frames)
  - backhand: frames 2142-2166 (24 frames)
Processing video: D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_17.MP4
FPS: 29.97002997002997, Total frames: 3515
Processed 100/3515 frames
Processed 200/3515 frames
Processed 300/3515 frames
Processed 400/3515 frames
Processed 500/3515 frames
Processed 600/3515 frames
Processed 700/3515 frames
Processed 800/3515 frames
Processed 900/3515 frames
Processed 1000/3515 frames
Processed 1100/3515 frames
Processed 1200/3515 frames
Processed 1300/3515 frames
Processed 1400/3515 frames
Processed 1500/3515 frames
Processed 1600/3515 frames
Processed 1700/3515 frames
Processed 1800/3515 frames
Processed 1900/3515 frames
Processed 2000/3515 frames
Processed 2100/3515 frames
Processed 2200/3515 frames
Processed 2300/3515 frames
Processed 2400/3515 frames
Processed 2500/3515 frames
Processed 2600/3515 frames
Processed 2700/3515 frames
Processed 2800/3515 frames
Processed 2900/3515 frames
Processed 3000/3515 frames
Processed 3100/3515 frames
Processed 3200/3515 frames
Processed 3300/3515 frames
Processed 3400/3515 frames
Processed 3500/3515 frames
Extracted 3515 pose sequences
Creating sequences from 3515 frames
Window size: 30, Overlap: 15
Created 227 sequences
Label distribution:
  backhand: 31
  fronthand: 22
  neutral: 174

======================================================================
Processing: secao_2.json
======================================================================
Video: secao_2.MP4
Annotations: 44
  - fronthand: frames 332-351 (19 frames)
  - fronthand: frames 370-383 (13 frames)
  - fronthand: frames 410-427 (17 frames)
  - fronthand: frames 454-473 (19 frames)
  - fronthand: frames 494-508 (14 frames)
  - backhand: frames 531-546 (15 frames)
  - backhand: frames 576-593 (17 frames)
  - fronthand: frames 621-633 (12 frames)
  - fronthand: frames 728-744 (16 frames)
  - fronthand: frames 782-798 (16 frames)
  - fronthand: frames 835-856 (21 frames)
  - fronthand: frames 902-916 (14 frames)
  - backhand: frames 943-957 (14 frames)
  - fronthand: frames 1003-1016 (13 frames)
  - backhand: frames 1050-1068 (18 frames)
  - backhand: frames 1103-1117 (14 frames)
  - backhand: frames 1222-1239 (17 frames)
  - backhand: frames 1269-1284 (15 frames)
  - backhand: frames 1323-1338 (15 frames)
  - fronthand: frames 1376-1395 (19 frames)
  - fronthand: frames 1431-1445 (14 frames)
  - fronthand: frames 1476-1495 (19 frames)
  - fronthand: frames 1526-1540 (14 frames)
  - fronthand: frames 1576-1593 (17 frames)
  - backhand: frames 1624-1639 (15 frames)
  - fronthand: frames 1671-1685 (14 frames)
  - fronthand: frames 1716-1735 (19 frames)
  - fronthand: frames 1769-1784 (15 frames)
  - backhand: frames 1814-1834 (20 frames)
  - backhand: frames 1870-1884 (14 frames)
  - fronthand: frames 1919-1934 (15 frames)
  - fronthand: frames 1961-1979 (18 frames)
  - fronthand: frames 2008-2027 (19 frames)
  - fronthand: frames 2052-2072 (20 frames)
  - fronthand: frames 2114-2134 (20 frames)
  - fronthand: frames 2169-2184 (15 frames)
  - fronthand: frames 2216-2234 (18 frames)
  - slice esquerda: frames 2272-2289 (17 frames)
  - fronthand: frames 2420-2436 (16 frames)
  - backhand: frames 2463-2479 (16 frames)
  - fronthand: frames 2508-2524 (16 frames)
  - fronthand: frames 2561-2577 (16 frames)
  - fronthand: frames 2619-2636 (17 frames)
  - backhand: frames 2671-2693 (22 frames)
Processing video: D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_2.MP4
FPS: 59.94005994005994, Total frames: 7492
Processed 100/7492 frames
Processed 200/7492 frames
Processed 300/7492 frames
Processed 400/7492 frames
Processed 500/7492 frames
Processed 600/7492 frames
Processed 700/7492 frames
Processed 800/7492 frames
Processed 900/7492 frames
Processed 1000/7492 frames
Processed 1100/7492 frames
Processed 1200/7492 frames
Processed 1300/7492 frames
Processed 1400/7492 frames
Processed 1500/7492 frames
Processed 1600/7492 frames
Processed 1700/7492 frames
Processed 1800/7492 frames
Processed 1900/7492 frames
Processed 2000/7492 frames
Processed 2100/7492 frames
Processed 2200/7492 frames
Processed 2300/7492 frames
Processed 2400/7492 frames
Processed 2500/7492 frames
Processed 2600/7492 frames
Processed 2700/7492 frames
Processed 2800/7492 frames
Processed 2900/7492 frames
Processed 3000/7492 frames
Processed 3100/7492 frames
Processed 3200/7492 frames
Processed 3300/7492 frames
Processed 3400/7492 frames
Processed 3500/7492 frames
Processed 3600/7492 frames
Processed 3700/7492 frames
Processed 3800/7492 frames
Processed 3900/7492 frames
Processed 4000/7492 frames
Processed 4100/7492 frames
Processed 4200/7492 frames
Processed 4300/7492 frames
Processed 4400/7492 frames
Processed 4500/7492 frames
Processed 4600/7492 frames
Processed 4700/7492 frames
Processed 4800/7492 frames
Processed 4900/7492 frames
Processed 5000/7492 frames
Processed 5100/7492 frames
Processed 5200/7492 frames
Processed 5300/7492 frames
Processed 5400/7492 frames
Processed 5500/7492 frames
Processed 5600/7492 frames
Processed 5700/7492 frames
Processed 5800/7492 frames
Processed 5900/7492 frames
Processed 6000/7492 frames
Processed 6100/7492 frames
Processed 6200/7492 frames
Processed 6300/7492 frames
Processed 6400/7492 frames
Processed 6500/7492 frames
Processed 6600/7492 frames
Processed 6700/7492 frames
Processed 6800/7492 frames
Processed 6900/7492 frames
Processed 7000/7492 frames
Processed 7100/7492 frames
Processed 7200/7492 frames
Processed 7300/7492 frames
Processed 7400/7492 frames
Extracted 7492 pose sequences
Creating sequences from 7492 frames
Window size: 30, Overlap: 15
Created 489 sequences
Label distribution:
  backhand: 7
  fronthand: 23
  neutral: 458
  slice esquerda: 1

======================================================================
Processing: secao_3.json
======================================================================
Video: secao_3.MP4
Annotations: 37
  - fronthand: frames 505-525 (20 frames)
  - backhand: frames 545-565 (20 frames)
  - backhand: frames 593-610 (17 frames)
  - backhand: frames 647-670 (23 frames)
  - backhand: frames 712-731 (19 frames)
  - fronthand: frames 754-770 (16 frames)
  - fronthand: frames 808-824 (16 frames)
  - backhand: frames 861-877 (16 frames)
  - backhand: frames 903-920 (17 frames)
  - fronthand: frames 958-972 (14 frames)
  - slice esquerda: frames 1061-1080 (19 frames)
  - fronthand: frames 1258-1276 (18 frames)
  - fronthand: frames 1296-1315 (19 frames)
  - backhand: frames 1346-1366 (20 frames)
  - backhand: frames 1403-1422 (19 frames)
  - fronthand: frames 1658-1678 (20 frames)
  - fronthand: frames 1709-1724 (15 frames)
  - backhand: frames 1753-1768 (15 frames)
  - backhand: frames 1790-1807 (17 frames)
  - fronthand: frames 1855-1869 (14 frames)
  - backhand: frames 1900-1921 (21 frames)
  - fronthand: frames 1991-2010 (19 frames)
  - fronthand: frames 2201-2223 (22 frames)
  - fronthand: frames 2247-2263 (16 frames)
  - fronthand: frames 2302-2318 (16 frames)
  - backhand: frames 2342-2361 (19 frames)
  - slice esquerda: frames 2390-2409 (19 frames)
  - fronthand: frames 2494-2517 (23 frames)
  - backhand: frames 2555-2571 (16 frames)
  - backhand: frames 2606-2622 (16 frames)
  - backhand: frames 2725-2744 (19 frames)
  - backhand: frames 2775-2790 (15 frames)
  - backhand: frames 2933-2947 (14 frames)
  - backhand: frames 2989-3009 (20 frames)
  - fronthand: frames 3041-3064 (23 frames)
  - fronthand: frames 3093-3110 (17 frames)
  - fronthand: frames 3135-3156 (21 frames)
Processing video: D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_3.MP4
FPS: 59.94005994005994, Total frames: 8975
Processed 100/8975 frames
Processed 200/8975 frames
Processed 300/8975 frames
Processed 400/8975 frames
Processed 500/8975 frames
Processed 600/8975 frames
Processed 700/8975 frames
Processed 800/8975 frames
Processed 900/8975 frames
Processed 1000/8975 frames
Processed 1100/8975 frames
Processed 1200/8975 frames
Processed 1300/8975 frames
Processed 1400/8975 frames
Processed 1500/8975 frames
Processed 1600/8975 frames
Processed 1700/8975 frames
Processed 1800/8975 frames
Processed 1900/8975 frames
Processed 2000/8975 frames
Processed 2100/8975 frames
Processed 2200/8975 frames
Processed 2300/8975 frames
Processed 2400/8975 frames
Processed 2500/8975 frames
Processed 2600/8975 frames
Processed 2700/8975 frames
Processed 2800/8975 frames
Processed 2900/8975 frames
Processed 3000/8975 frames
Processed 3100/8975 frames
Processed 3200/8975 frames
Processed 3300/8975 frames
Processed 3400/8975 frames
Processed 3500/8975 frames
Processed 3600/8975 frames
Processed 3700/8975 frames
Processed 3800/8975 frames
Processed 3900/8975 frames
Processed 4000/8975 frames
Processed 4100/8975 frames
Processed 4200/8975 frames
Processed 4300/8975 frames
Processed 4400/8975 frames
Processed 4500/8975 frames
Processed 4600/8975 frames
Processed 4700/8975 frames
Processed 4800/8975 frames
Processed 4900/8975 frames
Processed 5000/8975 frames
Processed 5100/8975 frames
Processed 5200/8975 frames
Processed 5300/8975 frames
Processed 5400/8975 frames
Processed 5500/8975 frames
Processed 5600/8975 frames
Processed 5700/8975 frames
Processed 5800/8975 frames
Processed 5900/8975 frames
Processed 6000/8975 frames
Processed 6100/8975 frames
Processed 6200/8975 frames
Processed 6300/8975 frames
Processed 6400/8975 frames
Processed 6500/8975 frames
Processed 6600/8975 frames
Processed 6700/8975 frames
Processed 6800/8975 frames
Processed 6900/8975 frames
Processed 7000/8975 frames
Processed 7100/8975 frames
Processed 7200/8975 frames
Processed 7300/8975 frames
Processed 7400/8975 frames
Processed 7500/8975 frames
Processed 7600/8975 frames
Processed 7700/8975 frames
Processed 7800/8975 frames
Processed 7900/8975 frames
Processed 8000/8975 frames
Processed 8100/8975 frames
Processed 8200/8975 frames
Processed 8300/8975 frames
Processed 8400/8975 frames
Processed 8500/8975 frames
Processed 8600/8975 frames
Processed 8700/8975 frames
Processed 8800/8975 frames
Processed 8900/8975 frames
Extracted 8975 pose sequences
Creating sequences from 8975 frames
Window size: 30, Overlap: 15
Created 589 sequences
Label distribution:
  backhand: 17
  fronthand: 17
  neutral: 553
  slice esquerda: 2

======================================================================
Processing: secao_4.json
======================================================================
Video: secao_4.MP4
Annotations: 26
  - fronthand: frames 446-463 (17 frames)
  - fronthand: frames 492-507 (15 frames)
  - backhand: frames 533-549 (16 frames)
  - fronthand: frames 633-652 (19 frames)
  - fronthand: frames 710-736 (26 frames)
  - backhand: frames 756-776 (20 frames)
  - backhand: frames 799-817 (18 frames)
  - backhand: frames 839-853 (14 frames)
  - fronthand: frames 945-963 (18 frames)
  - fronthand: frames 988-1007 (19 frames)
  - fronthand: frames 1247-1265 (18 frames)
  - fronthand: frames 1298-1313 (15 frames)
  - fronthand: frames 1345-1356 (11 frames)
  - fronthand: frames 1381-1396 (15 frames)
  - fronthand: frames 1418-1436 (18 frames)
  - fronthand: frames 1466-1485 (19 frames)
  - fronthand: frames 1517-1532 (15 frames)
  - fronthand: frames 1558-1576 (18 frames)
  - fronthand: frames 1601-1614 (13 frames)
  - backhand: frames 1814-1829 (15 frames)
  - backhand: frames 1857-1873 (16 frames)
  - fronthand: frames 1906-1922 (16 frames)
  - fronthand: frames 1944-1958 (14 frames)
  - fronthand: frames 1983-2003 (20 frames)
  - backhand: frames 2041-2059 (18 frames)
  - backhand: frames 2086-2101 (15 frames)
Processing video: D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_4.MP4
FPS: 47.952047952047955, Total frames: 5020
Processed 100/5020 frames
Processed 200/5020 frames
Processed 300/5020 frames
Processed 400/5020 frames
Processed 500/5020 frames
Processed 600/5020 frames
Processed 700/5020 frames
Processed 800/5020 frames
Processed 900/5020 frames
Processed 1000/5020 frames
Processed 1100/5020 frames
Processed 1200/5020 frames
Processed 1300/5020 frames
Processed 1400/5020 frames
Processed 1500/5020 frames
Processed 1600/5020 frames
Processed 1700/5020 frames
Processed 1800/5020 frames
Processed 1900/5020 frames
Processed 2000/5020 frames
Processed 2100/5020 frames
Processed 2200/5020 frames
Processed 2300/5020 frames
Processed 2400/5020 frames
Processed 2500/5020 frames
Processed 2600/5020 frames
Processed 2700/5020 frames
Processed 2800/5020 frames
Processed 2900/5020 frames
Processed 3000/5020 frames
Processed 3100/5020 frames
Processed 3200/5020 frames
Processed 3300/5020 frames
Processed 3400/5020 frames
Processed 3500/5020 frames
Processed 3600/5020 frames
Processed 3700/5020 frames
Processed 3800/5020 frames
Processed 3900/5020 frames
Processed 4000/5020 frames
Processed 4100/5020 frames
Processed 4200/5020 frames
Processed 4300/5020 frames
Processed 4400/5020 frames
Processed 4500/5020 frames
Processed 4600/5020 frames
Processed 4700/5020 frames
Processed 4800/5020 frames
Processed 4900/5020 frames
Processed 5000/5020 frames
Extracted 5020 pose sequences
Creating sequences from 5020 frames
Window size: 30, Overlap: 15
Created 325 sequences
Label distribution:
  backhand: 5
  fronthand: 14
  neutral: 306

======================================================================
Processing: secao_5.json
======================================================================
Video: secao_5.MP4
Annotations: 44
  - fronthand: frames 437-461 (24 frames)
  - fronthand: frames 478-497 (19 frames)
  - fronthand: frames 573-592 (19 frames)
  - fronthand: frames 614-639 (25 frames)
  - backhand: frames 654-675 (21 frames)
  - backhand: frames 695-716 (21 frames)
  - fronthand: frames 738-759 (21 frames)
  - fronthand: frames 874-897 (23 frames)
  - fronthand: frames 915-936 (21 frames)
  - backhand: frames 959-973 (14 frames)
  - fronthand: frames 1083-1105 (22 frames)
  - fronthand: frames 1132-1155 (23 frames)
  - backhand: frames 1177-1200 (23 frames)
  - backhand: frames 1227-1249 (22 frames)
  - backhand: frames 1272-1294 (22 frames)
  - backhand: frames 1423-1446 (23 frames)
  - fronthand: frames 1472-1497 (25 frames)
  - backhand: frames 1516-1540 (24 frames)
  - backhand: frames 1568-1587 (19 frames)
  - backhand: frames 1605-1624 (19 frames)
  - backhand: frames 1654-1671 (17 frames)
  - backhand: frames 1700-1718 (18 frames)
  - backhand: frames 1743-1760 (17 frames)
  - backhand: frames 1783-1801 (18 frames)
  - backhand: frames 1828-1846 (18 frames)
  - fronthand: frames 1877-1899 (22 frames)
  - fronthand: frames 1919-1938 (19 frames)
  - fronthand: frames 1966-1988 (22 frames)
  - fronthand: frames 2013-2029 (16 frames)
  - fronthand: frames 2057-2072 (15 frames)
  - fronthand: frames 2108-2124 (16 frames)
  - backhand: frames 2149-2169 (20 frames)
  - fronthand: frames 2206-2225 (19 frames)
  - backhand: frames 2741-2760 (19 frames)
  - backhand: frames 2948-2970 (22 frames)
  - backhand: frames 2996-3016 (20 frames)
  - backhand: frames 3050-3067 (17 frames)
  - backhand: frames 3095-3110 (15 frames)
  - fronthand: frames 3142-3166 (24 frames)
  - fronthand: frames 3303-3324 (21 frames)
  - backhand: frames 3469-3487 (18 frames)
  - backhand: frames 3511-3529 (18 frames)
  - fronthand: frames 3732-3750 (18 frames)
  - fronthand: frames 3771-3792 (21 frames)
Processing video: D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_5.MP4
FPS: 47.952047952047955, Total frames: 8142
Processed 100/8142 frames
Processed 200/8142 frames
Processed 300/8142 frames
Processed 400/8142 frames
Processed 500/8142 frames
Processed 600/8142 frames
Processed 700/8142 frames
Processed 800/8142 frames
Processed 900/8142 frames
Processed 1000/8142 frames
Processed 1100/8142 frames
Processed 1200/8142 frames
Processed 1300/8142 frames
Processed 1400/8142 frames
Processed 1500/8142 frames
Processed 1600/8142 frames
Processed 1700/8142 frames
Processed 1800/8142 frames
Processed 1900/8142 frames
Processed 2000/8142 frames
Processed 2100/8142 frames
Processed 2200/8142 frames
Processed 2300/8142 frames
Processed 2400/8142 frames
Processed 2500/8142 frames
Processed 2600/8142 frames
Processed 2700/8142 frames
Processed 2800/8142 frames
Processed 2900/8142 frames
Processed 3000/8142 frames
Processed 3100/8142 frames
Processed 3200/8142 frames
Processed 3300/8142 frames
Processed 3400/8142 frames
Processed 3500/8142 frames
Processed 3600/8142 frames
Processed 3700/8142 frames
Processed 3800/8142 frames
Processed 3900/8142 frames
Processed 4000/8142 frames
Processed 4100/8142 frames
Processed 4200/8142 frames
Processed 4300/8142 frames
Processed 4400/8142 frames
Processed 4500/8142 frames
Processed 4600/8142 frames
Processed 4700/8142 frames
Processed 4800/8142 frames
Processed 4900/8142 frames
Processed 5000/8142 frames
Processed 5100/8142 frames
Processed 5200/8142 frames
Processed 5300/8142 frames
Processed 5400/8142 frames
Processed 5500/8142 frames
Processed 5600/8142 frames
Processed 5700/8142 frames
Processed 5800/8142 frames
Processed 5900/8142 frames
Processed 6000/8142 frames
Processed 6100/8142 frames
Processed 6200/8142 frames
Processed 6300/8142 frames
Processed 6400/8142 frames
Processed 6500/8142 frames
Processed 6600/8142 frames
Processed 6700/8142 frames
Processed 6800/8142 frames
Processed 6900/8142 frames
Processed 7000/8142 frames
Processed 7100/8142 frames
Processed 7200/8142 frames
Processed 7300/8142 frames
Processed 7400/8142 frames
Processed 7500/8142 frames
Processed 7600/8142 frames
Processed 7700/8142 frames
Processed 7800/8142 frames
Processed 7900/8142 frames
Processed 8000/8142 frames
Processed 8100/8142 frames
Extracted 8142 pose sequences
Creating sequences from 8142 frames
Window size: 30, Overlap: 15
Created 530 sequences
Label distribution:
  backhand: 27
  fronthand: 24
  neutral: 479

======================================================================
Processing: secao_6.json
======================================================================
Video: secao_6.MP4
Annotations: 38
  - fronthand: frames 273-299 (26 frames)
  - fronthand: frames 324-349 (25 frames)
  - fronthand: frames 372-390 (18 frames)
  - fronthand: frames 415-433 (18 frames)
  - fronthand: frames 455-474 (19 frames)
  - fronthand: frames 502-518 (16 frames)
  - backhand: frames 544-563 (19 frames)
  - backhand: frames 597-610 (13 frames)
  - backhand: frames 638-661 (23 frames)
  - backhand: frames 687-700 (13 frames)
  - backhand: frames 726-744 (18 frames)
  - backhand: frames 776-796 (20 frames)
  - backhand: frames 828-843 (15 frames)
  - backhand: frames 887-901 (14 frames)
  - fronthand: frames 938-954 (16 frames)
  - fronthand: frames 1046-1066 (20 frames)
  - fronthand: frames 1099-1115 (16 frames)
  - fronthand: frames 1146-1163 (17 frames)
  - slice esquerda: frames 1195-1217 (22 frames)
  - slice direita: frames 1250-1266 (16 frames)
  - slice esquerda: frames 1296-1313 (17 frames)
  - fronthand: frames 1348-1361 (13 frames)
  - slice direita: frames 1393-1410 (17 frames)
  - fronthand: frames 1443-1465 (22 frames)
  - fronthand: frames 1507-1526 (19 frames)
  - fronthand: frames 1565-1586 (21 frames)
  - fronthand: frames 1611-1632 (21 frames)
  - fronthand: frames 1658-1672 (14 frames)
  - backhand: frames 1699-1717 (18 frames)
  - backhand: frames 1741-1756 (15 frames)
  - backhand: frames 1930-1945 (15 frames)
  - backhand: frames 1964-1981 (17 frames)
  - backhand: frames 1998-2016 (18 frames)
  - backhand: frames 2070-2087 (17 frames)
  - backhand: frames 2039-2053 (14 frames)
  - backhand: frames 2109-2121 (12 frames)
  - backhand: frames 2147-2163 (16 frames)
  - backhand: frames 2184-2201 (17 frames)
Processing video: D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_6.MP4
FPS: 47.952047952047955, Total frames: 4933
Processed 100/4933 frames
Processed 200/4933 frames
Processed 300/4933 frames
Processed 400/4933 frames
Processed 500/4933 frames
Processed 600/4933 frames
Processed 700/4933 frames
Processed 800/4933 frames
Processed 900/4933 frames
Processed 1000/4933 frames
Processed 1100/4933 frames
Processed 1200/4933 frames
Processed 1300/4933 frames
Processed 1400/4933 frames
Processed 1500/4933 frames
Processed 1600/4933 frames
Processed 1700/4933 frames
Processed 1800/4933 frames
Processed 1900/4933 frames
Processed 2000/4933 frames
Processed 2100/4933 frames
Processed 2200/4933 frames
Processed 2300/4933 frames
Processed 2400/4933 frames
Processed 2500/4933 frames
Processed 2600/4933 frames
Processed 2700/4933 frames
Processed 2800/4933 frames
Processed 2900/4933 frames
Processed 3000/4933 frames
Processed 3100/4933 frames
Processed 3200/4933 frames
Processed 3300/4933 frames
Processed 3400/4933 frames
Processed 3500/4933 frames
Processed 3600/4933 frames
Processed 3700/4933 frames
Processed 3800/4933 frames
Processed 3900/4933 frames
Processed 4000/4933 frames
Processed 4100/4933 frames
Processed 4200/4933 frames
Processed 4300/4933 frames
Processed 4400/4933 frames
Processed 4500/4933 frames
Processed 4600/4933 frames
Processed 4700/4933 frames
Processed 4800/4933 frames
Processed 4900/4933 frames
Extracted 4933 pose sequences
Creating sequences from 4933 frames
Window size: 30, Overlap: 15
Created 321 sequences
Label distribution:
  backhand: 13
  fronthand: 16
  neutral: 287
  slice direita: 2
  slice esquerda: 3

======================================================================
Processing: secao_8.json
======================================================================
Video: secao_8.MP4
Annotations: 16
  - fronthand: frames 355-379 (24 frames)
  - fronthand: frames 399-415 (16 frames)
  - fronthand: frames 436-452 (16 frames)
  - fronthand: frames 474-495 (21 frames)
  - fronthand: frames 525-538 (13 frames)
  - backhand: frames 565-582 (17 frames)
  - backhand: frames 723-746 (23 frames)
  - backhand: frames 774-794 (20 frames)
  - backhand: frames 873-890 (17 frames)
  - fronthand: frames 1265-1284 (19 frames)
  - backhand: frames 1326-1342 (16 frames)
  - fronthand: frames 1376-1391 (15 frames)
  - fronthand: frames 1525-1548 (23 frames)
  - fronthand: frames 1575-1595 (20 frames)
  - backhand: frames 1620-1636 (16 frames)
  - backhand: frames 1659-1676 (17 frames)
Processing video: D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_8.MP4
FPS: 59.94005994005994, Total frames: 5098
Processed 100/5098 frames
Processed 200/5098 frames
Processed 300/5098 frames
Processed 400/5098 frames
Processed 500/5098 frames
Processed 600/5098 frames
Processed 700/5098 frames
Processed 800/5098 frames
Processed 900/5098 frames
Processed 1000/5098 frames
Processed 1100/5098 frames
Processed 1200/5098 frames
Processed 1300/5098 frames
Processed 1400/5098 frames
Processed 1500/5098 frames
Processed 1600/5098 frames
Processed 1700/5098 frames
Processed 1800/5098 frames
Processed 1900/5098 frames
Processed 2000/5098 frames
Processed 2100/5098 frames
Processed 2200/5098 frames
Processed 2300/5098 frames
Processed 2400/5098 frames
Processed 2500/5098 frames
Processed 2600/5098 frames
Processed 2700/5098 frames
Processed 2800/5098 frames
Processed 2900/5098 frames
Processed 3000/5098 frames
Processed 3100/5098 frames
Processed 3200/5098 frames
Processed 3300/5098 frames
Processed 3400/5098 frames
Processed 3500/5098 frames
Processed 3600/5098 frames
Processed 3700/5098 frames
Processed 3800/5098 frames
Processed 3900/5098 frames
Processed 4000/5098 frames
Processed 4100/5098 frames
Processed 4200/5098 frames
Processed 4300/5098 frames
Processed 4400/5098 frames
Processed 4500/5098 frames
Processed 4600/5098 frames
Processed 4700/5098 frames
Processed 4800/5098 frames
Processed 4900/5098 frames
Processed 5000/5098 frames
Extracted 5098 pose sequences
Creating sequences from 5098 frames
Window size: 30, Overlap: 15
Created 334 sequences
Label distribution:
  backhand: 7
  fronthand: 9
  neutral: 318

======================================================================
TOTAL DATASET
======================================================================
Total sequences: 9896
Sequence shape: (9896, 30, 132)

Label mapping:
  backhand -> 0 (347 samples)
  fronthand -> 1 (421 samples)
  neutral -> 2 (9012 samples)
  saque -> 3 (73 samples)
  slice direita -> 4 (18 samples)
  slice esquerda -> 5 (25 samples)

⚠️  WARNING: Some classes have very few samples (min: 18)
   Consider collecting more data or adjusting window parameters

Train set: 7916 samples
Test set: 1980 samples

======================================================================
BUILDING MODEL
======================================================================
2025-11-16 04:06:14.020323: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ lstm (LSTM)                          │ (None, 30, 128)             │         133,632 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 30, 128)             │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_1 (LSTM)                        │ (None, 30, 64)              │          49,408 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 30, 64)              │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_2 (LSTM)                        │ (None, 32)                  │          12,416 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (Dropout)                  │ (None, 32)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 32)                  │           1,056 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_3 (Dropout)                  │ (None, 32)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 6)                   │             198 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 196,710 (768.40 KB)
 Trainable params: 196,710 (768.40 KB)
 Non-trainable params: 0 (0.00 B)

======================================================================
TRAINING MODEL
======================================================================
Epoch 1/100
2025-11-16 04:06:17.464365: E tensorflow/core/util/util.cc:131] oneDNN supports DT_HALF only on platforms with AVX-512. Falling back to the default Eigen-based implementation if present.
198/198 ━━━━━━━━━━━━━━━━━━━━ 17s 68ms/step - accuracy: 0.9076 - loss: 0.4960 - val_accuracy: 0.9173 - val_loss: 0.3759 - learning_rate: 0.0010
Epoch 2/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.4339 - val_accuracy: 0.9173 - val_loss: 0.3754 - learning_rate: 0.0010
Epoch 3/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.4284 - val_accuracy: 0.9173 - val_loss: 0.3747 - learning_rate: 0.0010
Epoch 4/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.4298 - val_accuracy: 0.9173 - val_loss: 0.3777 - learning_rate: 0.0010
Epoch 5/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.4246 - val_accuracy: 0.9173 - val_loss: 0.3763 - learning_rate: 0.0010
Epoch 6/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.4253 - val_accuracy: 0.9173 - val_loss: 0.3739 - learning_rate: 0.0010
Epoch 7/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.4178 - val_accuracy: 0.9173 - val_loss: 0.3737 - learning_rate: 0.0010
Epoch 8/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 64ms/step - accuracy: 0.9090 - loss: 0.4177 - val_accuracy: 0.9173 - val_loss: 0.3748 - learning_rate: 0.0010
Epoch 9/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.4222 - val_accuracy: 0.9173 - val_loss: 0.3771 - learning_rate: 0.0010
Epoch 10/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.4207 - val_accuracy: 0.9173 - val_loss: 0.3712 - learning_rate: 0.0010
Epoch 11/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.4167 - val_accuracy: 0.9173 - val_loss: 0.3695 - learning_rate: 0.0010
Epoch 12/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 66ms/step - accuracy: 0.9090 - loss: 0.4138 - val_accuracy: 0.9173 - val_loss: 0.3759 - learning_rate: 0.0010
Epoch 13/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.4165 - val_accuracy: 0.9173 - val_loss: 0.3673 - learning_rate: 0.0010
Epoch 14/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.4104 - val_accuracy: 0.9173 - val_loss: 0.3596 - learning_rate: 0.0010
Epoch 15/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.4125 - val_accuracy: 0.9173 - val_loss: 0.3766 - learning_rate: 0.0010
Epoch 16/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.4096 - val_accuracy: 0.9173 - val_loss: 0.3609 - learning_rate: 0.0010
Epoch 17/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.4054 - val_accuracy: 0.9173 - val_loss: 0.3577 - learning_rate: 0.0010
Epoch 18/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.4035 - val_accuracy: 0.9173 - val_loss: 0.3594 - learning_rate: 0.0010
Epoch 19/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.3995 - val_accuracy: 0.9173 - val_loss: 0.3640 - learning_rate: 0.0010
Epoch 20/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.4042 - val_accuracy: 0.9173 - val_loss: 0.3595 - learning_rate: 0.0010
Epoch 21/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.3987 - val_accuracy: 0.9173 - val_loss: 0.3629 - learning_rate: 0.0010
Epoch 22/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.4013 - val_accuracy: 0.9173 - val_loss: 0.3565 - learning_rate: 0.0010
Epoch 23/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 64ms/step - accuracy: 0.9090 - loss: 0.3971 - val_accuracy: 0.9173 - val_loss: 0.3603 - learning_rate: 0.0010
Epoch 24/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.3972 - val_accuracy: 0.9173 - val_loss: 0.3570 - learning_rate: 0.0010
Epoch 25/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.3938 - val_accuracy: 0.9173 - val_loss: 0.3568 - learning_rate: 0.0010
Epoch 26/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.3968 - val_accuracy: 0.9173 - val_loss: 0.3559 - learning_rate: 0.0010
Epoch 27/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.3938 - val_accuracy: 0.9173 - val_loss: 0.3571 - learning_rate: 0.0010
Epoch 28/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.3948 - val_accuracy: 0.9173 - val_loss: 0.3534 - learning_rate: 0.0010
Epoch 29/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.3944 - val_accuracy: 0.9173 - val_loss: 0.3545 - learning_rate: 0.0010
Epoch 30/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.3952 - val_accuracy: 0.9173 - val_loss: 0.3549 - learning_rate: 0.0010
Epoch 31/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.3939 - val_accuracy: 0.9173 - val_loss: 0.3540 - learning_rate: 0.0010
Epoch 32/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.3906 - val_accuracy: 0.9173 - val_loss: 0.3567 - learning_rate: 0.0010
Epoch 33/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.3940 - val_accuracy: 0.9173 - val_loss: 0.3557 - learning_rate: 0.0010
Epoch 34/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 65ms/step - accuracy: 0.9090 - loss: 0.3875 - val_accuracy: 0.9173 - val_loss: 0.3556 - learning_rate: 5.0000e-04
Epoch 35/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 66ms/step - accuracy: 0.9090 - loss: 0.3875 - val_accuracy: 0.9173 - val_loss: 0.3538 - learning_rate: 5.0000e-04
Epoch 36/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 66ms/step - accuracy: 0.9090 - loss: 0.3905 - val_accuracy: 0.9173 - val_loss: 0.3553 - learning_rate: 5.0000e-04
Epoch 37/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 66ms/step - accuracy: 0.9090 - loss: 0.3888 - val_accuracy: 0.9173 - val_loss: 0.3560 - learning_rate: 5.0000e-04
Epoch 38/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 66ms/step - accuracy: 0.9090 - loss: 0.3872 - val_accuracy: 0.9173 - val_loss: 0.3541 - learning_rate: 5.0000e-04
Epoch 39/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 66ms/step - accuracy: 0.9090 - loss: 0.3873 - val_accuracy: 0.9173 - val_loss: 0.3520 - learning_rate: 2.5000e-04
Epoch 40/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 66ms/step - accuracy: 0.9090 - loss: 0.3848 - val_accuracy: 0.9173 - val_loss: 0.3542 - learning_rate: 2.5000e-04
Epoch 41/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 66ms/step - accuracy: 0.9090 - loss: 0.3839 - val_accuracy: 0.9173 - val_loss: 0.3522 - learning_rate: 2.5000e-04
Epoch 42/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 66ms/step - accuracy: 0.9090 - loss: 0.3831 - val_accuracy: 0.9173 - val_loss: 0.3492 - learning_rate: 2.5000e-04
Epoch 43/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 66ms/step - accuracy: 0.9090 - loss: 0.3828 - val_accuracy: 0.9173 - val_loss: 0.3504 - learning_rate: 2.5000e-04
Epoch 44/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 66ms/step - accuracy: 0.9090 - loss: 0.3802 - val_accuracy: 0.9173 - val_loss: 0.3484 - learning_rate: 2.5000e-04
Epoch 45/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 66ms/step - accuracy: 0.9090 - loss: 0.3805 - val_accuracy: 0.9173 - val_loss: 0.3501 - learning_rate: 2.5000e-04
Epoch 46/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 67ms/step - accuracy: 0.9090 - loss: 0.3787 - val_accuracy: 0.9173 - val_loss: 0.3533 - learning_rate: 2.5000e-04
Epoch 47/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 67ms/step - accuracy: 0.9090 - loss: 0.3815 - val_accuracy: 0.9173 - val_loss: 0.3506 - learning_rate: 2.5000e-04
Epoch 48/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 67ms/step - accuracy: 0.9090 - loss: 0.3787 - val_accuracy: 0.9173 - val_loss: 0.3539 - learning_rate: 2.5000e-04
Epoch 49/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 67ms/step - accuracy: 0.9090 - loss: 0.3795 - val_accuracy: 0.9173 - val_loss: 0.3583 - learning_rate: 2.5000e-04
Epoch 50/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 67ms/step - accuracy: 0.9090 - loss: 0.3766 - val_accuracy: 0.9173 - val_loss: 0.3522 - learning_rate: 1.2500e-04
Epoch 51/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 67ms/step - accuracy: 0.9090 - loss: 0.3746 - val_accuracy: 0.9173 - val_loss: 0.3520 - learning_rate: 1.2500e-04
Epoch 52/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 67ms/step - accuracy: 0.9090 - loss: 0.3737 - val_accuracy: 0.9173 - val_loss: 0.3524 - learning_rate: 1.2500e-04
Epoch 53/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 67ms/step - accuracy: 0.9090 - loss: 0.3726 - val_accuracy: 0.9173 - val_loss: 0.3528 - learning_rate: 1.2500e-04
Epoch 54/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 67ms/step - accuracy: 0.9090 - loss: 0.3718 - val_accuracy: 0.9173 - val_loss: 0.3524 - learning_rate: 1.2500e-04
Epoch 55/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 67ms/step - accuracy: 0.9090 - loss: 0.3695 - val_accuracy: 0.9173 - val_loss: 0.3527 - learning_rate: 6.2500e-05
Epoch 56/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 67ms/step - accuracy: 0.9090 - loss: 0.3705 - val_accuracy: 0.9173 - val_loss: 0.3529 - learning_rate: 6.2500e-05
Epoch 57/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 67ms/step - accuracy: 0.9090 - loss: 0.3710 - val_accuracy: 0.9173 - val_loss: 0.3522 - learning_rate: 6.2500e-05
Epoch 58/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 67ms/step - accuracy: 0.9090 - loss: 0.3670 - val_accuracy: 0.9173 - val_loss: 0.3523 - learning_rate: 6.2500e-05
Epoch 59/100
198/198 ━━━━━━━━━━━━━━━━━━━━ 13s 67ms/step - accuracy: 0.9090 - loss: 0.3671 - val_accuracy: 0.9173 - val_loss: 0.3520 - learning_rate: 6.2500e-05

======================================================================
EVALUATION
======================================================================
62/62 ━━━━━━━━━━━━━━━━━━━━ 2s 27ms/step - accuracy: 0.9106 - loss: 0.3709

Test Accuracy: 0.9106
Test Loss: 0.3709
62/62 ━━━━━━━━━━━━━━━━━━━━ 2s 31ms/step

Classification Report:
D:\Mestrado\redes_neurais\video_tennis_analysis\video_tennis_analysis\.venv\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
D:\Mestrado\redes_neurais\video_tennis_analysis\video_tennis_analysis\.venv\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
D:\Mestrado\redes_neurais\video_tennis_analysis\video_tennis_analysis\.venv\Lib\site-packages\sklearn\metrics\_classification.py:1731: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", result.shape[0])
                precision    recall  f1-score   support

      backhand       0.00      0.00      0.00        69
     fronthand       0.00      0.00      0.00        84
       neutral       0.91      1.00      0.95      1803
         saque       0.00      0.00      0.00        15
 slice direita       0.00      0.00      0.00         4
slice esquerda       0.00      0.00      0.00         5

      accuracy                           0.91      1980
     macro avg       0.15      0.17      0.16      1980
  weighted avg       0.83      0.91      0.87      1980


Confusion Matrix:
[[   0    0   69    0    0    0]
 [   0    0   84    0    0    0]
 [   0    0 1803    0    0    0]
 [   0    0   15    0    0    0]
 [   0    0    4    0    0    0]
 [   0    0    5    0    0    0]]

Model saved to D:\Mestrado\redes_neurais\video_tennis_analysis\video_tennis_analysis\output/tennis_stroke_model.keras
Training history saved to D:\Mestrado\redes_neurais\video_tennis_analysis\video_tennis_analysis\output/training_history.png

======================================================================
TRAINING COMPLETE!
======================================================================

Model and artifacts saved to: D:\Mestrado\redes_neurais\video_tennis_analysis\video_tennis_analysis\output/

Next steps:
1. Check training_history.png for overfitting
2. Test on new videos: python detect_strokes.py <video_path>
3. If accuracy is low, consider:
   - Adding more training videos
   - Adjusting window_size in CONFIG
   - Balancing classes (equal forehand/backhand samples)
(video-tennis-analysis-py3.12) PS D:\Mestrado\redes_neurais\video_tennis_analysis\video_tennis_analysis>