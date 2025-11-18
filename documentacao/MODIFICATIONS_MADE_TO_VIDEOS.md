MODIFICAÇÕES FEITAS NOS VÍDEOS:

python src/preprocess_video.py "D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_10.MP4" "D:\Mestrado\redes_neurais\dados_filtrados\videos_corrigidos\secao_10_corrigidos.MP4"  --static-zoom 1.4 --fisheye
python src/preprocess_video.py "D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_11.MP4" "D:\Mestrado\redes_neurais\dados_filtrados\videos_corrigidos\secao_11_corrigidos.MP4"  --static-zoom 1.4 --fisheye
python src/preprocess_video.py "D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_12.MP4" "D:\Mestrado\redes_neurais\dados_filtrados\videos_corrigidos\secao_12_corrigidos.MP4"  --static-zoom 1.4 --fisheye
python src/preprocess_video.py "D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_13.MP4" "D:\Mestrado\redes_neurais\dados_filtrados\videos_corrigidos\secao_13_corrigidos.MP4"  --static-zoom 1.4 --fisheye

---------
# LUZ
python src/preprocess_video.py "D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_6.MP4" "D:\Mestrado\redes_neurais\dados_filtrados\videos_corrigidos\secao_6_corrigidos.MP4"  --auto-brighten --fisheye

# ZOOM E LUZ
python src/preprocess_video.py "D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_5.MP4" "D:\Mestrado\redes_neurais\dados_filtrados\videos_corrigidos\secao_5_corrigidos.MP4"  --auto-brighten --static-zoom 1.6 --fisheye
no 5 a correcao de luz nao ficou boa
python src/preprocess_video.py "D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_5.MP4" "D:\Mestrado\redes_neurais\dados_filtrados\videos_corrigidos\secao_5_corrigidos.MP4"  --static-zoom 1.6 --fisheye

# ZOOM
python src/preprocess_video.py "D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_4.MP4" "D:\Mestrado\redes_neurais\dados_filtrados\videos_corrigidos\secao_4_corrigidos.MP4"  --static-zoom 1.6 --fisheye
python src/preprocess_video.py "D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_3.MP4" "D:\Mestrado\redes_neurais\dados_filtrados\videos_corrigidos\secao_3_corrigidos.MP4"  --static-zoom 1.4 --fisheye
python src/preprocess_video.py "D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_2.MP4" "D:\Mestrado\redes_neurais\dados_filtrados\videos_corrigidos\secao_2_corrigidos.MP4"  --static-zoom 1.4 --fisheye
python src/preprocess_video.py "D:\Mestrado\redes_neurais\dados_filtrados\videos\secao_1.MP4" "D:\Mestrado\redes_neurais\dados_filtrados\videos_corrigidos\secao_1_corrigidos.MP4"  --static-zoom 1.4 --fisheye
