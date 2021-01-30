import argparse
import cv2
import numpy as np
from skimage.transform import rotate
import time
import sys


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('ana_resim', type=str,help='Arama yapilacak buyuk resmin yolu')
parser.add_argument('kucuk_resim', type=str,help='Rotate edilmis veya edilmemis kucuk resmin yolu')
parser.add_argument('--stride',type=int,help='Dongu icerisinde ana resim her defasinda kac derece arttirilarak dondurulecek',default=10)
parser.add_argument('--threshold', type=float,help='Normalized cross correlation icin minimum threshold degeri',default=0.8)
parser.add_argument('--sadace_ilk', type=bool,help='Ana resim her defasinda dondurulup korelasyona tabi tutulurken sadece thresholdu gecen ilk dondurulmus resim mi dikkate alinacak',default=True)
parser.add_argument('--sadece_max', type=bool,help='Thresholdu gecen korelasyonlardan sadece en buyuk olani mi dikkate alinacak',default=True)

args = parser.parse_args()

ACI_STRIDE = args.stride
THRESHOLD = args.threshold
ANA_RESIM_YOL = args.ana_resim
TEMPLATE_RESIM_YOL = args.kucuk_resim
SADECE_ILK = args.sadace_ilk
SADECE_MAX = args.sadece_max

ana_resim = cv2.imread(ANA_RESIM_YOL).astype(np.float32)
template = cv2.imread(TEMPLATE_RESIM_YOL).astype(np.float32)
w, h = template.shape[:-1]


thresholdu_gecenler_index = list()
corr_degerleri = list()
rotasyonlar = list()
for i in range(360,0,-ACI_STRIDE):
    rotated = rotate(ana_resim,i)
    corr_matris = cv2.matchTemplate(rotated, template, cv2.TM_CCORR_NORMED)
    x_ler,y_ler = np.where(corr_matris >= THRESHOLD)[::-1]
    if len(x_ler) != 0:
        thresholdu_gecenler_index.append(np.concatenate([x_ler.reshape(-1,1),y_ler.reshape(-1,1)],axis=1))
        corr_degerleri.append(corr_matris[y_ler,x_ler])
        rotasyonlar.extend([i]*len(x_ler))
        if SADECE_ILK == True:
            break            


if len(thresholdu_gecenler_index) == 0:
    print(f"{THRESHOLD} threshold degeri icin siniri gecen korelasyon olmadi.")
    sys.exit()
    
    
thresholdu_gecenler_index = np.concatenate(thresholdu_gecenler_index,axis=0)
corr_degerleri = np.concatenate(corr_degerleri,axis=0)
rotasyonlar = np.array(rotasyonlar)


if SADECE_MAX == True:
    corr_argmax = corr_degerleri.argmax()
    
    maks_koor = thresholdu_gecenler_index[corr_argmax]
    corr = corr_degerleri[corr_argmax]
    aci = rotasyonlar[corr_argmax]
    
    rotated_image = rotate(ana_resim,aci)
    
    cv2.rectangle(rotated_image, tuple(maks_koor), (maks_koor[0] + w, maks_koor[1] + h), (0,0,255))
    cv2.imwrite('result.png', rotated_image)
    print(f"Orjinal resmin {aci%360} derece rotate edilmis hali icin: top-left = {maks_koor}, bottom-right={maks_koor+[w,h]}, corr = {corr}")
else:
    for i in range(len(corr_degerleri)):
        index = thresholdu_gecenler_index[i]
        corr = corr_degerleri[i]
        aci = rotasyonlar[i]
        rotated_image = rotate(ana_resim,aci)
        
        cv2.rectangle(rotated_image, tuple(index), (index[0] + w, index[1] + h), (0,0,255))
        print(f"Orjinal resmin {aci%360} derece rotate edilmis hali icin: top-left = {index}, bottom-right={index+[w,h]}, corr = {corr}")

