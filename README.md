Error yang saya alami saat run experiment:

1. API request ke artifact error
2. Kernel crash (killed) berulang kali saat jalanin log nya
  
untuk problem yang kedua saya udah coba cari tahu apa problemnya lewat google/gpt, kemungkinan karena spek laptop saya yang kurang mencukupi, saat log nya jalan saya cek alokasi cpu sama memorinya langsung naik ke 100%. 

Akhirnya saya coba minta tolong ke gpt untuk perbaikin kodenya supaya data yang di train sedikit aja sama gunain log yang ngga terlalu makan resource, ini [kodenya](https://github.com/Rico-febrian/mlflow-lazada-error/blob/main/lazada-project/lzd_project/notebooks/experiment.py), hasilnya berhasil ke deploy ke mlflow dengan warning yang cukup banyak. 
