# Vinicius-Datathon-2026

## Tong quan
Du an nay tap trung vao bai toan du bao doanh thu (Revenue) va gia von (COGS) theo ngay cho Datathon 2026. Codebase gom EDA, xu ly du lieu, va cac notebook/ script huan luyen + du bao.

## Cau truc thu muc
```
Datathon/
	data/
	MCQ.ipynb
	README.md
	EDA_Analyse/
		data_fetch.py
		Descriptive.ipynb
		output/
	Forecast/
		vinicius-model/
			requirements.txt
			data/
				submission.csv
			model/
				1-preprocessing.ipynb
				2-train_model-and-shap.ipynb
				3-final-training.ipynb
				EDA.ipynb
				final_submission.csv
				old_forecast.py
			original-data/
			processed-data/
```

## Thiet lap moi truong
1. Cai dat Python 3.9+.
2. Cai dat thu vien co ban:
	 ```bash
	 pip install -r Forecast/vinicius-model/requirements.txt
	 ```
3. Neu chay script du bao hoac notebook huan luyen, can cai them:
	 ```bash
	 pip install lightgbm xgboost scikit-learn kagglehub
	 ```

## Tai du lieu
File [EDA_Analyse/data_fetch.py](EDA_Analyse/data_fetch.py) dung kagglehub de tai bo du lieu cua cuoc thi. Hay bao dam ban da cau hinh tai khoan Kaggle truoc khi chay:
```bash
python EDA_Analyse/data_fetch.py
```

## EDA va phan tich
- [EDA_Analyse/Descriptive.ipynb](EDA_Analyse/Descriptive.ipynb): thong ke mo ta va bieu do.
- [Forecast/vinicius-model/model/EDA.ipynb](Forecast/vinicius-model/model/EDA.ipynb): EDA phuc vu du bao.

## Huan luyen va du bao
File [Forecast/vinicius-model/model/old_forecast.py](Forecast/vinicius-model/model/old_forecast.py) la script du bao tu dong. Y tuong chinh:
- Tao dac trung thoi gian (nam, thang, thu, chu ky sin/cos).
- Dac trung mua vu (Tet, le VN, cuoi thang, giua thang).
- Lag va rolling mean tu du lieu train.
- Train LightGBM + XGBoost, sau do du bao de quy cho tap test.

Chay script:
```bash
python Forecast/vinicius-model/model/old_forecast.py
```
Ket qua se duoc ghi vao thu muc [Forecast/vinicius-model/data/submission.csv](Forecast/vinicius-model/data/submission.csv).

Neu muon huan luyen theo notebook, xem chuoi notebook trong thu muc [Forecast/vinicius-model/model](Forecast/vinicius-model/model).

## Ghi chu
- Du lieu phu (orders, payments, ...) chi co trong giai doan train. Tap test khong co du lieu phu.
- Du bao can dua nhieu vao dac trung lich va chu ky mua vu.