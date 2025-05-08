📑 data_schema.md – Health Data Schema for Claude
📁 data/raw/cardiovascular_diseases/cardio_train.csv
Format: ;-separated CSV

Samples:

id,age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active,cardio
0,18393,2,168,62,110,80,1,1,0,0,1,0
Target column: cardio (0/1)

Key features:

age: in days → convert to years

ap_hi, ap_lo: blood pressure

cholesterol, gluc: encoded 1–3 (low/normal/high)

📁 data/raw/heart_disease/scp_statements.csv
Format: standard CSV

Samples:

male,age,education,currentSmoker,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,...
1,39,4,0,0,0,0,0,...
Target column: TenYearCHD (0/1 – 10-year coronary heart disease risk)

Key features:

totChol, sysBP, diaBP, glucose, BMI, heartRate

📁 data/raw/kaggle_diabetes/diabetes_012_health_indicators.csv
Format: standard CSV

Samples:

Diabetes_012,HighBP,HighChol,CholCheck,BMI,Smoker,...
0.0,1.0,1.0,1.0,40.0,1.0,...
Target column: Diabetes_012 (0: no, 1: prediabetes, 2: diabetic → map to binary)

Key features:

HighBP, BMI, Smoker, Age, Income, DiffWalk, GenHlth, etc.

🧠 Unified Label Mapping for y_train (for supervised learning)
Source	Column	Mapping for Binary
cardio	cardio	0, 1
heart_disease	TenYearCHD	0, 1
diabetes	Diabetes_012	0, 1 (where > 0 → 1)