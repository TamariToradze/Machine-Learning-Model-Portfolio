# ფინანსური თაღლითობის აღმოჩენის სისტემა

პროექტის მიმოხილვა
ეს პროექტი წარმოადგენს ფინანსური თაღლითობის აღმოჩენის სისტემას, რომელიც იყენებს IEEE-CIS მონაცემებს. სისტემა აფასებს რამდენიმე მანქანური სწავლების ალგორითმს თაღლითური ტრანზაქციების იდენტიფიცირებისთვის.

გამოყენებული მოდელები
- Logistic Regression
- Random Forest
- AdaBoost
- XGBoost

რეპოზიტორიის სტრუქტურა
model_experiment_logreg.ipynb   Logistic Regression-ის pipeline
model_inference_logreg.ipynb    საუკეთესო LR მოდელით ტესტ სეტზე პროგნოზი
model_experiment_rf.ipynb       Random Forest-ის pipeline
model_inference_rf.ipynb        საუკეთესო RF მოდელით ტესტ სეტზე პროგნოზი
model_experiment_ada.ipynb      AdaBoost-ის pipeline
model_inference_ada.ipynb       საუკეთესო AdaBoost მოდელით ტესტ სეტზე პროგნოზი
model_experiment_xgb.ipynb      XGBoost-ის pipeline
model_inference_xgb.ipynb       საუკეთესო XGBoost მოდელით ტესტ სეტზე პროგნოზი
xgb.csv                         XGBoost-ის პროგნოზები ტესტ სეტზე


Cleaning/Feature Engineering
- ცარიელი მნიშვნელობები: რიცხვითი სვეტები შეივსო ნულებით
- კატეგორიული ცვლადები: გამოყენებულია სპეციალური კოდირება
- კოდირების მეთოდი: გამოყენებულია WOE (Weight of Evidence) კოდირება
- ფიჩერების შემცირება: წაიშალა 6 მაღალი კარდინალობის მქონე ცვლადი

ხმაურის შემცირება
- წაიშალა სვეტები გამოტოვებული მნიშვნელობების მაღალი რაოდენობით
- წაიშალა ერთი მნიშვნელობის მქონე სვეტები
- წაიშალა მაღალი სიხშირის მუდმივი მნიშვნელობების მქონე სვეტები

Feature Selection
- კორელაციის ანალიზი heat map-ების გამოყენებით D, V, C და ID ტიპის სვეტებისთვის
- გამოყენებულია კორელაციის ზღვარი 0.8
- გამოყენებულია RFE (Recursive Feature Elimination) ალგორითმი ფიჩერების შესარჩევად

მოდელის ტრენირება

ჰიპერპარამეტრების ოპტიმიზაცია
- გამოვიყენებულია GridSearch პარამეტრების შესარჩევად
- ოპტიმიზირებული პარამეტრები გამოყენებულია საბოლოო pipeline-ში

მოდელების შედეგები (F1 მაჩვენებლები)

მოდელი        Train/Test
XGBoost:      0.77/0.75
RandomForest: 0.83/0.75
LogReg:       0.59/0.58
AdaBoost:     0.66/0.65


საბოლოო მოდელის არჩევანი: XGBoost
შერჩეულია შემდეგი კრიტერიუმების საფუძველზე:
- მაღალი F1 და recall მაჩვენებელი ტესტ სეტზე
- მაღალი ROC AUC
- მინიმალური overfitting (მცირე სხვაობა train და test F1 შორის)

MLflow თრექინგი
 მოდელების ბმულები
- [XGBoost მოდელი](https://dagshub.com/TamariToradze/IEEE-CIS-Fraud-Detection_ass2/models/XGB_2/1)
- [Logistic Regression მოდელი](https://dagshub.com/TamariToradze/IEEE-CIS-Fraud-Detection_ass2/models/LogisticRegression/1)
- [Random Forest მოდელი](https://dagshub.com/TamariToradze/IEEE-CIS-Fraud-Detection_ass2/models/RandomForest/1)
- [AdaBoost მოდელი](https://dagshub.com/TamariToradze/IEEE-CIS-Fraud-Detection_ass2/models/AdaBoost/1)

 ექსპერიმენტები
[ყველა ექსპერიმენტის ნახვა](https://dagshub.com/TamariToradze/IEEE-CIS-Fraud-Detection_ass2/experiments)

