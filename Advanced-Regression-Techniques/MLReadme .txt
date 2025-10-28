dagshub link:
https://dagshub.com/TamariToradze/ML_Advanced-Regression-Techniques

House Price Prediction - Kaggle Competition
კონკურსის მიმოხილვა
ეს პროექტი წარმოადგენს Kaggle-ის კონკურსის ნაწილს, სადაც მთავარი ამოცანაა საცხოვრებელი სახლების ფასების პროგნოზირება სხვადასხვა მახასიათებლების საფუძველზე.


მიდგომა პრობლემის გადასაჭრელად
Cleaning მიდგომები
1.Outlier-ების იდენტიფიკაცია და მოშორება Scatter Plot-ების გამოყენებით       
2.მაღალი უნიკალური მნიშვნელობების მქონე სვეტების (მაგ. ID) ამოღება
3.Nan მნიშვნელობების დამუშავება
ვიზუალიზაციის (ავაგე გრაგიკი, რომელიც აჩვენებს თუ რომელ სვეტში რამდენია NaN ველიუების პროცენტული შემცველობა) საშუალებით გამოვლენილია სვეტები მაღალი NaN მაჩვენებლით. სვეტები > 70% NaN მნიშვნელობებით ამოღებულია ('PoolQC', 'MiscFeature', 'Alley', 'Fence'...). დარჩენილი NaN მნიშვნელობები შევსებულია შესაბამისი მნიშვნელობებით ( 0 , საშუალო) 

Feature Engineering
1.რიცხვითი სვეტების განცალკევება არარიცხვითებისგან
2.კატეგორიული ცვლადების დამუშავება
One-hot encoding გამოყენებულია კატეგორიული ცვლადებისთვის
Ordinal encoding გამოყენებულია რიგითი კატეგორიული მონაცემებისთვის
3.დავამატე რამდენიმე მახასიათებელი , რომელიც აერთიანებდა ამოშლილ რამდენიმე სვეტს:
totalSquare, totalarea: სახლის საერთო ფართობი
remodelage, ageOfHouse: სახლის და რემონტის ასაკი
bathCount: აბაზანების საერთო რაოდენობა
porchsf: აივნის ფართობი
4.კომბინირებული ფიჩერების შექმნა (მაგ. სრული ფართობი, საერთო სააბაზანოების რაოდენობა, სახლის ფლობის წელი )


Feature Selection:
კორელაციის ანალიზის საფუძველზე ზედმეტად დაკავშირებული ფიჩერების ამოღება (მაგ. GarageArea)

Id: მხოლოდ საიდენტიფიკაციო ნომერია და არ აქვს პროგნოზირებისთვის სასარგებლო მნიშნელობა
Alley, PoolQC, Fence, MiscFeature: ბევრი გამოტოვებული მონაცემი აქვთ (NaN )
BsmtFinSF1, BsmtFinSF2: TotalBsmtSF უკვე მოიცავს ამ ინფორმაციას 
HalfBath, BsmtHalfBath: ნაკლებად მნიშვნელოვანია FullBath-თან შედარებით
GarageYrBlt, GarageQual, GarageCond: კორელირებულია სხვა გარაჟის მახასიათებლებთან
MiscVal: ბუნდოვანი მნიშვნელობის მქონეა

დავტოვე მნიშვნელოვანი მახასიათებლები:
ფართობთან დაკავშირებული (LotArea, GrLivArea და ა.შ.)
მდგომარეობისა და ხარისხის მაჩვენებლები (OverallQual, ExterQual და ა.შ.)
ასაკთან დაკავშირებული (YearBuilt, YearRemodAdd)
კატეგორიული ცვლადები (MSZoning, Neighborhood და ა.შ.)
გარაჟისა და აბაზანის მახასიათებლები




Training:
ტესტირებული მოდელები :Linear Regression, Lasso, Ridge, Polynomial, XGBoost.


Hyperparameter ოპტიმიზაციის მიდგომა:


Ridge და Lasso რეგრესია:
გამოყენებულია GridSearchCV ჰიპერპარამეტრების ოპტიმიზაციისთვის. პარამეტრების ძიება ხდება alpha და fit_intercept შორის, რაც საშუალებას იძლევა მივიღოთ საუკეთესო მოდელი.


Polynomial Regression:
 აქაც გამოვიყენე GridSearchCV პოლინომიური რეგრესიის ხარისხისა (degree) და fit_intercept პარამეტრების ოპტიმიზაციისთვის.


XGBoost:
გამოვიყენე GridSearchCV ობიექტი, საუკეთესო ჰიპერპარამეტრების მოსაძებნად. პარამეტრების ((max_depth, learning_rate, n_estimators) კომბინაციებიდან ვირჩევ საუკეთესოს.




საბოლოო მოდელის შერჩევა და დასაბუთება:


🔻Linear Regression - ყველაზე ცუდი შედეგი:
        უარყოფითი R² (-2.25e+17)
        ძალიან მაღალი RMSE (17,642,228.2)
        აშკარაა underfitting




🔻LASSO და Ridge - თითქმის იდენტური, ძალიან კარგი შედეგები:
        R² ≈ 0.89 (test), 0.93 (train)
        დაბალი RMSE ≈ 0.12 (test)
        კარგი CV score ≈ 0.916
        მინიმალური overfitting
        სტაბილური შედეგები




🔻Polynomial Regression:
        იდეალური ფიტი training data-ზე (R² = 1.0)
        ცუდი შედეგი test data-ზე (R² = 0.53)
        აშკარა overfitting


🔻XGBoost - საუკეთესო შედეგები:
        R² = 0.853
        ყველაზე დაბალი RMSE = 0.1424
        დაბალი MAPE = 0.77%
        კარგი ბალანსი accuracy-სა და generalization-ს შორის
 
🔻Logistic Regression-არ მიცდია, რადგან
                 ლოგისტიკური რეგრესია არ არის კარგი არჩევანი სახლის  გაყიდვის ფასების         პროგნოზირებისთვის (გვაქვს უწყვეტი მონაცემები და არა binary). ის სპეციალურად         შექმნილია ორობითი კლასიფიკაციის პრობლემებისთვის - სადაც შედეგი არის         კატეგორიული და აქვს მხოლოდ ორი შესაძლო მნიშვნელობა (მაგ., 0 ან 1, დიახ ან         არა... ა.შ.).




📌 Conclusion:
XGBoost არის საუკეთესო არჩევანი, რადგან:


1.აჩვენებს მაღალ სიზუსტეს
2.არ გვაქვს overfitting პრობლემა
3.აქვს ყველაზე დაბალი error metrics
4.არის უფრო მოქნილი და კარგად მუშაობს კომპლექსურ დამოკიდებულებებზე


ალტერნატივად შეგვიძლია განვიხილოთ LASSO ან Ridge, თუ გვჭირდება უფრო მარტივი და ინტერპრეტირებადი მოდელი.






-------------------------------------------------------------------------------------------------------------------------------


MLflow Tracking:
1.Linear Regression:
https://dagshub.com/TamariToradze/ML_Advanced-Regression-Techniques.mlflow/#/experiments/5/runs/e7f1da098a2e47cd952145daa2934a0e
R²        -2.25e+17  -ძალიან ცუდია
Adjusted R²        -6.96e+18 - ასევე ცუდია
MAE        15,318,137.8        🔻 მაღალი ცდომილება
MSE        3.11e+16        🔻 ძალიან მაღალია
RMSE        17,642,228.2        🔻 დიდი ერორი
AIC / BIC        12,651.6 / 13,362.2        ⚠️ მაღალია- ცუდი მოდელია...
F-statistic        86.58        ✅ 
F p-value        1.11e-16✅ 


გვაქვს underfitting არ არის საკმარისად კომპლექსური მოდელი. 
 შეგვიძლია ვცადოთ რეგულარიზაცია.
----------------------------------------------------------------------------
2. LASSO
https://dagshub.com/TamariToradze/ML_Advanced-Regression-Techniques.mlflow/#/experiments/7/runs/bf27934dbee84e27b189818ee7b4b9cb
Lasso Regression-ის შედეგები 


- cv_mean_score: 0.916
- cv_std_score: 0.014
ეს მიუთითებს კარგ მოდელზე თანმიმდევრული შესრულებით. 


- f_p_value: 1.21e-11
- f_statistic: 3.749
ძალიან დაბალი p-მნიშვნელობა  მიანიშნებს, რომ მოდელი არის statistically valid.


- test_r2: 0.887
- test_adj_r2: 0.65
- test_mae: 0.081
- test_mse: 0.016
- test_rmse: 0.125
MAE, MSE და RMSE დაბალია, რაც კარგ სიზუსტეს გულისხმობს.


- train_r2: 0.932
- train_adj_r2: 0.918
- train_mae: 0.074
- train_mse: 0.01
- train_rmse: 0.102
უკეთესი პერფორმანსი გვაქვს ტრეინინგ მონაცემებზე , რაც მოსალოდნელი იყო.
სერიოზული overfitting არ გვაქვს რადგან გეფი მცირეა.
📌 Conclusion:
საკმაოდ ზუსტია : გვაქვს მაღალი R² და მცირე error metrics.
სტაბილურია: მცირე std CV შედეგები
მცირე p-value რაც კარგია.
მცირე overfitting.
-----------------------------------------------------------
3.Ridge
https://dagshub.com/TamariToradze/ML_Advanced-Regression-Techniques.mlflow/#/experiments/6/runs/501f4fbf854740aba226f2334c7fdb3d


Test Set Metrics:
test_mse (Mean Squared Error):0.0156
test_rmse (Root Mean Squared Error):0.1251
test_mae (Mean Absolute Error):0.0801
test_r2 (R² Score):0.8866
test_adj_r2 (Adjusted R²):0.6501
გვაქვს მცირე ერორები (MSE, RMSE, MAE) და მაღალი  R²(1-თან ახლოს)  რაც კარგია.


Training Set Metrics:
train_mse: 0.0105
train_rmse: 0.1024
train_mae: 0.0742
train_r2: 0.9315
train_adj_r2: 0.9176
გვიჩვენებს, რამდენად კარგად ერგება მოდელი იმ მონაცემებს, რომლებზეც  დავატრენინგეთ.
გვაქვს კარგი პერფორმანსი და ოდნავ უკეთესი შედეგი, ვიდრე test set-ს.
მცირე გეფი= მინიმალური overfiting.


Statistical Metrics:
f_statistic: 3.7491
f_p_value: 1.206e-11
 მაღალი F-statistic და ძალიან დაბალი p-value = მოდელი არის statistically valid.


Cross-Validation Metrics:
cv_mean_score: 0.9156
cv_std_score: 0.0136
მაღალი mean CV score და low standard deviation = სტაბილური მოდელია.
great sign of consistency


📌 Conclusion:
კარგია LASSO ან Ridge რეგრესია - ორივე მოდელი თითქმის იდენტურ შედეგებს აჩვენებს. LASSO-ს უპირატესობა არის ის, რომ ის ახდენს feature selection-საც და ნულამდე ამცირებს ნაკლებად მნიშვნელოვან ფიჩერებს, რაც მოდელს უფრო მარტივს ხდის.]


---------------------------------------------------------------------------
4.Polynomial Regression:
https://dagshub.com/TamariToradze/ML_Advanced-Regression-Techniques.mlflow/#/experiments/10/runs/0ffe17cf45c143ea92f45ef7847c76e2


train_mse        ~1.17e-29
train_rmse        ~3.42e-15
train_r2        1.0
train_adj_r2        1.0
train_mae        ~2.51e-15
მოდელი კარგად ერგება training dataს, რაც მიანიშნებს overfitingზე


test_mae        0.1475
test_mse        0.0645
test_rmse        0.2540
test_r2        0.5324
test_adj_r2        0.5275
test მონაცემებზე აღარ გვაქვს კარგი შედეგი r2-ის მნიშვნელობა საკმაოდ დაბალია trainingთან შედარებით ,რაც მიუთითებს overfitting-ზე


cv_mean_score        0.8192
cv_std_score        0.0374
აქ კარგი შედეგია, შესაძლოა test set იყო unsual.


aic, bic        Negative
f_statistic        107.82
f_p_value        ~1e-16


📌 Conclusion:
ზედმეტად კომპლექსურია, რაც მოსალოდნელიც იყო.
მოდელმა ისწავლა training data ზედმეტად კარგად... შესაბამისად, კარგად აღარ მუშაობს test data-ზე. გვაქვს overfiting.
cross-validationის კარგი შედეგი აჩვენებს რომ შეიძლება test setში იყოს პრობელმა.


-------------------------------------------------------------------------------------


5.XGBoost
https://dagshub.com/TamariToradze/ML_Advanced-Regression-Techniques.mlflow/#/experiments/9/runs/38223eec33af47a1b44135eb75d4759c

R-squared (R2)  0.8668-  0.8-ზე მაღალია რაც კარგი შედეგია , მოდელი ხსნის მონაცემთა ვარიაციის 87%ს.


RMSE  0.1355- დაბალი მშვნელობაა, პრედიქშენი საშუალოდ 0.13ით სცდება რეალურ მნიშვნელობებს, რაც კარგია ასევე.


MAE 0.0854- პროგნოზი 0.085იტ სცდება, რაც კიდევ უფრო კარგი შედეგია.


MAPE  0.0072 (0.72%) - ესეც ძალიან დაბალი და ძალიან კარგია...


Explained Variance = 0.8670  კარგია


📌 Conclusion:
მოდელის მეტრიკები აჩვენებს ძალიან კარგ შედეგებს. დაბალი ცდომილების მაჩვენებლები (RMSE, MAE, MAPE) და მაღალი  R2, Explained Variance მიუთითებს, რომ მოდელი კარგად ერგება მონაცემებს და აქვს მაღალი პრედიქციული სიზუსტე.

Model Registry:
https://dagshub.com/TamariToradze/ML_Advanced-Regression-Techniques.mlflow/#/models
