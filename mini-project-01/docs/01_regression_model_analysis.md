# 002. 모델 실험 및 오류 분석

**프로젝트:** 공유자전거 대여 수요 예측 (회귀)  
**작성 범위:** CP-04~CP-08

> Notebook에서 확인한 실제 성능과 관찰을 작성합니다. 각 항목의 `___`를 실제 값이나 1~2문장으로 바꾸세요.

## 1. 모델 평가 기준 (CP-04)

- **주 평가 지표:** MAE
- **보조 지표:** RMSE, R²
- **모델 비교에 사용한 데이터:** Train과 Validation 데이터를 모두 평가해 성능을 기록했으며, 모델 비교는 Validation 성능을 기준으로 했다
- **MAE를 주요 지표로 선택한 이유:** 예측값이 실제값에서 평균적으로 얼마나 벗어났는지 쉽게 알수 있다. RMSE는 큰 오차를 보고싶을 때 사용한다.

## 2. 모델별 실험 결과 (CP-04~CP-06)

대표 설정별 Train MAE / Validation MAE를 기록합니다. 전체 실험 결과는 Notebook의 `results_df`에서 확인합니다.

- **DummyRegressor (mean):** Train 119.572 / Validation 194.109
- **LinearRegression:** Train 55.425  / Validation 99.707
- **DecisionTreeRegressor (`max_depth=12`):** Train 32.728  / Validation 80.659
- **RandomForestRegressor (설정  n_estimators=100, max_depth=15):** Train 22.155 / Validation 64.323

**실험에서 발견한 점**

- **Decision Tree 깊이 변화에 따른 성능:** max_depth를 3에서 12로 늘리자 Train MAE는 81.203에서 32.728로, Validation MAE는 134.941에서 80.659로 낮아졌다. 트리 깊이가 깊어질수록 두 데이터에서 성능이 좋아졌다.
- **Random Forest 두 설정의 차이:** n_estimators=100, max_depth=15 설정은 n_estimators=50, max_depth=10보다 Validation MAE가 76.444에서 64.323으로 낮아졌다. RMSE도 더 낮아졌고 R2는 1에 더 가까워졌다.
- **Train·Validation 성능 차이에서 관찰한 현상:** 모든 모델에서 Train MAE보다 Validation MAE가 높아, 학습 데이터보다 새로운 기간의 대여량을 예측할 때 오차가 커졌다. Validation의 cnt 평균(281.04)이 Train(159.72)보다 높아 시기별 수요 수준이 달랐으며, 이 차이가 성능에 영향을 주었을 가능성이 있다. (`001_data_analysis`의 2절 `cnt` 평균 차이와 연결해 작성)

## 3. 예측 오류 분석 (CP-07)

- **비교한 후보 모델 2개(Validation MAE 상위 2개):** RandomForestRegressor (n_estimators=100, max_depth=15) / DecisionTreeRegressor (max_depth=12)
- **실제값·예측값 그래프에서 관찰한 점:** 예측값은 실제값이 커질수록 함께 커지는 경향이 있지만, 모든 점이 완벽한 예측 기준선에 놓이지는 않아 오차가 남았다.
- **시간대별 MAE가 큰 구간:** 두 모델 모두 오후 5~6시(17~18시)에 가장 컸다. Random Forest는 17시 120.9, 18시 112.5였고, Decision Tree는 17시 129.8, 18시 128.5였다.
- **모델이 상대적으로 잘 예측하지 못하는 상황:**  퇴근 시간대처럼 대여량이 높고 수요가 크게 변하는 시간대.
- **(선택) 특성 중요도에서 확인한 점:** Permutation importance 결과, is_commute_hour를 섞었을 때 MAE 증가량이 가장 컸고(평균 91.908), hr(65.403), hum(38.019), workingday(21.369)가 그 뒤를 이었다. 따라서 이 모델의 Validation 예측에서는 출퇴근 시간 여부와 시간대가 특히 중요한 역할을 한 것으로 보인다.

**분석:** 이 오류가 최종 모델 선택에 어떤 영향을 주었는지 작성합니다. 
시간대별로 두 후보 모델 모두 17~18시에 오차가 가장 컸지만, Random Forest는 전체 Validation MAE가 Decision Tree보다 낮고 대부분의 시간대에서도 오차가 작았다. 따라서 최종 모델로 Random Forest(n_estimators=100, max_depth=15)를 선택하되, 퇴근 시간대 예측에는 오차가 클 수 있음을 고려했다. 
___

## 4. 최종 모델과 Test 평가 (CP-08)

- **Test 확인 전에 선택한 최종 모델·설정:** RandomForestRegressor(n_estimators=100, max_depth=15)
- **재학습에 사용한 데이터(Train + Validation) 크기:**  14755행
- **Test MAE:** 66.57849814333602
- **Test RMSE:** 95.91219619679921
- **Test R²:** 0.8011613057697564
- **모델 저장·재로딩 및 예측 결과:** 저장 전후 예측 일치: True

**해석:** 최종 Test 결과로 알게 된 점을 1~2문장으로 작성합니다.  
Test에서는 MAE와 RMSE가 조금 높고 R²는 조금 낮아, Validation보다 예측 성능이 다소 떨어졌지만 큰 차이는 아니다. 최종 모델이 처음 보는 Test 데이터에도 비슷한 수준의 성능을 보였다고 해석할 수 있습니다.
___

> Test는 최종 평가용입니다. 서로 다른 데이터 집합에서 나온 점수를 직접 개선율로 비교하지 않습니다.

## 5. 실험 요약

- **가장 중요한 실험 결과:** Random Forest가 후보 모델 중 Validation MAE가 가장 낮았다
- **남아 있는 모델의 한계:** 17~18시 퇴근 시간대의 오차가 다른 시간보다 컸다. Train 과 Validation의 cnt 값이 차이 났다.

> 최종 모델을 **왜 선택했는지**는 `adr.md`의 의사결정 ②에 작성합니다.
