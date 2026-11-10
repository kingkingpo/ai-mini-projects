# 004. 모델 실험 및 오류 분석

**프로젝트:** 온라인 쇼핑 구매 여부 예측 (PyTorch 신경망 분류)  
**작성 범위:** CP-04~CP-08

> Notebook에서 확인한 실제 수치와 관찰을 기록합니다. `___`는 실제 값 또는 1~2문장으로 바꾸세요. Precision·Recall·F1은 구매 클래스(`Revenue=1`) 기준입니다.

## 1. 기준 모델 및 평가 지표 (CP-04)

- **양성 클래스:** 구매 (`Revenue=1`)
- **주 평가 지표:** F1 (구매 클래스)
- **보조 지표:** Precision, Recall, Accuracy
- **모델 비교에 사용한 데이터:** Train, Validation
- **DummyClassifier — Validation Accuracy / F1:** 0.845 / 0.00
- **LogisticRegression — Train F1 / Validation F1:** 0.505 / 0.49
- **LogisticRegression — Validation Precision / Recall:** 0.734 / 0.367
- **LogisticRegression의 Validation 혼동행렬:** TN 1526 / FP 38 / FN 181 / TP 105

**분석:** 불균형 데이터에서 기준 모델의 Accuracy와 F1이 다르게 나타나는 이유를 작성합니다.  
DummyClassifier(strategy="most_frequent")가 다수 클래스인 음성만 예측한다. 그래서 F1 값도 확인해서 이 모델을 써도 되는지 확인해야 한다. Precision과 Recall이 0 이므로 F1 = 2 × Precision × Recall ÷ (Precision + Recall) 공식을 쓰면 F1 = 2 × 0 × 0 ÷ (0 + 0) 로 0이 나온다. 
___

## 2. PyTorch MLP-A 구현 및 학습 (CP-05)

- **신경망 구조:** 입력 → 64 → 32 → 1 (ReLU, Dropout 0.1)
- **손실 함수 / 옵티마이저:** `BCEWithLogitsLoss` / Adam
- **학습 조건:** 최대 25 epoch, batch size 256, learning rate 0.001
- **Validation F1이 가장 높았던 에포크:** 25
- **선택된 MLP-A의 Train F1 / Validation F1:** 0.709 / 0.633
- **MLP-A Validation Precision / Recall:** 0.717 /0.566

**분석:** Train loss와 Train·Validation F1 그래프에서 확인한 학습 경향을 작성합니다.  
 에포크 값이 늘어나며 Train loss 가 급격히 감소한다. 에포크가 커지면서 처음엔 Train·Validation F1의 점수가 비슷하게 늘어나지만 어느 정도 에포크가 커지면 과대적합이 생겨서 Train 값이 Validation 값 보다 많이 커진다. 
___

## 3. 클래스 가중 손실과 MLP 비교 (CP-06)

- **MLP-B 변경 사항:** 양성 클래스 가중치(`pos_weight`)를 적용한 `BCEWithLogitsLoss`
- **Train 데이터의 `pos_weight`(미구매 수 ÷ 구매 수):** 5.46
- **MLP-B의 최적 에포크(최고 Validation F1 기준):** 17
- **MLP-B Train F1 / Validation F1:** 0.692 / 0.645
- **MLP-B Validation Precision / Recall:** 0.575 / 0.734
- **MLP-A와 MLP-B의 Validation F1 차이:** 0.633 - 0.645 = - 0.012 
- **LogisticRegression과 두 MLP의 성능 비교에서 발견한 점:** LogisticRegression보다 두 MLP의 Validation F1이 높았다. 
- **Train·Validation F1 차이에서 관찰한 점:** DummyClassifier 는 0 이지만 Precision, Recall 이 0 이라 나온 값이기에 의미가 없다. LogisticRegression가 두 MLP 보다 Train·Validation F1 차이가 낮다.

**분석:** 양성 클래스 가중치를 적용한 뒤 Precision·Recall·F1이 어떻게 달라졌는지 작성합니다.  
MLP-B는 MLP-A보다 Precision이 0.717에서 0.575로 낮아졌지만, Recall은 0.566에서 0.734로 높아졌고, F1도 0.633에서 0.645로 소폭 상승했다. 그래서 Recall이 더 중요하면 MLP-B 가 낫다고 생각한다. 
___

> 전체 실험 결과는 Notebook의 `results_df`와 `results/02_model_comparison.csv`를 참고합니다.

## 4. 예측 오류 및 임계값 분석 (CP-07)

- **비교한 두 신경망:** MLP-A / MLP-B
- **기본 임계값 0.5에서 MLP-A의 FP / FN:** 64 / 124
- **기본 임계값 0.5에서 MLP-B의 FP / FN:** 155 / 76
- **임계값 0.3에서 Precision·Recall 및 FP·FN 변화:** 임계값 0.5때보다 Precision, FN은 낮아지고 Recall, FP는 높아졌다. 
- **임계값 0.7에서 Precision·Recall 및 FP·FN 변화:** 임계값 0.5때보다 Precision, FN은 높아지고 Recall, FP는 낮아졌다. 
- **Validation F1이 가장 높은 MLP·임계값 조합:** MLP-B (weighted BCE) 임계값 0.5	
- **선택한 MLP의 최적 에포크 수:** 17
- **(선택) PR 곡선 또는 추가 임계값 실험에서 확인한 점:** 임계값이 낮아지면 Precision, FN은 낮아지고 Recall, FP는 높아졌다. 임계값이 높아지면 그 반대로 움직였다.

**분석:** 모델과 임계값의 선택에서 Precision·Recall 및 FP·FN을 어떻게 고려했는지 작성합니다. 
 MLP-B, 임계값 0.5를 선택한 이유는 구매 고객을 더 많이 찾아내면서(FN 76) F1도 가장 높았다. 다만 FP가 155건으로 오탐이 적지는 않다. 임계값을 0.7로 올리면 FP는 99건으로 줄지만 FN이 111건으로 늘고, 0.3으로 낮추면 FN은 46건으로 줄지만 FP가 283건까지 증했다. 따라서 구매 고객을 놓치는 것과 오탐을 늘리는 것 사이의 균형을 고려해 F1이 가장 높은 조합을 택했다.
___

> 임계값별 상세 결과는 `threshold_df`와 `results/02_threshold_comparison.csv`를 참고합니다.

## 5. 최종 신경망과 Test 평가 (CP-08)

- **Test 확인 전에 확정한 MLP 설정 / 임계값:** MLP-B / 0.5
- **최종 재학습 데이터(Train + Validation):** 10480행
- **최종 재학습 에포크 수 / 가중 손실 적용 여부:** 17 / 적용
- **Test Accuracy / Precision / Recall / F1:** 0.859 / 0.527 / 0.843 / 0.649
- **Test 혼동행렬:** TN 1348 / FP 216 / FN 45 / TP 241
- **모델 번들(`02_final_classifier.joblib`)·설정 JSON 저장 및 재로딩 확인:** 완료(저장 전후 예측 일치: True)
- **`predict_purchase()` 실행 결과:** 구매 확률 0.0058, 예측 0(미구매), 임계값 0.5 실제 Revenue도 0

**해석:** 최종 Test 결과와 동일한 모델·임계값의 Validation 결과를 비교해 알게 된 점을 작성합니다.
동일한 MLP-B와 임계값 0.5를 사용했을 때 Test F1은 0.649로 Validation F1(0.645)과 비슷했다. Test에서는 Recall이 더 높아 실제 구매 고객을 더 많이 찾아냈지만, Precision은 낮고 FP는 늘어 오탐이 많아졌다.  
___

> Test 결과는 모델 선택에 사용하지 않습니다. 최종 Test 수치는 Notebook과 이 문서에만 기록합니다.

## 6. 실험 요약

- **가장 중요한 실험 결과:** MLP-B가 Validation F1 0.645로 가장 높았고, Test에서도 F1 0.649를 기록했다. 가중치 적용으로 구매 고객을 찾는 Recall이 높아졌다.
- **남아 있는 모델의 한계:** Test Recall은 0.843으로 높았지만 Precision은 0.527이고 FP가 216건 발생해, 미구매 고객을 구매로 잘못 예측하는 경우가 많았다.

> **최종 MLP 설정·임계값의 선택과 근거**는 `adr.md`의 의사결정 ④에 작성합니다.
