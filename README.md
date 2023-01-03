# car_brand

차량의 연식과 차종을 인식하는 모델을 만들어보고자 함


1. 데이터허브에서 다운 받은 데이터를 split을 통해서 train데이터와 test데이터로 분리 시킴
2. view.py를 통해서 제대로 분리가 되었는지 확인
3. train.py에 있는 코드로 분리된 데이터들을 efficientNet으로 학습시킴-> 이 때 best_weight를 저장함
4. detect.py에 있는 코드와 train.py에서 찾은 best_weigth를 활용하여 테스트 데이터에서 원하는 값이 나오는 지 시각화
