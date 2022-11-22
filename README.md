# Image Classification Project(Naver Webtoon)


### ○ 프로젝트 주제 
딥툰 개발을 위한 그림체 인식<br> 
(※딥툰 : 작가가 시나리오를 입력하면 자신에 맞는 장르의 그림체를 그려주는 웹툰 자동 생성 기술)

### ○ 프로젝트 기간
22.08.08. ~ 22.08.19.

### ○ 프로젝트 목적
네이버의 웹툰의 그림체를 이용해 장르를 구별하게 이미지 분류 

### ○ 데이터셋
네이버의 10개의 장르에 있는 모든 웹툰의 모든 회차 썸네일 이미지 , 작가, 별점, 장르 크롤링<br>
총, 82549 썸네일을 데이터셋으로 사용

<table>
    <thead>
        <tr>
            <th>label</th>
            <th>feature</th>
            <th>label</th>
            <th>feature</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>판타지</td>
            <td>19000</td>
            <td>액션</td>
            <td>5994</td>
        </tr>
        <tr>
            <td>드라마</td>
            <td>15782</td>
            <td>스릴러</td>
            <td>4885</td>
        </tr>
        <tr>
            <td>순정</td>
            <td>13710</td>
            <td>무협</td>
            <td>3122</td>
        </tr>
        <tr>
            <td>코믹</td>
            <td>9242</td>
            <td>스포츠</td>
            <td>2383</td>
        </tr>
        <tr>
            <td>데일리</td>
            <td>7330</td>
            <td>감성</td>
            <td>1101</td>
        </tr>
  </tbody>
</table>

### ○ 전처리
1.이미지 112X112사이즈로 resize, 패딩을 넣어주어 이미지 훼손을 줄임<br>

        for label, filenames in dataset.items():
            for filename in filenames:
                img = cv2.imread(filename) # cv2.imread(filename = 파일경로)

                    # 이미지의 x, y가 112이 넘을 경우 작게해주기
                    percent = 1
                    if(img.shape[1] > img.shape[0]) :       # 이미지의 가로가 세보다 크면 가로를 112으로 맞추고 세로를 비율에 맞춰서
                        percent = 112/img.shape[1]
                    else :
                        percent = 112/img.shape[0]

                    img = cv2.resize(img, dsize=(0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_LINEAR)
                            # 이미지 범위 지정
                    y,x,h,w = (0,0,img.shape[0], img.shape[1])

                    # 그림 주변에 검은색으로 칠하기
                    w_x = (112-(w-x))/2  # w_x = (112 - 그림)을 뺀 나머지 영역 크기 [ 그림나머지/2 [그림] 그림나머지/2 ]
                    h_y = (112-(h-y))/2

                    if(w_x < 0):         # 크기가 -면 0으로 지정.
                        w_x = 0
                    elif(h_y < 0):
                        h_y = 0

                    M = np.float32([[1,0,w_x], [0,1,h_y]])  #(2*3 이차원 행렬)
                    img_re = cv2.warpAffine(img, M, (112, 112)) #이동변환

                    # cv2.imwrite('{0}.jpg',image .format(file)) #파일저장
                    cv2.imwrite('/content/resized/{0}/{1}'.format(label, filename.split("/")[-1]) , img_re)
2. 장르별 라벨링<br>

        label2index = {'daily' : 0, 'comic' : 1 , 'fantasy' : 2 , 'action' : 3,
                       'drama' : 4, 'pure' : 5, 'sensibility' : 6, 'thrill' : 7, 'historical' : 8, 'sports' : 9}

3. Zero Centering<br>
        
        # zero-centering
        def zero_mean(image):
            return np.mean(image, axis=0)

        zero_mean_img = zero_mean(x_train)
        zero_mean_img.shape

        x_train -= zero_mean_img
        x_val -= zero_mean_img
        x_test -= zero_mean_img
        
### ○ 모델핸들링 & 모델선정
- 모델 : ResNet50, InceptionV3, VGG19, VGG16 사용<br>
- 모델 핸들링 : CNN Layers(완전개방, 10층 개방, 20층 개방, 30층 개방, 40층 개방, 완전개방)<br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dense(128,256,512)<br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Learning Rate(0.001, 0.0001, 0.00001)<br>

<div><img width="500" alt="modelselection" src="https://user-images.githubusercontent.com/79880476/203184863-f7acc46f-9bfb-4275-b96e-1a63a6bc92b4.jpg"><p>- 모델핸들링을 통해 통제조건 속에서 ResNet50이 전반적으로 성능이 제일 잘 나왔다.</p></div>

- 미세조정(Fine-tune)
<div><img width="500" alt="finetune" src="https://user-images.githubusercontent.com/79880476/203184899-9ea6717f-2237-4756-8287-eef65bd5b62f.jpg"><p>- 제일 잘 나왔던 여러 통제 조건에서 CNN Layer만 다르게 줬을 때 Accuracy 0.5478로 성능이 제일 잘나왔다.</p></div>

### ○ 결과
- 여러가지 전처리와 모델 핸들링을 통해 최대 Accuracy 0.5478가 나왔다.<br>

### ○ 자체 피드백 
- 비슷한 썸네일에 대한 모호한 라벨링 (개그? 일상? / 감성 ? 순정?)<br>
<div><img width="300" alt="labels" src="https://user-images.githubusercontent.com/79880476/203188333-87e44364-6f0e-4b71-8496-8c6d51c0f473.jpg"><p>- 사람 눈으로도 분류 불가능한 모호한 장르가 모델의 성능에도 문제를 끼쳤다고 판단했다.</p></div>

- 데이터양의 불균형<br>
<div><img width="500" alt="imbalance" src="https://user-images.githubusercontent.com/79880476/203188322-6fc22123-16b8-45fc-a039-3f09ec34f4ee.jpg"><p>- 다양성을 위해 모든 장르의 썸네일을 크롤링했지만 데이터가 워낙 불균형해 해소하면 성능이 더 오를 것이라 판단했다.</p></div>

### ○ 성능개선의 노력
- 데이터양의 불균형<br>
<div><img width="500" alt="imbalance" src="https://user-images.githubusercontent.com/79880476/203188322-6fc22123-16b8-45fc-a039-3f09ec34f4ee.jpg"><p>- 다양성을 위해 모든 장르의 썸네일을 크롤링했지만 데이터가 워낙 불균형해 해소하면 성능이 더 오를 것이라 판단했다.</p></div>

- 데이터양의 불균형<br>
<div><img width="500" alt="imbalance" src="https://user-images.githubusercontent.com/79880476/203188322-6fc22123-16b8-45fc-a039-3f09ec34f4ee.jpg"><p>- 다양성을 위해 모든 장르의 썸네일을 크롤링했지만 데이터가 워낙 불균형해 해소하면 성능이 더 오를 것이라 판단했다.</p></div>

### ○ 결론

### ○ 참고 논문


### ○ 역할
- 김용재 : 
- 이훈석 : 
- 오혜인 : 
- 석민재 : 
- 한서연 : 
- 정효제 : 
