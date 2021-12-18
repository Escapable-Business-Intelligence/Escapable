# Escapable

### 1. 배경  
방탈출 카페가 성행함에 따라 영업을 돕는 여러가지 부가서비스가 생겨났다.
그중 우리가 주목한 부분은 방탈출 카페 추천 서비스이다.
방탈출 카페를 이용하는 고객들의 만족도는 난이도, 분위기, 활동성등 여러가지 요인에 의해 영향을 받을 수 있다.
따라서 단순히 전체적인 평가의 평균이 추천의 기준이 될 수 없기 때문에 유저개개인을 위한 추천기능이 필요하다.
  
### 2. 목표
  - 방탈출 카페끼리 비교를 좀더 명확한 방식으로 할 수 있다.
      
  - 기본적인 만족도 예상 및 탈출 성공,실패를 동시에 제공하여 합리적인 선택을 가능하게 한다.
  
  - 취향에 맞춘 방탈출 카페 추천을 통해 **유저**의 방탈출 이용의 만족도를 높인다.
      
### 3. 구현방법
  - [**전국방탈출**](https://www.roomescape.co.kr/theme/detail.php?theme=578) 리뷰를 통한 서울 지역 방탈출 데이터 분석
  - Matrix factorization을 통한 추천 최적화
  - Latent Factor Collaborate Filtering을 이용한 추천 시스템 알고리즘 구축
  - Django 및 파이썬 + html을 통한 추천 사이트 구축

### 4. 참고자료
  - 전국방탈출 리뷰

-------------------
  ![image](https://user-images.githubusercontent.com/55437339/139399748-8942524c-7991-4beb-8526-d9cfb3c27b7e.png)
    
      
        
        
### 5. 웹 구성
![image](https://user-images.githubusercontent.com/55437339/146642418-0c59ad18-299b-41a0-a51c-975385e8409b.png)  
초록색 버튼을 누르면 설문조사 유형을 선택할 수 있다.  
  
![image](https://user-images.githubusercontent.com/55437339/146642447-6b018065-021c-4823-be36-12b96f1813a8.png)  
왼쪽 추천 형식은 테마 추천만, 오른쪽 추천 형식은 테마 추천 뿐만 아니라 탈출 성공과 체감 난이도 예측 결과까지 받아볼 수 있다.  
  
![image](https://user-images.githubusercontent.com/55437339/146642486-626bf43b-c63c-464a-9db2-c4897b51c606.png)  
경험해 본 테마와 그 평점만 남기면 테마를 추천받아볼 수 있다. (데이터가 많을수록 정확도가 높아진다.)  
  
![image](https://user-images.githubusercontent.com/55437339/146642521-cf124d65-f86c-48c8-a65e-f28a4b8f527b.png)  
경험해 본 테마와 그 평점, 체감 난이도, 탈출 성공 여부를 남기면 테마와 그 성공 여부, 체감 난이도 예측 결과도 추가로 받아볼 수 있다. (데이터가 많을수록 정확도가 높아진다.)  
    
      
        
        
 ### 6. 평가 방식
![image](https://user-images.githubusercontent.com/55437339/146642568-14dcc9f7-9c41-43a7-9e04-ce11c3ec505a.png)  
추천 대상자의 예측 평점 평가 결과  
  
![image](https://user-images.githubusercontent.com/55437339/146642653-cabf4857-1760-4801-ab74-37338dbe5e0b.png)  
추천 대상자의 탈출 성공 여부의 예측 결과 평가 결과  
  
![image](https://user-images.githubusercontent.com/55437339/146642658-43c321e8-2db6-4f26-8cd1-3935c4ca0d3f.png)  
추천 대상자의 체감 난이도의 예측 결과 평가 결과  
