import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from .models import Review

def get_rmse(R, P, Q, non_zeros):
    error = 0
    
    # 2개의 분해된 행렬 P와 Q.T의 내적으로 예측 R 행렬 생성
    full_pred_matrix = np.dot(P, Q.T)
    
    # 실제 R 행렬에서 널이 아닌 값의 위치 index 추출해서 실제 R행렬과 예측 행렬의 RMSE(오차율) 추출
    x_non_zero_i = [non_zero[0] for non_zero in non_zeros]
    y_non_zero_i = [non_zero[1] for non_zero in non_zeros]
    R_non_zeros = R[x_non_zero_i, y_non_zero_i]
    full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_i, y_non_zero_i]
    mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
    rmse = np.sqrt(mse)
    
    return rmse


def matrix_factorization(R, K, steps=200, learning_rate=0.01, r_lambda=0.01):
    num_users, num_items = R.shape
    
    # P와 Q 행렬의 크기를 지정하고 정규 분포를 가진 랜덤한 값으로 입력
    np.random.seed(1)
    P = np.random.normal(scale=1./K, size=(num_users, K))
    Q = np.random.normal(scale=1./K, size=(num_items, K))
    
    prev_rmse = 10000
    break_count = 0
    
    # R>0인 행 위치, 열 위치, 값을 non_zeros list 객체에 저장
    non_zeros = [(i, j, R[i,j]) for i in range(num_users) for j in range(num_items) if R[i,j]>0]
    
    # SGD 기법으로 P와 Q 행렬을 반복 업데이트
    for step in range(steps):
        for i, j, r in non_zeros:
            # 실제 값과 예측 값의 차이인 오류 값 구하기
            eij = r - np.dot(P[i,:], Q[j,:].T)
            
            # 정규화를 반영한 SGD 업데이트 공식 적용
            P[i,:] = P[i,:] + learning_rate * (eij*Q[j,:] - r_lambda*P[i,:])
            Q[j,:] = Q[j,:] + learning_rate * (eij*P[i,:] - r_lambda*Q[j,:])
            
        rmse = get_rmse(R, P, Q, non_zeros)
        
        print("### iteration step:", step, " rmse:", rmse)
        
    return P, Q


review_list = Review.objects.all()
matrix = pd.DataFrame.from_records([c.to_dict() for c in review_list])
review_matrix = matrix.pivot_table('user_rate', index='user_nickname', columns='thema_name', aggfunc='first') # 사용자-테마 행렬

P, Q = matrix_factorization(review_matrix.values, K=50, steps=10, learning_rate=0.01, r_lambda=0.01)
pred_matrix = np.dot(P, Q.T)

ratings_pred_matrix = pd.DataFrame(data=pred_matrix, index=review_matrix.index, columns=review_matrix.columns)

def get_unplayed_themes(review_matrix, userName):
    # userId로 입력받은 사용자의 모든 영화 정보를 추출해 Series로 반환
    # 반환된 user_rating은 영화 제목을 인덱스로 가지는 Series 객체
    user_rating = review_matrix.loc[userName,:]
    
    # user_rating이 0보다 크면 기존에 관람한 영화. 대상 인덱스를 추출해서 list 객체로 만들기
    already_played = user_rating[user_rating>0].index.tolist()
    
    # 모든 영화 제목을 list 객체로 만들기
    themes_list = review_matrix.columns.tolist()
    
    # list comprehension으로 already_seen에 해당하는 영화는 movies_list에서 제외
    unplayed_list = [theme for theme in themes_list if theme not in already_played]
    
    return unplayed_list

def recomm_movie_by_userid(pred_df, userId, unseen_list, top_n=10):
    recomm_movies = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]
    
    return recomm_movies

# 사용자가 관람하지 않은 영화제목 추출
unplayed_list = get_unplayed_themes(review_matrix, ' 탈출출해')

# 잠재 요인 협업 필터링으로 영화 추천
recomm_themes = recomm_movie_by_userid(ratings_pred_matrix, ' 탈출출해', unplayed_list, top_n=5)

# 평점 데이터를 DataFrame으로 생성
recomm_themes = pd.DataFrame(data=recomm_themes.values, index=recomm_themes.index, columns=['pred_score'])

recomm_themes