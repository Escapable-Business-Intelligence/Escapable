from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import JsonResponse

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from .models import Review
from .models import Cafe
from .models import Thema


def index(request):
    return render(request, 'index.html')


def detail(request, cafe_index):
    return HttpResponse("You're looking at cafe %s." % cafe_index)


def recommend(request):
    return render(request, 'recommend.html')


def recommend1(request):
    cafe_list = Cafe.objects.all()

    context = {
        'cafe_list': cafe_list,
    }

    return render(request, 'recommend1.html', context)


def recommend2(request):
    cafe_list = Cafe.objects.all()

    context = {
        'cafe_list': cafe_list,
    }

    return render(request, 'recommend2.html', context)


@csrf_exempt
def selectThema(request):
    cafeName = json.loads(request.body)

    thema_list = Thema.objects.filter(cafe_name=cafeName)
    thema_list = list(thema_list.values())

    return JsonResponse(thema_list, safe=False)


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


def matrix_factorization_rate(R, K, steps=200, learning_rate=0.01, r_lambda=0.01):
    num_users, num_items = R.shape

    # P와 Q 행렬의 크기를 지정하고 정규 분포를 가진 랜덤한 값으로 입력
    np.random.seed(1)
    P = np.random.normal(scale=1. / K, size=(num_users, K))
    Q = np.random.normal(scale=1. / K, size=(num_items, K))

    prev_rmse = 10000
    break_count = 0

    # R>0인 행 위치, 열 위치, 값을 non_zeros list 객체에 저장
    non_zeros = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R[i, j] > 0]

    # SGD 기법으로 P와 Q 행렬을 반복 업데이트
    for step in range(steps):
        for i, j, r in non_zeros:
            # 실제 값과 예측 값의 차이인 오류 값 구하기
            eij = r - np.dot(P[i, :], Q[j, :].T)

            # 정규화를 반영한 SGD 업데이트 공식 적용
            P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda * P[i, :])
            Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda * Q[j, :])

        rmse = get_rmse(R, P, Q, non_zeros)

        print("### iteration step:", step, " rmse:", rmse)

    return P, Q


def matrix_factorization_Escape(R, K, steps=200, learning_rate=0.01, r_lambda=0.01):
    R = np.array(R, dtype=float)
    num_users, num_items = R.shape
    # P와 Q 행렬의 크기를 지정하고 정규 분포를 가진 랜덤한 값으로 입력
    np.random.seed(1)
    P = np.random.normal(scale=1. / K, size=(num_users, K))
    Q = np.random.normal(scale=1. / K, size=(num_items, K))

    prev_rmse = 10000
    break_count = 0

    # R>0인 행 위치, 열 위치, 값을 non_zeros list 객체에 저장
    str_non_zeros = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if
                     R[i, j] == 1 or R[i, j] == -1]
    for i in range(num_users):
        for j in range(num_items):
            if R[i, j] == 1:
                R[i, j] = 1
            elif R[i, j] == -1:
                R[i, j] = 0
    non_zeros = []
    for i, j, r in str_non_zeros:
        if r == 1:
            non_zeros.append((i, j, 1))
        elif r == -1:
            non_zeros.append((i, j, 0))
    # SGD 기법으로 P와 Q 행렬을 반복 업데이트
    for step in range(steps):
        for i, j, r in non_zeros:
            # 실제 값과 예측 값의 차이인 오류 값 구하기
            eij = r - np.dot(P[i, :], Q[j, :].T)

            # 정규화를 반영한 SGD 업데이트 공식 적용
            P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda * P[i, :])
            Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda * Q[j, :])

        rmse = get_rmse(R, P, Q, non_zeros)

        print("### iteration step:", step, " rmse:", rmse)

    return P, Q


def matrix_factorization_Escape(R, K, steps=200, learning_rate=0.01, r_lambda=0.01):
    R = np.array(R, dtype=float)
    num_users, num_items = R.shape
    # P와 Q 행렬의 크기를 지정하고 정규 분포를 가진 랜덤한 값으로 입력
    np.random.seed(1)
    P = np.random.normal(scale=1. / K, size=(num_users, K))
    Q = np.random.normal(scale=1. / K, size=(num_items, K))

    prev_rmse = 10000
    break_count = 0

    # R>0인 행 위치, 열 위치, 값을 non_zeros list 객체에 저장
    str_non_zeros = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if
                     R[i, j] == 1 or R[i, j] == -1]
    for i in range(num_users):
        for j in range(num_items):
            if R[i, j] == 1:
                R[i, j] = 1
            elif R[i, j] == -1:
                R[i, j] = 0
    non_zeros = []
    for i, j, r in str_non_zeros:
        if r == 1:
            non_zeros.append((i, j, 1))
        elif r == -1:
            non_zeros.append((i, j, 0))
    # SGD 기법으로 P와 Q 행렬을 반복 업데이트
    for step in range(steps):
        for i, j, r in non_zeros:
            # 실제 값과 예측 값의 차이인 오류 값 구하기
            eij = r - np.dot(P[i, :], Q[j, :].T)

            # 정규화를 반영한 SGD 업데이트 공식 적용
            P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda * P[i, :])
            Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda * Q[j, :])

        rmse = get_rmse(R, P, Q, non_zeros)

        print("### iteration step:", step, " rmse:", rmse)

    return P, Q


def matrix_factorization_difficulty(R, K, steps=200, learning_rate=0.01, r_lambda=0.01):
    # R = np.array(R,dtype = float)
    num_users, num_items = R.shape
    # P와 Q 행렬의 크기를 지정하고 정규 분포를 가진 랜덤한 값으로 입력
    np.random.seed(1)
    P = np.random.normal(scale=1. / K, size=(num_users, K))
    Q = np.random.normal(scale=1. / K, size=(num_items, K))

    prev_rmse = 10000
    break_count = 0

    # R>0인 행 위치, 열 위치, 값을 non_zeros list 객체에 저장
    str_non_zeros = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R[i, j] != 0]
    for i in range(num_users):
        for j in range(num_items):
            if R[i, j] != 0:
                if R[i, j] == '매우 쉬움':
                    R[i, j] = 1
                elif R[i, j] == '쉬움':
                    R[i, j] = 2
                elif R[i, j] == '보통':
                    R[i, j] = 3
                elif R[i, j] == '어려움':
                    R[i, j] = 4
                elif R[i, j] == '매우 어려움':
                    R[i, j] = 5
    non_zeros = []
    for i, j, r in str_non_zeros:
        if r == '매우 쉬움':
            non_zeros.append((i, j, 1))
        elif r == '쉬움':
            non_zeros.append((i, j, 2))
        elif r == '보통':
            non_zeros.append((i, j, 3))
        elif r == '어려움':
            non_zeros.append((i, j, 4))
        elif r == '매우 어려움':
            non_zeros.append((i, j, 5))

    # SGD 기법으로 P와 Q 행렬을 반복 업데이트
    for step in range(steps):
        for i, j, r in non_zeros:
            # 실제 값과 예측 값의 차이인 오류 값 구하기
            eij = r - np.dot(P[i, :], Q[j, :].T)

            # 정규화를 반영한 SGD 업데이트 공식 적용
            P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda * P[i, :])
            Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda * Q[j, :])

        rmse = get_rmse(R, P, Q, non_zeros)

        print("### iteration step:", step, " rmse:", rmse)

    return P, Q


def get_unplayed_themes_rate(review_matrix, userName):
    # userId로 입력받은 사용자의 모든 영화 정보를 추출해 Series로 반환
    # 반환된 user_rating은 영화 제목을 인덱스로 가지는 Series 객체
    user_rating = review_matrix.loc[userName, :]

    # user_rating이 0보다 크면 기존에 관람한 영화. 대상 인덱스를 추출해서 list 객체로 만들기
    already_played = user_rating[user_rating > 0].index.tolist()

    # 모든 영화 제목을 list 객체로 만들기
    themes_list = review_matrix.columns.tolist()

    # list comprehension으로 already_seen에 해당하는 영화는 movies_list에서 제외
    unplayed_list = [theme for theme in themes_list if theme not in already_played]

    return unplayed_list


def recomm_thema_by_userid_rate(pred_df, userId, unseen_list, top_n=10):
    print(pred_df)
    print(type(pred_df))
    recomm_thema = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]
    return recomm_thema


def matrix_factorization_Escape(R, K, steps=200, learning_rate=0.01, r_lambda=0.01):
    R = np.array(R, dtype=float)
    num_users, num_items = R.shape
    # P와 Q 행렬의 크기를 지정하고 정규 분포를 가진 랜덤한 값으로 입력
    np.random.seed(1)
    P = np.random.normal(scale=1. / K, size=(num_users, K))
    Q = np.random.normal(scale=1. / K, size=(num_items, K))

    prev_rmse = 10000
    break_count = 0

    # R>0인 행 위치, 열 위치, 값을 non_zeros list 객체에 저장
    str_non_zeros = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if
                     R[i, j] == 1 or R[i, j] == -1]
    for i in range(num_users):
        for j in range(num_items):
            if R[i, j] == 1:
                R[i, j] = 1
            elif R[i, j] == -1:
                R[i, j] = 0
    non_zeros = []
    for i, j, r in str_non_zeros:
        if r == 1:
            non_zeros.append((i, j, 1))
        elif r == -1:
            non_zeros.append((i, j, 0))
    # SGD 기법으로 P와 Q 행렬을 반복 업데이트
    for step in range(steps):
        for i, j, r in non_zeros:
            # 실제 값과 예측 값의 차이인 오류 값 구하기
            eij = r - np.dot(P[i, :], Q[j, :].T)

            # 정규화를 반영한 SGD 업데이트 공식 적용
            P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda * P[i, :])
            Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda * Q[j, :])

        rmse = get_rmse(R, P, Q, non_zeros)

        print("### iteration step:", step, " rmse:", rmse)

    return P, Q

def Predict_Escape(pred_df, user_Id, datas):
    print(pred_df)
    data_list = list()
    for data in datas:
        print(user_Id,"   " ,data)
        if pred_df.loc[user_Id,data] < 0.5:
            data_list.append('탈출 실패')
        else:
            data_list.append('탈출 성공')
    return data_list
def Predict_Difficulty(pred_df, user_Id, datas):
    print(pred_df)
    data_list = list()
    for data in datas:
        print(user_Id,"   " ,data)
        if pred_df.loc[user_Id,data] < 0.5:
            data_list.append('매우쉬움')
        elif pred_df.loc[user_Id,data] < 1.5:
            data_list.append('쉬움')
        elif pred_df.loc[user_Id,data] < 2.5:
            data_list.append('보통')
        elif pred_df.loc[user_Id,data] < 3.5:
            data_list.append('어려움')
        else:
            data_list.append('매우어려움')
    return data_list

def get_unplayed_themes_rate(review_matrix, userName):
    # userId로 입력받은 사용자의 모든 영화 정보를 추출해 Series로 반환
    # 반환된 user_rating은 영화 제목을 인덱스로 가지는 Series 객체
    user_rating = review_matrix.loc[userName, :]

    # user_rating이 0보다 크면 기존에 관람한 영화. 대상 인덱스를 추출해서 list 객체로 만들기
    already_played = user_rating[user_rating > 0].index.tolist()

    # 모든 영화 제목을 list 객체로 만들기
    themes_list = review_matrix.columns.tolist()

    # list comprehension으로 already_seen에 해당하는 영화는 movies_list에서 제외
    unplayed_list = [theme for theme in themes_list if theme not in already_played]

    return unplayed_list


def recomm_thema_by_userid_rate(pred_df, userId, unseen_list, top_n=10):
    recomm_thema = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]

    return recomm_thema


@csrf_exempt
def recommendThema1(request):
    data_list = json.loads(request.body)

    review_list = Review.objects.all()
    matrix = pd.DataFrame.from_records([c.to_dict() for c in review_list])

    for i in range(len(data_list)):
        review_list = data_list[i]
        data = {'thema_name': review_list['thema_name'], 'user_rate': float(review_list['user_rate']),
                'user_nickname': 'ESCAPABLE'}
        matrix = matrix.append(data, ignore_index=True)

    review_matrix = matrix.pivot_table('user_rate', index='user_nickname', columns='thema_name',
                                       aggfunc='first')  # 사용자-테마 행렬

    P, Q = matrix_factorization_rate(review_matrix.values, K=50, steps=10, learning_rate=0.01, r_lambda=0.01)
    pred_matrix = np.dot(P, Q.T)

    ratings_pred_matrix = pd.DataFrame(data=pred_matrix, index=review_matrix.index, columns=review_matrix.columns)

    # 사용자가 관람하지 않은 영화제목 추출
    unplayed_list = get_unplayed_themes_rate(review_matrix, 'ESCAPABLE')

    # 잠재 요인 협업 필터링으로 영화 추천
    recomm_themes = recomm_thema_by_userid_rate(ratings_pred_matrix, 'ESCAPABLE', unplayed_list, top_n=6)

    # 평점 데이터를 DataFrame으로 생성
    recomm_themes = pd.DataFrame(data=recomm_themes.values, index=recomm_themes.index, columns=['pred_score'])

    recommend1 = []
    for theme in list(recomm_themes.index.values):
      thema = Thema.objects.get(thema_name=theme)
      db = dict()
      db['thema_index'] = thema.thema_index
      db['cafe_name'] = thema.cafe_name
      db['thema_name'] = thema.thema_name
      db['thema_limit_time'] = thema.thema_limit_time
      db['thema_genre'] = thema.thema_genre
      db['thema_level'] = thema.thema_level
      db['thema_activity'] = thema.thema_activity
      db['thema_number_of_people'] = thema.thema_number_of_people
      db['thema_info'] = thema.thema_info
      db['thema_picture'] = thema.thema_picture
      recommend1.append(db)

    print(recommend1)


    return JsonResponse(recommend1, safe=False)


@csrf_exempt
def recommendThema2(request):
    data_list = json.loads(request.body)

    review_list = Review.objects.all()
    matrix = pd.DataFrame.from_records([c.to_dict() for c in review_list])


    for i in range(len(data_list)):
        review_data = data_list[i]
        data = {'thema_name': review_data['thema_name'], 'user_difficulty': review_data['user_difficulty'],
                'user_nickname': 'ESCAPABLE'}
        matrix = matrix.append(data, ignore_index=True)

    review_matrix = matrix.pivot_table('user_difficulty', index='user_nickname', columns='thema_name',
                                       aggfunc='first')  # 사용자-테마 행렬

    P, Q = matrix_factorization_difficulty(review_matrix.values, K=50, steps=10, learning_rate=0.01, r_lambda=0.01)
    pred_matrix_difficulty = np.dot(P, Q.T)

    difficulty_pred_matrix = pd.DataFrame(data=pred_matrix_difficulty, index=review_matrix.index, columns=review_matrix.columns)
    print(difficulty_pred_matrix)
    review_list = Review.objects.all()
    matrix = pd.DataFrame.from_records([c.to_dict() for c in review_list])

    for i in range(len(data_list)):
        review_data = data_list[i]
        print(review_data)
        data = {'thema_name': review_data['thema_name'], 'user_escape': int(review_data['user_escape']),
                'user_nickname': 'ESCAPABLE'}
        matrix = matrix.append(data, ignore_index=True)

    review_matrix = matrix.pivot_table('user_escape', index='user_nickname', columns='thema_name',
                                       aggfunc='first')  # 사용자-테마 행렬

    P, Q = matrix_factorization_Escape(review_matrix.values, K=50, steps=10, learning_rate=0.01, r_lambda=0.01)
    pred_matrix_escape = np.dot(P, Q.T)



    Escapable_pred_matrix = pd.DataFrame(data=pred_matrix_escape, index=review_matrix.index, columns=review_matrix.columns)
    # 머신러닝 처리 -> 추천 테마 데이터 받아오기

    review_list = Review.objects.all()
    matrix = pd.DataFrame.from_records([c.to_dict() for c in review_list])

    for i in range(len(data_list)):
        review_list = data_list[i]
        data = {'thema_name': review_list['thema_name'], 'user_rate': float(review_list['user_rate']),
                'user_nickname': 'ESCAPABLE'}
        matrix = matrix.append(data, ignore_index=True)

    review_matrix = matrix.pivot_table('user_rate', index='user_nickname', columns='thema_name',
                                       aggfunc='first')  # 사용자-테마 행렬

    P, Q = matrix_factorization_rate(review_matrix.values, K=50, steps=10, learning_rate=0.01, r_lambda=0.01)
    pred_matrix = np.dot(P, Q.T)

    ratings_pred_matrix = pd.DataFrame(data=pred_matrix, index=review_matrix.index, columns=review_matrix.columns)

    # 사용자가 관람하지 않은 영화제목 추출
    unplayed_list = get_unplayed_themes_rate(review_matrix, 'ESCAPABLE')

    # 잠재 요인 협업 필터링으로 영화 추천
    recomm_themes = recomm_thema_by_userid_rate(ratings_pred_matrix, 'ESCAPABLE', unplayed_list, top_n=6)

    # 평점 데이터를 DataFrame으로 생성
    recomm_themes = pd.DataFrame(data=recomm_themes.values, index=recomm_themes.index, columns=['pred_score'])

    datas = recomm_themes.index.values.tolist()
    print(type(recomm_themes.index.values))
    print(recomm_themes.index.values.tolist())
    predict_escape = Predict_Escape(Escapable_pred_matrix, 'ESCAPABLE',datas) # escapable
    predict_difficulty = Predict_Difficulty(difficulty_pred_matrix, 'ESCAPABLE',datas) # difficulty
    result_list = [[3,'escape','difficulty'] for i in range(6)]
    for i in range(6):
        result_list[i][0] = list(recomm_themes.index.values)[i]
        result_list[i][1] = predict_escape[i]
        result_list[i][2] = predict_difficulty[i]

    cnt = 0
    recommend2 = []
    for theme in list(recomm_themes.index.values):
        thema = Thema.objects.get(thema_name=theme)
        db = dict()
        db['thema_index'] = thema.thema_index
        db['cafe_name'] = thema.cafe_name
        db['thema_name'] = thema.thema_name
        db['thema_limit_time'] = thema.thema_limit_time
        db['thema_genre'] = thema.thema_genre
        db['thema_level'] = thema.thema_level
        db['thema_activity'] = thema.thema_activity
        db['thema_number_of_people'] = thema.thema_number_of_people
        db['thema_info'] = thema.thema_info
        db['thema_picture'] = thema.thema_picture
        db['pred_escape'] = predict_escape[cnt]
        db['pred_difficulty'] = predict_difficulty[cnt]
        cnt+=1
        recommend2.append(db)
        print(db)
    
    print(recommend2)

    return JsonResponse(recommend2, safe=False)