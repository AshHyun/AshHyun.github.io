---
title: "[ML] 데이터셋 분할하기(훈련 세트, 테스트 세트)(1)"
search: true
categories:
 - 머신러닝
tags:
 - 머신러닝
 - 테스트 세트
last_modified_at: 2020-08-31 23:17
layout: jupyter
classes: wide
excerpt: 데이터를 훈련 세트와 테스트 세트로 분할하기
toc: true
toc_sticky: true
toc_label: "목차"

---
## 소개

**머신러닝**을 통해 모델이 예측이나 분류를 할 수 있도록 학습시킬 때, 올바르게 훈련 세트와 테스트 세트를 나누는 것은 생각보다 정말 중요한 작업입니다. 만약 모델이 훈련 세트의 잘못된 경향을 학습하게 될 경우 모델 전체의 정확도와 정밀도 등이 떨어지게 됩니다. 나아가 테스트 세트가 잘못되었을 경우에도, 모델을 검증하고 평가할 때 오류가 생기게 됩니다.<br>
<br>
우선 머신러닝 모델을 훈련시키고 검증하기 위해 데이터를 분할할 때는 보통 <br>

- **훈련 세트(Training Set)**
- **평가 세트(Test Set)**

이런 식으로 분할합니다. **검증 세트(Validation Set)** 를 따로 나눌 수도 있지만, 사실 검증 세트와 평가 세트는 크게 다르지 않아 생략하겠습니다.<br><br>

이 포스팅은 **[Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow](https://book.naver.com/bookdb/book_detail.nhn?bid=16328592)**를 참고하여 작성하였습니다.

---

## 훈련 세트와 테스트 세트


1. **훈련 세트**는 말 그대로 머신러닝 모델이 데이터의 경향성을 학습하기 위해 사용하는 데이터 세트입니다. 모델이 훈련 데이터셋 내에서의 최적점을 찾는 알고리즘으로 움직이기 때문에, 훈련 데이터에서의 이상치, 스케일, 측정 오차 등은 모델의 완성도에 민감한 영향을 끼칩니다. <br> 또한, 전체 데이터 자체에는 문제가 없더라도, 만약 대한민국 전체 인구의 키에 관련된 모델을 만들기 위해 데이터를 가져왔는데 남성의 키를 훈련 세트로, 여성의 키를 테스트 세트로 쓰게 된다면 모델도 개판, 평가도 개판일 것입니다. 이는 극단적인 예시지만, 이런 중요한 feature의 분배에 오류가 생긴다면 모델의 성능도 그에 따라 과대적합 or 과소적합 되게 됩니다.<br>

2. **테스트 세트** 또한 말 그대로 완성된 모델을 평가하기 위해 사용하는 세트입니다. 지도 학습일 경우 정답이 표시된 레이블(Label)을 따로 분리시켜놓고, 모델이 도출한 정답과 따로 분리해놓은 레이블을 비교하며 평가하게 됩니다. 보통 전체 데이터가 100이라면 80정도를 훈련 세트, 20정도를 테스트 세트로 사용하는 편입니다.

---

## 데이터셋 분할하기 (무작위)


전체 데이터셋을 __훈련 세트__와 __테스트 세트__로 나눌 때, 보통 가장 쉽게 생각할 수 있는 방법이 무작위로 샘플을 나누는 것일 겁니다. 우선 예시가 될 타이타닉 데이터셋을 가져오겠습니다.

<div class="prompt input_prompt">
</div>

<div class="input_area" markdown="1">

```python
import pandas as pd

data = pd.read_csv('kaggle/titanic/train.csv')
```

</div>

이제 무작위로 데이터를 나눠 볼 겁니다. 함수로 예시를 보여드리겠습니다.

<div class="prompt input_prompt">
</div>

<div class="input_area" markdown="1">

```python
import numpy as np

def split_train_test(data, test_ratio):
#   np.random.seed() 
    shuffled_indices = np.random.permutation(len(data)) 
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(data, 0.2)
print(len(train_set), len(test_set))
```

</div>

{:.output_stream}

```
713 178

```

이렇게 하면 대충 8:2로 데이터를 나눈 것을 보실 수 있습니다.<br>하지만 이 방식으로는 무작위로 테스트 세트를 나눌 수 있지만, 단점이 있습니다. 이걸 다시 실행하게 되면 테스트 세트가 바뀌게 됩니다. 그래서 저기 주석 처리해놓은 `np.random.seed()`에 난수 초깃값을 지정하면, 항상 같은 테스트 세트가 생성되게 됩니다.<br>
<br>
하지만 이 해법 또한 **업데이트된 데이터셋**을 사용하려면 문제가 됩니다. 테스트 세트에 있던 데이터가 훈련 세트로 분류되고, 훈련 세트에 있던 데이터가 테스트 세트로 가는 등, 결국 모델은 훈련 세트와 테스트 세트 모두를 가지고 훈련을 하게 될 겁니다. 그렇게 되면 테스트 세트로 평가한 그 지표는 믿을 수가 없게 되죠.<br><br>

## 데이터셋 분할하기 (해싱)

위 문제에 대한 일반적인 해결책은 샘플의 식별자를 사용하여 데이터를 테스트 세트로 분류할지 훈련 세트로 분류할지 판단하는 것입니다. (물론 이건 데이터 세트가 국가명, ID, 주민번호 등과 같은 고유한 식별자를 가지고 있다고 가정합니다) 각 샘플마다 식별자의 해시값을 계산하여 해시 최댓값의 20%보다 작거나 같은 샘플만 테스트 세트로 분류할 수 있습니다. 해시는 무작위성을 띄므로 결과적으로 20% 정도의 데이터만 테스트 세트로 구성되게 되죠. 이 방식으로 테스트 세트와 훈련 세트를 분류할 경우, 데이터가 추가되거나 삭제된 데이터셋을 사용해도 테스트 세트가 동일하게 유지됩니다. 고유한 식별자는 바뀌지 않기 때문에 해시값은 바뀌지 않고, 분류도 일정하게 이루어지는거죠.<br><br>
다음은 함수로 이 개념을 구현한 것입니다.

<div class="prompt input_prompt">
</div>

<div class="input_area" markdown="1">

```python
from zlib import crc32

def test_set_check(id, test_ratio):
    return crc32(np.int64(id)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

# 승객 ID를 식별자로 사용했습니다.
train_set, test_set = split_train_test_by_id(data, 0.2, "PassengerId")

print(len(train_set), len(test_set))
```

</div>

{:.output_stream}

```
714 177

```

사이킷런에도 데이터셋을 나누는 `train_test_split`이라는 함수를 제공하고 있습니다. 이는 위에서 보여드렸던 `split_train_test`와 아주 비슷하지만 두가지 특징이 더 있습니다.

1. `split_train_test`에서 `np.random.seed()`에 난수 초깃값을 지정했던 것처럼, `random_state`라는 _parameter_를 가지고 있습니다.
2. 컬럼의 개수가 같은 여러 데이터셋을 함수에 넘겨서 같은 인덱스를 기반으로 나눌 수 있습니다. <br> 보통 데이터베이스를 관리하는 입장에서는 독립성과 유연성 때문에 종속적이거나 중복되는 데이터를 하나의 테이블에 저장하고 있지 않습니다. 그래서 데이터프레임이 레이블(식별자)에 따라 여러 개로 나뉘어 있을 때 매우 유용하게 사용할 수 있는 기능입니다.

<br>
아래는 예시 코드입니다.

<div class="prompt input_prompt">
</div>

<div class="input_area" markdown="1">

```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(data, test_size=0.2, random_state=1006)

print(len(train_set), len(test_set))
```

</div>

{:.output_stream}

```
712 179

```

다음 포스팅에서는 계층적 샘플링에 대해 설명하도록 하겠습니다. 감사합니다!
