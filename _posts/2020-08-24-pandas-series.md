---
title: "Series 자료구조(feat.인덱스)"
search: true
categories:
 - Pandas
tags:
 - Pandas
 - 판다스
 - Series
last_modified_at: 2020-08-24 23:11
layout: jupyter
excerpt: "[Pandas] Series 자료구조 알아보기"
classes: wide
toc: true
toc_sticky: true
toc_label: "목차"
---

---
## 0. Pandas에 대해

***Pandas***는 데이터 분석을 위해 많은 이들에게 사용되는 파이썬 라이브러리 패키지입니다. 데이터프레임을 원하는 대로 가공하는 것은 데이터 과학자들에게 필수 요소이기 때문에, ***Pandas***는 빅데이터, 머신러닝의 기본 소양이 되는 라이브러리라고 할 수 있습니다. <br>이후에 이 블로그에서 다룰 머신러닝이나 인공지능 관련 글에서도 ***Pandas***는 길게 설명하지 않을 것이기에, 틈틈이 실습을 하면서 판다스를 체득하시길 추천합니다. <br>
이 글에서는 Series에 관련된 생성과 조작에 대해서 다루겠습니다.

---

## 1. Pandas Series 생성하기

*Series*는 가장 간단한 1차원적인 자료구조라고 생각할 수 있습니다. 다른 프로그래밍 언어를 다뤄본 분들이라면 바로 이해할 만한 배열, 리스트, 딕셔너리와 비슷합니다. 실제로 배열, 리스트 등과 같은 시퀀스 데이터를 받아서 바로 *Series* 객체로 변환할 수 있습니다. <br><br>
예시를 보여드리겠습니다. 

<div class="prompt input_prompt">
In&nbsp;[7]:
</div>

<div class="input_area" markdown="1">

```python
import pandas as pd
import numpy as np

data = [2,3,5,7,9]
s = pd.Series(data)
s
```

</div>




{:.output_data_text}

```
0    2
1    3
2    5
3    7
4    9
dtype: int64
```



자료형은 굳이 숫자가 아니어도 됩니다. 문자열로 이루어져도 상관 없으며, 문자열과 숫자가 섞여도 자연스럽게 변환이 가능합니다. 높은 호환성과 범용성이 ***Pandas***의 큰 장점이라고 할 수 있죠.

<div class="prompt input_prompt">
In&nbsp;[8]:
</div>

<div class="input_area" markdown="1">

```python
# 이렇게도 가능합니다. 물론 이런 데이터를 다루고 싶은 사람은 없겠지만요.
data = ['apple', 'banana', 100, 200, '300'] 
s = pd.Series(data)
s
```

</div>




{:.output_data_text}

```
0     apple
1    banana
2       100
3       200
4       300
dtype: object
```



그런데 `pd.Series()`에 넘겨준 리스트 외에도, 출력을 보시면 `0, 1, 2, 3, 4`가 순서대로 나열되어 있습니다. 왼쪽 `[0, 1, 2, 3, 4]`는 Series 객체의 ***index***이고, `['apple', 'banana', 100, 200, '300']`은 Series 객체의 ***values***입니다. 
<br><br>
*index*와 *values*는 python의 *dictionary*자료형을 아시는 분이라면 이해하기 편하실 겁니다. 보통 데이터를 조회할 때 쓰이는 key값이 *index*이고, 그에 해당하는 값이 *values* 입니다. 실제로 *dictionary* 를 이용해서 Series를 생성할 수도 있습니다.<br><br>
예시를 보시죠.

<div class="prompt input_prompt">
In&nbsp;[9]:
</div>

<div class="input_area" markdown="1">

```python
data = {"국어" : 100, "수학" : 92, "영어" : 96}
s = pd.Series(data)
s
```

</div>




{:.output_data_text}

```
국어    100
수학     92
영어     96
dtype: int64
```



이렇게 딕셔너리를 이용해서 *Series*를 생성하면 인덱스에도 각 *value*에 해당되는 key값이 적용되는 것을 볼 수 있습니다.

