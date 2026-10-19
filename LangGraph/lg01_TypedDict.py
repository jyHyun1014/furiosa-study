# Type Hint (타입 힌트)
# Type Hint(타입 힌트)는 변수, 함수의 매개변수, 반환값 등에 데이터 타입을 명시하는 문법
# Python은 동적 타입 언어이기 때문에 변수 선언 시 타입을 지정할 필요가 없지만, Type Hint를 사용하면 코드의 가독성을 높이고 타입 관련 오류를 사전에 발견하는 데 도움이 됨

# Type Hint 사용
name: str = "홍길동"
age: int = 39
height: float = 175.5
is_student: bool = False

# List: 요소의 타입 지정
names: list[str] = ["김만득", "홍길동"]

# Tuple: 각 요소의 타입 지정
person: tuple[str, int] = ("김만득", 39)

# Set: 요소의 타입 지정
unique_numbers: set[int] = {1, 2, 3}

# Dict: Key와 Value 타입 지정
sample_dict: dict[str, int | str] = { # 딕셔너리의 Key는 str, 딕셔너리의 Value는 str 또는 int
    'name': '김만득',
    'age': 39,
    'job': '회사원',
}
print(sample_dict) # {'name': '김만득', 'age': 39, 'job': '회사원'}
#####################################################################
# TypedDict: 딕셔너리의 Key별로 Value 타입을 지정할 수 있는 타입 힌트
# 일반 dict는 모든 Key와 Value에 공통 타입을 지정하지만, TypedDict는 각 Key에 서로 다른 타입을 지정할 수 있음
# TypedDict는 실제 클래스로 객체를 생성하는 것이 아니라, 딕셔너리의 구조와 타입을 정의하는 용도로 사용됨
# 실행 시 타입을 강제하지 않으며, IDE나 정적 타입 검사 도구를 통해 타입 오류를 확인할 수 있음

from typing import TypedDict

# 딕셔너리의 Key와 각 Value의 타입 정의
class Person(TypedDict):
    name: str # name Key의 Value는 str
    age: int # age Key의 Value는 int
    job: str # job Key의 Value는 str

# Person 클래스를 이용하여 딕셔너리 생성
person = Person(name='홍길동', age=40, job='궁수')
print(person) # {'name': '홍길동', 'age': 40, 'job': '궁수'}

# 일반 딕셔너리를 생성하고 Person 타입 힌트 적용
person2: Person = {'name': '김몽이', 'age': 20, 'job': '전사'}
print(person2) # {'name': '김몽이', 'age': 20, 'job': '전사'}