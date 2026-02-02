# egghouse.database 사용 가이드

PostgreSQL 데이터베이스 관리 유틸리티.

---

## 개요

연구 목적의 간단한 PostgreSQL 관리 도구입니다:
- 데이터베이스/스키마/테이블 관리
- CRUD 연산 (Insert, Select, Update, Delete)
- Upsert (Insert on Conflict)
- 날짜 범위 쿼리
- pandas DataFrame 변환

---

## 설치

```bash
pip install psycopg2-binary
```

---

## 설정 방법

### 환경 변수

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=solar_data       # 또는 DB_DATABASE
export DB_USER=username
export DB_PASSWORD=password
export DB_LOG_QUERIES=true      # 쿼리 로깅 (선택)
```

```python
from egghouse.database import PostgresManager, load_config

config = load_config()  # 환경 변수에서 자동 로드
db = PostgresManager(**config['database'])
```

### YAML 파일

```yaml
# database.yaml
database:
  host: localhost
  port: 5432
  database: solar_data
  user: username
  password: password
  log_queries: true
```

```python
config = load_config('database.yaml')
db = PostgresManager(**config['database'])
```

### JSON 파일

```json
{
  "database": {
    "host": "localhost",
    "port": 5432,
    "database": "solar_data",
    "user": "username",
    "password": "password"
  }
}
```

```python
config = load_config('database.json')
db = PostgresManager(**config['database'])
```

### 직접 지정

```python
from egghouse.database import PostgresManager, from_dict

# 딕셔너리로 생성
config = from_dict({
    'host': 'localhost',
    'port': 5432,
    'database': 'solar_data',
    'user': 'username',
    'password': 'password'
})
db = PostgresManager(**config)

# 또는 직접 인자로 전달
db = PostgresManager(
    host='localhost',
    port=5432,
    database='solar_data',
    user='username',
    password='password',
    log_queries=True
)
```

### 예제 설정 파일 생성

```python
from egghouse.database import create_example_config

create_example_config('config.example.yaml')
```

---

## PostgresManager 사용법

### 연결 관리

```python
from egghouse.database import PostgresManager

# 기본 연결
db = PostgresManager(
    host='localhost',
    database='solar_data',
    user='user',
    password='pass'
)

# 작업 완료 후 닫기
db.close()

# Context manager 사용 (권장)
with PostgresManager(**config) as db:
    db.insert('users', {'name': 'test'})
    # 자동으로 close() 호출
```

---

## 데이터 조작 (CRUD)

### Insert

```python
# 단일 행 삽입
db.insert('users', {'name': 'Eunsu', 'email': 'eunsu@kasi.re.kr'})

# 다중 행 삽입
db.insert('users', [
    {'name': 'User1', 'email': 'user1@example.com'},
    {'name': 'User2', 'email': 'user2@example.com'}
])

# ID 반환 (SERIAL 컬럼 있을 때)
new_id = db.insert('users', {'name': 'New'}, return_id=True)
```

### Select

```python
# 모든 행 조회
users = db.select('users')

# 특정 컬럼만 조회
users = db.select('users', columns=['id', 'name'])

# WHERE 조건
users = db.select('users', where={'name': 'Eunsu'})

# 정렬 및 제한
users = db.select('users', order_by='created_at DESC', limit=10)

# 조합
users = db.select('users',
    columns=['id', 'name', 'email'],
    where={'active': True},
    order_by='name ASC',
    limit=100
)
```

### Update

```python
# WHERE 절 필수!
affected_rows = db.update('users',
    data={'email': 'new@example.com'},
    where={'name': 'Eunsu'}
)
print(f"Updated {affected_rows} rows")
```

### Delete

```python
# WHERE 절 필수!
deleted_rows = db.delete('users', where={'name': 'Eunsu'})
print(f"Deleted {deleted_rows} rows")
```

---

## Upsert (Insert or Update)

충돌 시 업데이트하는 INSERT ON CONFLICT DO UPDATE.

```python
# 단일 행 upsert
db.upsert('observations',
    data={'filepath': '/data/aia.fits', 'wavelength': 171, 'processed': True},
    conflict_columns='filepath',
    update_columns=['processed']
)

# 다중 행 upsert
db.upsert('observations', [
    {'filepath': f1, 'wavelength': 171, 'status': 'done'},
    {'filepath': f2, 'wavelength': 193, 'status': 'done'}
], conflict_columns='filepath')

# 복합 키 충돌
db.upsert('data',
    data={'date': '2024-01-01', 'wavelength': 171, 'count': 10},
    conflict_columns=['date', 'wavelength']
)

# update_columns 미지정 시 conflict_columns 외 모든 컬럼 업데이트
db.upsert('users',
    data={'username': 'eunsu', 'email': 'new@email.com', 'status': 'active'},
    conflict_columns='username'
)  # email, status가 업데이트됨
```

---

## 날짜 범위 쿼리

```python
from datetime import datetime

start = datetime(2024, 1, 1)
end = datetime(2024, 12, 31)

# 기본 (start <= date < end)
results = db.select_date_range('observations',
    date_column='date',
    start_date=start,
    end_date=end
)

# 끝 날짜 포함 (start <= date <= end)
results = db.select_date_range('observations',
    date_column='date',
    start_date=start,
    end_date=end,
    inclusive_end=True
)

# 추가 조건과 함께
results = db.select_date_range('observations',
    date_column='timestamp',
    start_date=start,
    end_date=end,
    columns=['id', 'filepath', 'wavelength'],
    where={'wavelength': 171},
    order_by='timestamp DESC',
    limit=1000
)
```

---

## 테이블 관리

### 테이블 생성

```python
db.create_table('observations', {
    'id': 'SERIAL PRIMARY KEY',
    'filepath': 'VARCHAR(500) UNIQUE NOT NULL',
    'wavelength': 'INTEGER',
    'date': 'TIMESTAMP',
    'processed': 'BOOLEAN DEFAULT FALSE',
    'created_at': 'TIMESTAMP DEFAULT NOW()'
})
```

### 테이블 목록

```python
# 전체 정보 (이름, 크기)
tables = db.list_tables()
# [{'name': 'users', 'size': '8192 bytes'}, ...]

# 이름만
table_names = db.list_tables(names_only=True)
# ['users', 'observations', ...]
```

### 테이블 구조 확인

```python
columns = db.describe_table('observations')
for col in columns:
    print(f"{col['name']}: {col['type']}")
```

### 테이블 존재 확인

```python
if db.table_exists('observations'):
    print("Table exists")
```

### 테이블 삭제

```python
db.drop_table('old_table')
db.drop_table('parent_table', cascade=True)  # 의존 객체도 삭제
```

---

## 유틸리티

### 행 수 세기

```python
total = db.count('observations')
filtered = db.count('observations', where={'wavelength': 171})
```

### 테이블 비우기

```python
db.truncate('temp_data')
db.truncate('parent_table', cascade=True)
```

### VACUUM

```python
db.vacuum()                        # 전체 데이터베이스
db.vacuum('observations')          # 특정 테이블
db.vacuum('observations', full=True)  # VACUUM FULL
```

---

## DataFrame 변환

```python
from egghouse.database import to_dataframe

results = db.select('observations')
df = to_dataframe(results)

# 날짜 컬럼 파싱
df = to_dataframe(results, parse_dates=['date', 'created_at'])

# 직접 사용
import pandas as pd
df = pd.DataFrame(results)
```

---

## Raw SQL 실행

```python
# 결과 없는 쿼리
db.execute("CREATE INDEX idx_wavelength ON observations(wavelength)")

# 결과 있는 쿼리
results = db.execute(
    "SELECT * FROM observations WHERE wavelength = %s",
    params=(171,),
    fetch=True
)

# 파라미터화 쿼리 (SQL injection 방지)
results = db.execute(
    "SELECT * FROM users WHERE name = %s AND status = %s",
    params=('Eunsu', 'active'),
    fetch=True
)
```

---

## 스키마 관리

```python
# 스키마 생성
db.create_schema('solar')

# 스키마 내 테이블 생성
db.create_table('observations', {...}, schema='solar')

# 스키마 내 테이블 조작
db.insert('observations', data, schema='solar')
db.select('observations', schema='solar')

# 스키마 목록
schemas = db.list_schemas()

# 스키마 삭제
db.drop_schema('solar', cascade=True)
```

---

## 데이터베이스 관리

```python
# 데이터베이스 없이 연결 (관리용)
db = PostgresManager(host='localhost', user='admin', password='pass')

# 데이터베이스 생성
db.create_database('new_db')

# 데이터베이스 목록
databases = db.list_databases()

# 데이터베이스 삭제
db.drop_database('old_db', force=True)  # 연결 강제 종료 후 삭제
```

---

## SQL Injection 방지

PostgresManager는 `psycopg2.sql` 모듈을 사용하여 SQL Injection을 방지합니다:

```python
# 안전: 파라미터화 쿼리 사용
db.select('users', where={'name': user_input})

# 안전: sql.Identifier로 테이블/컬럼명 처리
db.insert(table_name, data)  # 내부적으로 sql.Identifier 사용

# 위험: 직접 문자열 조합 (하지 마세요!)
# db.execute(f"SELECT * FROM {table_name}")  # NEVER DO THIS
```

---

## 전체 예제

```python
from datetime import datetime, timedelta
from egghouse.database import PostgresManager, load_config, to_dataframe

# 설정 로드
config = load_config('database.yaml')

with PostgresManager(**config['database']) as db:
    # 테이블 생성
    if not db.table_exists('observations'):
        db.create_table('observations', {
            'id': 'SERIAL PRIMARY KEY',
            'filepath': 'VARCHAR(500) UNIQUE NOT NULL',
            'wavelength': 'INTEGER',
            'date': 'TIMESTAMP',
            'processed': 'BOOLEAN DEFAULT FALSE'
        })

    # 데이터 삽입
    db.insert('observations', {
        'filepath': '/data/aia_171_20240101.fits',
        'wavelength': 171,
        'date': datetime(2024, 1, 1, 12, 0, 0)
    })

    # Upsert (중복 시 업데이트)
    db.upsert('observations', {
        'filepath': '/data/aia_171_20240101.fits',
        'wavelength': 171,
        'date': datetime(2024, 1, 1, 12, 0, 0),
        'processed': True
    }, conflict_columns='filepath')

    # 쿼리
    today = datetime.now()
    week_ago = today - timedelta(days=7)

    results = db.select_date_range('observations',
        date_column='date',
        start_date=week_ago,
        end_date=today,
        where={'wavelength': 171}
    )

    # DataFrame 변환
    df = to_dataframe(results, parse_dates=['date'])
    print(f"Found {len(df)} observations")
```

---

## 의존성

| 패키지 | 용도 |
|--------|------|
| psycopg2-binary | PostgreSQL 연결 |
| pyyaml | YAML 설정 파일 |
| pandas (선택) | DataFrame 변환 |

설치:
```bash
pip install psycopg2-binary pyyaml pandas
```

---

## 주의사항

1. **Autocommit**: PostgresManager는 autocommit 모드로 동작합니다. 트랜잭션이 필요하면 직접 관리하세요.
2. **WHERE 필수**: `update()`와 `delete()`는 WHERE 절 없이 실행할 수 없습니다 (안전 장치).
3. **연결 관리**: context manager(`with`)를 사용하거나 작업 후 `close()`를 호출하세요.
4. **비밀번호**: 코드에 비밀번호를 하드코딩하지 마세요. 환경 변수나 설정 파일을 사용하세요.
