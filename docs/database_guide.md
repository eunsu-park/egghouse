# egghouse.database 사용 가이드

PostgreSQL 데이터베이스 관리 유틸리티.

> **v0.7+ / v0.8+**: 선언형 스키마 생성과 대량 레코드 헬퍼는 아래
> [선언형 스키마](#선언형-스키마-v07) 및
> [대량 레코드 헬퍼](#대량-레코드-헬퍼-v07) 절을 참조하세요.
> 솔라/우주기상 DB 도메인 레이어(`egghouse.swdb`, v0.8)는
> [egghouse.swdb 연계](#egghouseswdb-연계-v08) 절을 참조하세요.

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

## 선언형 스키마 (v0.7+)

`PostgresManager.create_table()`이 테이블을 하나씩 만드는 반면,
`egghouse.database.schema` 모듈은 **설정 딕셔너리 하나로 전체 스키마를
선언적으로 생성**합니다. setup-sw-db의 `core/database.py`에서 일반화하여
가져온 인프라로, 프로젝트별 도메인 코드 없이 config dict + import만으로
스키마를 구축할 수 있습니다.

### schema_config 형식

`schema_config`는 `테이블명 -> 테이블 스펙` 의 평범한 dict입니다.
테이블 스펙은 `컬럼명 -> SQL 타입` 매핑에 더해 다음 예약 메타 키를
선택적으로 가집니다:

| 메타 키 | 의미 |
|---------|------|
| `_primary_key` | 복합 PRIMARY KEY 를 구성할 컬럼 리스트 |
| `_unique` | UNIQUE 제약. 컬럼명(str) 또는 컬럼 리스트(다중 컬럼). 여러 개면 리스트의 리스트 |
| `_indexes` | CREATE INDEX 대상. 각 항목은 컬럼명 또는 컬럼 리스트 |

```python
schema = {
    "sdo": {
        "telescope":    "VARCHAR(10) NOT NULL",
        "channel":      "VARCHAR(20) NOT NULL",
        "datetime":     "TIMESTAMP NOT NULL",
        "file_path":    "VARCHAR(512) NOT NULL",
        "_primary_key": ["telescope", "channel", "datetime"],
        "_unique":      ["file_path"],
        "_indexes":     [["datetime"], ["telescope"]],
    },
}
```

이 형식은 **instrument-blind**(계측기 비의존)입니다. 모듈은 테이블이
"무엇을 의미하는지" 전혀 들여다보지 않고 선언된 컬럼과 제약만 다룹니다.
태양 이미지든 우주기상 시계열이든 임의의 선언형 config 를
`initialize_database`에 넘기면 정확히 그 테이블들이 생성됩니다.

> **식별자 안전성**: 테이블/컬럼/인덱스 식별자는
> `^[A-Za-z_][A-Za-z0-9_]*$` 패턴으로 검증되며, 위반 시 `ValueError`가
> 발생합니다. schema config 는 개발자가 작성하지만, 오타로 주입 가능한
> DDL 이 생성되면 조용히 위험해지므로 경계 검사를 둡니다. 컬럼 *타입*은
> 자유 형식 텍스트입니다.

### 순수 빌더 (DB 연결 불필요)

다음 함수들은 순수 함수로, 살아있는 PostgreSQL 없이 단위 테스트가
가능합니다.

#### build_create_table_sql

```python
from egghouse.database import build_create_table_sql

sql = build_create_table_sql("sdo", schema["sdo"])
# CREATE TABLE sdo (telescope VARCHAR(10) NOT NULL, ...,
#   PRIMARY KEY (telescope, channel, datetime), UNIQUE (file_path))
```

선언형 테이블 스펙을 단일 `CREATE TABLE` 문으로 변환합니다.
`_primary_key`가 주어지면 개별 컬럼 타입의 인라인 `PRIMARY KEY`는
제거되고 복합 `PRIMARY KEY (...)` 제약이 끝에 추가됩니다. `_unique`
항목은 컬럼명 또는 컬럼명 리스트(다중 컬럼 UNIQUE)일 수 있습니다.

##### 파라미터
- `table_name` (str): 테이블 이름 (식별자 검증됨).
- `table_spec` (dict): 컬럼 매핑 + 메타 키. 컬럼이 없으면 `ValueError`.

#### build_index_sql

```python
from egghouse.database import build_index_sql

stmts = build_index_sql("sdo", [["datetime"], ["telescope"]])
# ['CREATE INDEX idx_sdo_datetime ON sdo (datetime)',
#  'CREATE INDEX idx_sdo_telescope ON sdo (telescope)']
```

`_indexes`를 `CREATE INDEX` 문 리스트로 변환합니다. 각 항목은 컬럼명
또는 컬럼명 리스트입니다. 인덱스명은 `idx_<table>_<cols>` 규칙으로
생성됩니다.

##### 파라미터
- `table_name` (str): 테이블 이름.
- `indexes` (list | None): `_indexes` 값. 비어 있으면 빈 리스트 반환.

#### split_schema_meta

```python
from egghouse.database import split_schema_meta

columns, primary_key, unique, indexes = split_schema_meta(schema["sdo"])
# columns = {'telescope': 'VARCHAR(10) NOT NULL', ...}
# primary_key = ['telescope', 'channel', 'datetime']
```

테이블 스펙을 `(columns, primary_key, unique, indexes)` 튜플로 분리합니다.
호출자의 dict 는 변경하지 않습니다(non-mutating). 메타 키가 없으면 해당
값은 `None`입니다.

##### 파라미터
- `table_spec` (dict): 테이블 스펙.

### 연결 래퍼 (DB 연결 사용)

다음 함수들은 내부적으로 `PostgresManager`를 열어 실제 DB 에 적용하는
통합 레벨 함수입니다.

#### create_database

```python
from egghouse.database import create_database

ok = create_database(db_config, verbose=True)
```

대상 데이터베이스가 없으면 생성합니다(멱등). 먼저 대상 DB 로 접속을
시도하고, 성공하면 이미 존재하는 것입니다. 실패하면 관리용 DB
(`template1` → `postgres`)로 접속해 `CREATE DATABASE`를 실행합니다.
DB 가 존재하거나 생성되면 `True`, 실패하면 `False`를 반환합니다.

##### 파라미터
- `db_config` (dict): `PostgresManager` 인자(host, port, database, user, password, ...).
- `verbose` (bool, 키워드 전용): 진행 상황 출력. 기본 `False`.

#### create_tables_from_schema

```python
from egghouse.database import create_tables_from_schema

actions = create_tables_from_schema(db_config, schema, drop=False, verbose=True)
# {'sdo': 'created'}  # 또는 'recreated' / 'skipped'
```

`schema_config`에 기술된 모든 테이블을 생성합니다. 기존 테이블은
`drop=True`가 아니면 건너뜁니다(`skipped`). `drop=True`면 기존 테이블을
`CASCADE`로 삭제 후 재생성합니다(`recreated`).

##### 파라미터
- `db_config` (dict): `PostgresManager` 인자.
- `schema_config` (dict): `{테이블명: 테이블 스펙}`.
- `drop` (bool, 키워드 전용): 기존 테이블 삭제 후 재구축. 기본 `False`.
- `verbose` (bool, 키워드 전용): 테이블별 동작 출력. 기본 `False`.

반환값: `{테이블명: "created" | "recreated" | "skipped"}`.

#### initialize_database

```python
from egghouse.database import initialize_database

schema = {
    "sdo": {
        "telescope":    "VARCHAR(10) NOT NULL",
        "channel":      "VARCHAR(20) NOT NULL",
        "datetime":     "TIMESTAMP NOT NULL",
        "file_path":    "VARCHAR(512) NOT NULL",
        "_primary_key": ["telescope", "channel", "datetime"],
        "_unique":      ["file_path"],
        "_indexes":     [["datetime"], ["telescope"]],
    },
}

db_config = {
    "host": "localhost",
    "port": 5432,
    "database": "solar_data",
    "user": "username",
    "password": "password",
}

actions = initialize_database(db_config, schema, verbose=True)
```

`create_database`와 `create_tables_from_schema`를 결합한 편의 래퍼입니다.
DB 를 (멱등하게) 생성한 뒤 모든 테이블을 만들고 동작 dict 를 반환합니다.

##### 파라미터
- `db_config` (dict): `PostgresManager` 인자.
- `schema_config` (dict): `{테이블명: 테이블 스펙}`.
- `verbose` (bool, 키워드 전용): 진행 상황 출력. 기본 `False`.

---

## 대량 레코드 헬퍼 (v0.7+)

`egghouse.database.records` 모듈은 pandas DataFrame 의 대량 upsert 와
고아 레코드 정리를 제공합니다. SQL 문자열 빌더와 레코드 정규화는 순수
함수(PostgreSQL 없이 단위 테스트 가능)이며, 연결을 여는 함수는 통합
레벨입니다.

#### normalize_records

```python
from egghouse.database import normalize_records

records = normalize_records(df)
# [{'telescope': 'sdo', 'datetime': ..., ...}, ...]
```

DataFrame 을 dict 행 리스트로 변환합니다. 컬럼명은 소문자화되고,
`NaN`은 `None`으로 치환됩니다(psycopg2 가 SQL `NULL`을 기록하도록).
빈 DataFrame 은 빈 리스트를 반환합니다(원본 setup-sw-db 코드가 충돌하던
edge case 를 처리).

##### 파라미터
- `df`: pandas DataFrame.

#### build_upsert_sql

```python
from egghouse.database import build_upsert_sql

sql = build_upsert_sql("sdo", ["telescope", "channel", "datetime", "file_path"],
                       ["telescope", "channel", "datetime"])
# INSERT INTO sdo (telescope, channel, datetime, file_path)
#   VALUES (%s, %s, %s, %s)
#   ON CONFLICT (telescope, channel, datetime) DO NOTHING
```

`INSERT ... ON CONFLICT (...) DO NOTHING` 문을 생성하는 순수 함수입니다.
복합 conflict 타깃을 지원하며, execute 당 단일 위치 파라미터 행(`%s`
플레이스홀더)을 기대합니다. 식별자는 검증됩니다.

##### 파라미터
- `table` (str): 테이블 이름.
- `columns` (list[str]): 삽입할 컬럼 목록.
- `conflict_columns` (str | list[str]): 충돌 판정 컬럼. 문자열이면 단일 컬럼.

#### upsert_dataframe

```python
from egghouse.database import upsert_dataframe

inserted = upsert_dataframe(
    df, "sdo", db_config,
    conflict_columns=["telescope", "channel", "datetime"],
    batch=1000,
)
print(f"{inserted} rows inserted")

# 멱등성: 같은 데이터로 다시 실행하면 아무것도 삽입되지 않음
again = upsert_dataframe(df, "sdo", db_config,
                         conflict_columns=["telescope", "channel", "datetime"])
assert again == 0
```

행을 삽입하되 `conflict_columns` 충돌은 조용히 건너뜁니다. **멱등적**이라
이미 존재하는 행으로 재실행해도 아무것도 삽입되지 않습니다. 복합 PK 와
별개인 다른 UNIQUE 제약(예: `UNIQUE(file_path)`) 위반도 에러가 아니라
스킵으로 처리됩니다. 실제로 삽입된 행 수(스킵 제외)를 반환합니다.

##### 파라미터
- `df`: pandas DataFrame.
- `table` (str): 대상 테이블.
- `db_config` (dict): `PostgresManager` 인자.
- `conflict_columns` (str | list[str], 키워드 전용): 충돌 판정 컬럼. 기본 `"datetime"`.
- `batch` (int, 키워드 전용): 배치 크기. 기본 `1000`.

#### find_orphans

```python
from egghouse.database import find_orphans

missing = find_orphans(["/data/a.fits", "/data/b.fits"])
# 디스크에 더 이상 존재하지 않는 경로만 반환
```

주어진 경로 중 디스크에 더 이상 존재하지 않는 경로의 부분집합을
반환합니다.

##### 파라미터
- `file_paths` (list[str]): 검사할 경로 목록.

#### delete_orphans

```python
from egghouse.database import delete_orphans

deleted = delete_orphans("sdo", db_config, file_column="file_path")
print(f"{deleted} orphan rows deleted")
```

참조하는 파일이 디스크에서 사라진 행을 삭제합니다. 삭제된 행 수를
반환합니다.

##### 파라미터
- `table` (str): 대상 테이블.
- `db_config` (dict): `PostgresManager` 인자.
- `file_column` (str, 키워드 전용): 파일 경로 컬럼명. 기본 `"file_path"`.

---

## egghouse.swdb 연계 (v0.8+)

v0.8 의 `egghouse.swdb`는 위 일반(instrument-blind) 헬퍼 위에 구축된
솔라/우주기상 DB 도메인 레이어입니다. 본 모듈의 선언형 스키마와 대량
upsert 를 그대로 활용하면서, 참조 스키마(`SDO_SCHEMA`, `LASCO_SCHEMA`,
`SECCHI_SCHEMA`), FITS 핸들러(`FitsHandler` ABC, AIA 구현), 디렉터리
스캔/등록 같은 도메인 기능을 더합니다. swdb 의 내부 동작과 사용 예시는
`docs/swdb_guide.md` 및 루트 `README.MD`의 Modules 절을 참조하세요.
함수 시그니처는 `API_REFERENCE.md`, 변경 이력은 `CHANGELOG.md`에
있습니다.

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
