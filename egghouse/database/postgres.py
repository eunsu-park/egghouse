"""
PostgreSQL database management utilities.

Simple and practical PostgreSQL manager for research purposes.
"""

import logging
from typing import Optional, Dict, List, Any, Union
from contextlib import contextmanager

try:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extras import RealDictCursor
except ImportError:
    raise ImportError(
        "psycopg2 is required for PostgresManager. "
        "Install it with: pip install psycopg2-binary"
    )


class PostgresManager:
    """
    Simple PostgreSQL database manager.
    
    Features:
    - Database, schema, and table management
    - CRUD operations with simple interface
    - Query logging
    - Automatic connection handling
    
    Example:
        >>> db = PostgresManager(host='localhost', database='mydb', 
        ...                       user='user', password='pass', log_queries=True)
        >>> db.create_table('users', {'id': 'SERIAL PRIMARY KEY', 'name': 'VARCHAR(100)'})
        >>> db.insert('users', {'name': 'Eunsu'})
        >>> users = db.select('users', where={'name': 'Eunsu'})
        >>> db.close()
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 5432,
        database: str = None,
        user: str = None,
        password: str = None,
        log_queries: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize PostgreSQL manager.
        
        Args:
            host: Database host
            port: Database port
            database: Database name (can be None for database-level operations)
            user: Database user
            password: Database password
            log_queries: Whether to log SQL queries
            logger: Custom logger (if None, creates default logger)
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.log_queries = log_queries
        
        # Setup logger
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO if log_queries else logging.WARNING)
        
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Establish database connection."""
        try:
            conn_params = {
                'host': self.host,
                'port': self.port,
                'user': self.user,
                'password': self.password
            }
            if self.database:
                conn_params['database'] = self.database
            
            self.conn = psycopg2.connect(**conn_params)
            self.conn.autocommit = True
            
            db_info = f"{self.database}@{self.host}" if self.database else f"{self.host}"
            self.logger.info(f"Connected to PostgreSQL: {db_info}")
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.logger.info("Connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    @contextmanager
    def _cursor(self, dict_cursor=True):
        """
        Context manager for database cursor.
        
        Args:
            dict_cursor: If True, return results as dictionaries
        """
        cursor_factory = RealDictCursor if dict_cursor else None
        cursor = self.conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cursor
        finally:
            cursor.close()
    
    def execute(
        self,
        query: str,
        params: tuple = None,
        fetch: bool = False,
        dict_cursor: bool = True
    ) -> Optional[List[Dict]]:
        """
        Execute a SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters (for parameterized queries)
            fetch: Whether to fetch results
            dict_cursor: Return results as dictionaries
            
        Returns:
            Query results if fetch=True, None otherwise
        """
        if self.log_queries:
            log_query = query if not params else f"{query} | params: {params}"
            self.logger.info(f"Executing: {log_query}")
        
        try:
            with self._cursor(dict_cursor=dict_cursor) as cursor:
                cursor.execute(query, params)
                
                if fetch:
                    results = cursor.fetchall()
                    if dict_cursor:
                        results = [dict(row) for row in results]
                    self.logger.info(f"Fetched {len(results)} rows")
                    return results
                else:
                    if cursor.rowcount >= 0:
                        self.logger.info(f"Affected {cursor.rowcount} rows")
                    return None
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            raise
    
    # ==================== Database Operations ====================
    
    def create_database(self, db_name: str):
        """
        Create a new database.
        
        Args:
            db_name: Name of the database to create
        """
        query = sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
        self.execute(query.as_string(self.conn))
        self.logger.info(f"Database '{db_name}' created")
    
    def drop_database(self, db_name: str, force: bool = False):
        """
        Drop a database.
        
        Args:
            db_name: Name of the database to drop
            force: If True, terminate existing connections before dropping
        """
        if force:
            # Terminate existing connections
            terminate_query = f"""
            SELECT pg_terminate_backend(pg_stat_activity.pid)
            FROM pg_stat_activity
            WHERE pg_stat_activity.datname = '{db_name}'
            AND pid <> pg_backend_pid();
            """
            self.execute(terminate_query)
        
        query = sql.SQL("DROP DATABASE {}").format(sql.Identifier(db_name))
        self.execute(query.as_string(self.conn))
        self.logger.info(f"Database '{db_name}' dropped")
    
    def list_databases(self) -> List[Dict]:
        """
        List all databases.
        
        Returns:
            List of database information dictionaries
        """
        query = """
        SELECT datname as name, 
               pg_size_pretty(pg_database_size(datname)) as size
        FROM pg_database
        WHERE datistemplate = false
        ORDER BY datname;
        """
        return self.execute(query, fetch=True)
    
    # ==================== Schema Operations ====================
    
    def create_schema(self, schema_name: str):
        """
        Create a new schema.
        
        Args:
            schema_name: Name of the schema to create
        """
        query = sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
            sql.Identifier(schema_name)
        )
        self.execute(query.as_string(self.conn))
        self.logger.info(f"Schema '{schema_name}' created")
    
    def drop_schema(self, schema_name: str, cascade: bool = False):
        """
        Drop a schema.
        
        Args:
            schema_name: Name of the schema to drop
            cascade: If True, drop all objects in the schema
        """
        cascade_str = "CASCADE" if cascade else ""
        query = sql.SQL("DROP SCHEMA {} {}").format(
            sql.Identifier(schema_name),
            sql.SQL(cascade_str)
        )
        self.execute(query.as_string(self.conn))
        self.logger.info(f"Schema '{schema_name}' dropped")
    
    def list_schemas(self) -> List[Dict]:
        """
        List all schemas.
        
        Returns:
            List of schema names
        """
        query = """
        SELECT schema_name as name
        FROM information_schema.schemata
        WHERE schema_name NOT IN ('pg_catalog', 'information_schema')
        ORDER BY schema_name;
        """
        return self.execute(query, fetch=True)
    
    # ==================== Table Operations ====================
    
    def create_table(
        self,
        table_name: str,
        columns: Dict[str, str],
        schema: str = None
    ):
        """
        Create a new table.
        
        Args:
            table_name: Name of the table
            columns: Dictionary of column_name: column_definition
            schema: Schema name (optional)
            
        Example:
            >>> db.create_table('users', {
            ...     'id': 'SERIAL PRIMARY KEY',
            ...     'name': 'VARCHAR(100) NOT NULL',
            ...     'email': 'VARCHAR(255) UNIQUE',
            ...     'created_at': 'TIMESTAMP DEFAULT NOW()'
            ... })
        """
        full_table_name = f"{schema}.{table_name}" if schema else table_name
        
        column_defs = [f"{col} {definition}" for col, definition in columns.items()]
        columns_str = ", ".join(column_defs)
        
        query = f"CREATE TABLE IF NOT EXISTS {full_table_name} ({columns_str})"
        self.execute(query)
        self.logger.info(f"Table '{full_table_name}' created")
    
    def drop_table(self, table_name: str, schema: str = None, cascade: bool = False):
        """
        Drop a table.
        
        Args:
            table_name: Name of the table
            schema: Schema name (optional)
            cascade: If True, drop dependent objects
        """
        full_table_name = f"{schema}.{table_name}" if schema else table_name
        cascade_str = "CASCADE" if cascade else ""
        query = f"DROP TABLE IF EXISTS {full_table_name} {cascade_str}"
        self.execute(query)
        self.logger.info(f"Table '{full_table_name}' dropped")
    
    def list_tables(self, schema: str = 'public', names_only: bool = False) -> Union[List[Dict], List[str]]:
        """
        List all tables in a schema.
        
        Args:
            schema: Schema name (default: 'public')
            names_only: If True, return only table names as list of strings.
                       If False, return full info as list of dicts (default: False)
            
        Returns:
            If names_only=True: List of table names (strings)
            If names_only=False: List of table information dictionaries
            
        Example:
            >>> # Get full info
            >>> tables = db.list_tables()
            >>> print(tables)  # [{'name': 'users', 'size': '8192 bytes'}, ...]
            
            >>> # Get names only
            >>> names = db.list_tables(names_only=True)
            >>> print(names)  # ['users', 'products', ...]
        """
        query = """
        SELECT 
            table_name as name,
            pg_size_pretty(pg_total_relation_size(quote_ident(table_schema) || '.' || quote_ident(table_name))) as size
        FROM information_schema.tables
        WHERE table_schema = %s
        AND table_type = 'BASE TABLE'
        ORDER BY table_name;
        """
        results = self.execute(query, params=(schema,), fetch=True)
        
        if names_only:
            return [table['name'] for table in results]
        return results
    
    def describe_table(self, table_name: str, schema: str = 'public') -> List[Dict]:
        """
        Get table column information.
        
        Args:
            table_name: Name of the table
            schema: Schema name (default: 'public')
            
        Returns:
            List of column information dictionaries
        """
        query = """
        SELECT 
            column_name as name,
            data_type as type,
            character_maximum_length as max_length,
            is_nullable,
            column_default as default_value
        FROM information_schema.columns
        WHERE table_schema = %s
        AND table_name = %s
        ORDER BY ordinal_position;
        """
        return self.execute(query, params=(schema, table_name), fetch=True)
    
    def table_exists(self, table_name: str, schema: str = 'public') -> bool:
        """
        Check if a table exists.
        
        Args:
            table_name: Name of the table
            schema: Schema name (default: 'public')
            
        Returns:
            True if table exists, False otherwise
        """
        query = """
        SELECT EXISTS (
            SELECT 1 
            FROM information_schema.tables 
            WHERE table_schema = %s 
            AND table_name = %s
        );
        """
        result = self.execute(query, params=(schema, table_name), fetch=True)
        return result[0]['exists'] if result else False
    
    # ==================== Data Operations ====================
    
    def insert(
        self,
        table_name: str,
        data: Union[Dict, List[Dict]],
        schema: str = None,
        return_id: bool = False
    ) -> Optional[Any]:
        """
        Insert data into a table.
        
        Args:
            table_name: Name of the table
            data: Dictionary or list of dictionaries with column:value pairs
            schema: Schema name (optional)
            return_id: If True, return the inserted ID (requires RETURNING clause)
            
        Returns:
            Inserted ID if return_id=True, None otherwise
            
        Example:
            >>> db.insert('users', {'name': 'Eunsu', 'email': 'eunsu@kasi.re.kr'})
            >>> db.insert('users', [
            ...     {'name': 'User1', 'email': 'user1@example.com'},
            ...     {'name': 'User2', 'email': 'user2@example.com'}
            ... ])
        """
        full_table_name = f"{schema}.{table_name}" if schema else table_name
        
        # Handle single dictionary
        if isinstance(data, dict):
            data = [data]
        
        if not data:
            self.logger.warning("No data to insert")
            return None
        
        # Get columns from first record
        columns = list(data[0].keys())
        columns_str = ", ".join(columns)
        placeholders = ", ".join(["%s"] * len(columns))
        
        # Prepare values
        values = [tuple(record[col] for col in columns) for record in data]
        
        # Build query
        query = f"INSERT INTO {full_table_name} ({columns_str}) VALUES ({placeholders})"
        if return_id:
            query += " RETURNING id"
        
        if len(values) == 1:
            result = self.execute(query, params=values[0], fetch=return_id)
            return result[0]['id'] if return_id and result else None
        else:
            # Batch insert
            with self._cursor(dict_cursor=False) as cursor:
                cursor.executemany(query, values)
                self.logger.info(f"Inserted {len(values)} rows into '{full_table_name}'")
            return None
    
    def select(
        self,
        table_name: str,
        columns: List[str] = None,
        where: Dict[str, Any] = None,
        schema: str = None,
        order_by: str = None,
        limit: int = None
    ) -> List[Dict]:
        """
        Select data from a table.
        
        Args:
            table_name: Name of the table
            columns: List of columns to select (default: all columns)
            where: Dictionary of column:value pairs for WHERE clause
            schema: Schema name (optional)
            order_by: ORDER BY clause (e.g., 'created_at DESC')
            limit: Maximum number of rows to return
            
        Returns:
            List of result dictionaries
            
        Example:
            >>> db.select('users', where={'name': 'Eunsu'})
            >>> db.select('users', columns=['id', 'name'], order_by='created_at DESC', limit=10)
        """
        full_table_name = f"{schema}.{table_name}" if schema else table_name
        
        # Build SELECT clause
        columns_str = ", ".join(columns) if columns else "*"
        query = f"SELECT {columns_str} FROM {full_table_name}"
        
        # Build WHERE clause
        params = []
        if where:
            where_clauses = []
            for col, val in where.items():
                where_clauses.append(f"{col} = %s")
                params.append(val)
            query += " WHERE " + " AND ".join(where_clauses)
        
        # Add ORDER BY
        if order_by:
            query += f" ORDER BY {order_by}"
        
        # Add LIMIT
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute(query, params=tuple(params) if params else None, fetch=True)
    
    def select_date_range(
        self,
        table_name: str,
        date_column: str,
        start_date,
        end_date,
        columns: List[str] = None,
        where: Dict[str, Any] = None,
        schema: str = None,
        order_by: str = None,
        limit: int = None,
        inclusive_end: bool = False
    ) -> List[Dict]:
        """
        Select data within a date range.
        
        Args:
            table_name: Name of the table
            date_column: Name of the date/timestamp column
            start_date: Start datetime (inclusive)
            end_date: End datetime (exclusive by default, inclusive if inclusive_end=True)
            columns: List of columns to select (default: all columns)
            where: Additional WHERE conditions (dictionary)
            schema: Schema name (optional)
            order_by: ORDER BY clause (e.g., 'date DESC')
            limit: Maximum number of rows to return
            inclusive_end: If True, use <= for end_date; if False, use < (default: False)
            
        Returns:
            List of result dictionaries
            
        Example:
            >>> from datetime import datetime
            >>> start = datetime(2024, 1, 1)
            >>> end = datetime(2024, 12, 31)
            >>> db.select_date_range('observations', 'date', start, end)
            >>> db.select_date_range('observations', 'timestamp', start, end, 
            ...                      where={'region': 'AR12345'}, 
            ...                      order_by='timestamp DESC')
        """
        full_table_name = f"{schema}.{table_name}" if schema else table_name
        
        # Build SELECT clause
        columns_str = ", ".join(columns) if columns else "*"
        query = f"SELECT {columns_str} FROM {full_table_name}"
        
        # Build WHERE clause
        params = []
        where_clauses = []
        
        # Date range condition
        end_operator = "<=" if inclusive_end else "<"
        where_clauses.append(f"{date_column} >= %s AND {date_column} {end_operator} %s")
        params.extend([start_date, end_date])
        
        # Additional WHERE conditions
        if where:
            for col, val in where.items():
                where_clauses.append(f"{col} = %s")
                params.append(val)
        
        query += " WHERE " + " AND ".join(where_clauses)
        
        # Add ORDER BY
        if order_by:
            query += f" ORDER BY {order_by}"
        
        # Add LIMIT
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute(query, params=tuple(params), fetch=True)
    
    def update(
        self,
        table_name: str,
        data: Dict[str, Any],
        where: Dict[str, Any],
        schema: str = None
    ) -> int:
        """
        Update data in a table.
        
        Args:
            table_name: Name of the table
            data: Dictionary of column:value pairs to update
            where: Dictionary of column:value pairs for WHERE clause
            schema: Schema name (optional)
            
        Returns:
            Number of affected rows
            
        Example:
            >>> db.update('users', {'email': 'new@example.com'}, where={'name': 'Eunsu'})
        """
        full_table_name = f"{schema}.{table_name}" if schema else table_name
        
        # Build SET clause
        set_clauses = []
        params = []
        for col, val in data.items():
            set_clauses.append(f"{col} = %s")
            params.append(val)
        
        query = f"UPDATE {full_table_name} SET " + ", ".join(set_clauses)
        
        # Build WHERE clause
        if where:
            where_clauses = []
            for col, val in where.items():
                where_clauses.append(f"{col} = %s")
                params.append(val)
            query += " WHERE " + " AND ".join(where_clauses)
        else:
            raise ValueError("WHERE clause is required for UPDATE operation")
        
        with self._cursor(dict_cursor=False) as cursor:
            cursor.execute(query, tuple(params))
            return cursor.rowcount
    
    def delete(
        self,
        table_name: str,
        where: Dict[str, Any],
        schema: str = None
    ) -> int:
        """
        Delete data from a table.
        
        Args:
            table_name: Name of the table
            where: Dictionary of column:value pairs for WHERE clause
            schema: Schema name (optional)
            
        Returns:
            Number of deleted rows
            
        Example:
            >>> db.delete('users', where={'name': 'Eunsu'})
        """
        full_table_name = f"{schema}.{table_name}" if schema else table_name
        
        query = f"DELETE FROM {full_table_name}"
        
        # Build WHERE clause
        params = []
        if where:
            where_clauses = []
            for col, val in where.items():
                where_clauses.append(f"{col} = %s")
                params.append(val)
            query += " WHERE " + " AND ".join(where_clauses)
        else:
            raise ValueError("WHERE clause is required for DELETE operation")
        
        with self._cursor(dict_cursor=False) as cursor:
            cursor.execute(query, tuple(params))
            return cursor.rowcount
    
    # ==================== Utility Operations ====================
    
    def count(self, table_name: str, where: Dict[str, Any] = None, schema: str = None) -> int:
        """
        Count rows in a table.
        
        Args:
            table_name: Name of the table
            where: Dictionary of column:value pairs for WHERE clause (optional)
            schema: Schema name (optional)
            
        Returns:
            Number of rows
        """
        full_table_name = f"{schema}.{table_name}" if schema else table_name
        query = f"SELECT COUNT(*) as count FROM {full_table_name}"
        
        params = []
        if where:
            where_clauses = []
            for col, val in where.items():
                where_clauses.append(f"{col} = %s")
                params.append(val)
            query += " WHERE " + " AND ".join(where_clauses)
        
        result = self.execute(query, params=tuple(params) if params else None, fetch=True)
        return result[0]['count'] if result else 0
    
    def truncate(self, table_name: str, schema: str = None, cascade: bool = False):
        """
        Truncate a table (remove all rows).
        
        Args:
            table_name: Name of the table
            schema: Schema name (optional)
            cascade: If True, truncate dependent tables
        """
        full_table_name = f"{schema}.{table_name}" if schema else table_name
        cascade_str = "CASCADE" if cascade else ""
        query = f"TRUNCATE TABLE {full_table_name} {cascade_str}"
        self.execute(query)
        self.logger.info(f"Table '{full_table_name}' truncated")
    
    def vacuum(self, table_name: str = None, full: bool = False, analyze: bool = True):
        """
        Vacuum database or table.
        
        Args:
            table_name: Name of the table (None for entire database)
            full: If True, perform VACUUM FULL (more thorough but slower)
            analyze: If True, update statistics
        """
        # Vacuum requires autocommit
        old_autocommit = self.conn.autocommit
        self.conn.autocommit = True
        
        query_parts = ["VACUUM"]
        if full:
            query_parts.append("FULL")
        if analyze:
            query_parts.append("ANALYZE")
        if table_name:
            query_parts.append(table_name)
        
        query = " ".join(query_parts)
        self.execute(query)
        
        self.conn.autocommit = old_autocommit
        self.logger.info(f"Vacuum completed")


# Utility functions
def to_dataframe(results: List[Dict], parse_dates: List[str] = None):
    """
    Convert query results to pandas DataFrame.
    
    Args:
        results: List of dictionaries from select/execute queries
        parse_dates: List of column names to parse as datetime (optional)
        
    Returns:
        pandas.DataFrame
        
    Example:
        >>> results = db.select('observations')
        >>> df = to_dataframe(results, parse_dates=['date', 'timestamp'])
        
    Note:
        Requires pandas to be installed: pip install pandas
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for to_dataframe(). "
            "Install it with: pip install pandas"
        )
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # Parse date columns if specified
    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
    
    return df

