"""
Example usage of PostgresManager.

This demonstrates how to use the PostgresManager class for common database operations.
"""

from egghouse.database import PostgresManager
import logging

# Setup logging to see query execution
logging.basicConfig(level=logging.INFO)

def main():
    """Demonstrate PostgresManager usage."""
    
    # Initialize connection
    print("=" * 60)
    print("1. Connecting to PostgreSQL")
    print("=" * 60)
    
    # Option 1: Basic usage
    db = PostgresManager(
        host='localhost',
        port=5432,
        database='test_db',  # Make sure this database exists
        user='your_user',
        password='your_password',
        log_queries=True
    )
    
    # Option 2: Using context manager (auto-close)
    # with PostgresManager(host='localhost', database='test_db', ...) as db:
    #     db.create_table(...)
    
    try:
        # ==================== Database Operations ====================
        print("\n" + "=" * 60)
        print("2. Database Operations")
        print("=" * 60)
        
        # List databases
        databases = db.list_databases()
        print(f"\nAvailable databases: {len(databases)}")
        for db_info in databases[:5]:  # Show first 5
            print(f"  - {db_info['name']}: {db_info['size']}")
        
        # ==================== Schema Operations ====================
        print("\n" + "=" * 60)
        print("3. Schema Operations")
        print("=" * 60)
        
        # Create schema
        db.create_schema('research')
        
        # List schemas
        schemas = db.list_schemas()
        print(f"\nAvailable schemas: {[s['name'] for s in schemas]}")
        
        # ==================== Table Operations ====================
        print("\n" + "=" * 60)
        print("4. Table Operations")
        print("=" * 60)
        
        # Create table
        db.create_table(
            'solar_events',
            {
                'id': 'SERIAL PRIMARY KEY',
                'event_type': 'VARCHAR(50) NOT NULL',
                'start_time': 'TIMESTAMP NOT NULL',
                'end_time': 'TIMESTAMP',
                'intensity': 'FLOAT',
                'description': 'TEXT',
                'created_at': 'TIMESTAMP DEFAULT NOW()'
            },
            schema='research'
        )
        
        # Check if table exists
        exists = db.table_exists('solar_events', schema='research')
        print(f"\nTable 'research.solar_events' exists: {exists}")
        
        # Describe table
        columns = db.describe_table('solar_events', schema='research')
        print(f"\nTable structure:")
        for col in columns:
            print(f"  - {col['name']}: {col['type']} "
                  f"(nullable: {col['is_nullable']}, default: {col['default_value']})")
        
        # List tables
        tables = db.list_tables(schema='research')
        print(f"\nTables in 'research' schema:")
        for table in tables:
            print(f"  - {table['name']}: {table['size']}")
        
        # ==================== Data Operations ====================
        print("\n" + "=" * 60)
        print("5. Data Operations")
        print("=" * 60)
        
        # Insert single record
        print("\n--- INSERT ---")
        db.insert(
            'solar_events',
            {
                'event_type': 'solar_flare',
                'start_time': '2025-01-15 10:30:00',
                'intensity': 8.5,
                'description': 'X-class flare observed'
            },
            schema='research'
        )
        
        # Insert multiple records
        db.insert(
            'solar_events',
            [
                {
                    'event_type': 'CME',
                    'start_time': '2025-01-16 14:20:00',
                    'intensity': 7.2,
                    'description': 'Coronal mass ejection'
                },
                {
                    'event_type': 'solar_wind',
                    'start_time': '2025-01-17 08:15:00',
                    'intensity': 5.8,
                    'description': 'High-speed solar wind stream'
                }
            ],
            schema='research'
        )
        
        # Select data
        print("\n--- SELECT ---")
        all_events = db.select('solar_events', schema='research')
        print(f"Total events: {len(all_events)}")
        for event in all_events:
            print(f"  - {event['event_type']}: {event['start_time']} (intensity: {event['intensity']})")
        
        # Select with WHERE clause
        flares = db.select(
            'solar_events',
            where={'event_type': 'solar_flare'},
            schema='research'
        )
        print(f"\nSolar flares: {len(flares)}")
        
        # Select with columns and ordering
        recent_events = db.select(
            'solar_events',
            columns=['event_type', 'start_time', 'intensity'],
            order_by='intensity DESC',
            limit=2,
            schema='research'
        )
        print(f"\nTop 2 events by intensity:")
        for event in recent_events:
            print(f"  - {event['event_type']}: {event['intensity']}")
        
        # Count
        print("\n--- COUNT ---")
        total = db.count('solar_events', schema='research')
        flare_count = db.count('solar_events', where={'event_type': 'solar_flare'}, schema='research')
        print(f"Total events: {total}")
        print(f"Solar flares: {flare_count}")
        
        # Update
        print("\n--- UPDATE ---")
        affected = db.update(
            'solar_events',
            data={'intensity': 9.0, 'description': 'Major X-class flare observed'},
            where={'event_type': 'solar_flare'},
            schema='research'
        )
        print(f"Updated {affected} rows")
        
        # Verify update
        updated = db.select('solar_events', where={'event_type': 'solar_flare'}, schema='research')
        print(f"Updated flare intensity: {updated[0]['intensity']}")
        
        # Delete
        print("\n--- DELETE ---")
        deleted = db.delete(
            'solar_events',
            where={'event_type': 'solar_wind'},
            schema='research'
        )
        print(f"Deleted {deleted} rows")
        
        # Verify deletion
        remaining = db.count('solar_events', schema='research')
        print(f"Remaining events: {remaining}")
        
        # ==================== Raw SQL ====================
        print("\n" + "=" * 60)
        print("6. Raw SQL Execution")
        print("=" * 60)
        
        # Execute custom query
        result = db.execute(
            """
            SELECT event_type, AVG(intensity) as avg_intensity
            FROM research.solar_events
            GROUP BY event_type
            ORDER BY avg_intensity DESC
            """,
            fetch=True
        )
        print("\nAverage intensity by event type:")
        for row in result:
            print(f"  - {row['event_type']}: {row['avg_intensity']:.2f}")
        
        # ==================== Cleanup ====================
        print("\n" + "=" * 60)
        print("7. Cleanup")
        print("=" * 60)
        
        # Truncate table
        db.truncate('solar_events', schema='research')
        print("Table truncated")
        
        # Drop table
        db.drop_table('solar_events', schema='research')
        print("Table dropped")
        
        # Drop schema
        db.drop_schema('research', cascade=True)
        print("Schema dropped")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close connection
        db.close()
        print("\n" + "=" * 60)
        print("Connection closed")
        print("=" * 60)


if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         PostgresManager Usage Example                        ║
    ║                                                              ║
    ║  Make sure to update the connection parameters:             ║
    ║  - host, port, database, user, password                     ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    main()
