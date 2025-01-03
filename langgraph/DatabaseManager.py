import sqlite3
import pandas as pd
from sqlalchemy import create_engine
import tempfile
from typing import List, Any
from langgraph.State import InputState, OutputState

class DatabaseManager:
    """Class to handle database operations"""

    @staticmethod
    def setup_database(df: pd.DataFrame) -> str:
        """Create and setup SQLite database from dataframe"""
        # Create temporary file for SQLite database
        temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        db_path = temp_db.name
        temp_db.close()
        
        # Create SQLite database and add data
        engine = create_engine(f'sqlite:///{db_path}')
        
        # Convert datetime columns to string to avoid SQLite limitations
        df_copy = df.copy()
        for col in df_copy.select_dtypes(include=['datetime64']):
            df_copy[col] = df_copy[col].astype(str)
        
        df_copy.to_sql('user_data', engine, index=False, if_exists='replace')
        
        return db_path

    def __init__(self, db_path: str = None):
        self.db_path = db_path

    def get_schema(self, state: dict = None) -> dict:
        """Retrieve the database schema."""
        if not self.db_path:
            return {}
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            schema = {}
            for table in tables:
                table_name = table[0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                schema[table_name] = [col[1] for col in columns]
            conn.close()
            return schema
        except sqlite3.Error as e:
            raise Exception(f"Error fetching schema: {str(e)}")

    def execute_query(self, query: str) -> List[Any]:
        """Execute SQL query on the SQLite database and return results."""
        if not self.db_path:
            return []
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            conn.close()
            return results
        except sqlite3.Error as e:
            raise Exception(f"Error executing query: {str(e)}")
