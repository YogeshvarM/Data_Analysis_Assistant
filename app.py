import streamlit as st
import pandas as pd
from langchain.agents import AgentType
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import json
import os
import numpy as np
from typing import Dict, Any, List, Optional, Union
import sqlite3
import tempfile
from datetime import datetime
import base64
from pathlib import Path
import plotly.graph_objects as go
from streamlit.components.v1 import html
from langchain.agents import initialize_agent
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from langchain.chains import LLMChain
from langgraph.WorkflowManager import WorkflowManager
    
# Add after imports, before page configuration
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set page configuration
st.set_page_config(
    page_title="Data Analysis Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session states
def init_session_state():
    """Initialize all session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(
            memory_key="history",
            input_key="input",
            return_messages=True
        )
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'db_path' not in st.session_state:
        st.session_state.db_path = None
    if 'data_overview' not in st.session_state:
        st.session_state.data_overview = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'config'
    if 'is_configured' not in st.session_state:
        st.session_state.is_configured = False
    if 'data_description' not in st.session_state:
        st.session_state.data_description = None

def add_resizable_sidebar():
    """Add custom HTML/JavaScript for resizable sidebar"""
    sidebar_html = """
    <style>
        section[data-testid="stSidebar"] {
            position: relative;
            resize: horizontal;
            overflow: auto;
            min-width: 200px;
            max-width: 800px;
        }
        section[data-testid="stSidebar"]::after {
            content: '';
            position: absolute;
            right: 0;
            top: 0;
            bottom: 0;
            width: 5px;
            background: #e0e0e0;
            cursor: col-resize;
        }
    </style>
    """
    html(sidebar_html, height=0)

class DataAnalyzer:
    """Class to handle data analysis operations"""
    
    @staticmethod
    def get_data_overview(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive overview of the dataframe"""
        # Initialize the overview dictionary with all required keys
        overview = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # In MB
            'columns': {},
            'correlations': None,
            'visualizations': {
                'metrics': [
                    {"label": "Rows", "value": len(df)},
                    {"label": "Columns", "value": len(df.columns)},
                    {"label": "Memory (MB)", "value": f"{df.memory_usage(deep=True).sum() / 1024**2:.1f}"}
                ],
                'distributions': {},
                'correlation_matrix': None
            }
        }
        
        # Calculate correlations for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            overview['correlations'] = df[numeric_cols].corr().to_dict()
            correlation_df = pd.DataFrame(overview['correlations'])
            overview['visualizations']['correlation_matrix'] = {
                'z': correlation_df.values.tolist(),
                'x': correlation_df.columns.tolist(),
                'y': correlation_df.columns.tolist()
            }
        
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'missing_values': df[col].isna().sum(),
                'missing_percentage': (df[col].isna().sum() / len(df)) * 100,
                'unique_values': df[col].nunique(),
                'memory_usage': df[col].memory_usage(deep=True) / 1024**2
            }
            
            # Add sample values and frequency for categorical/object columns
            if df[col].dtype == 'object' or df[col].dtype == 'category':
                col_info.update({
                    'sample_values': list(set(df[col].dropna().sample(min(5, df[col].nunique())).tolist())),
                    'top_values': df[col].value_counts().head(5).to_dict()
                })
            
            # Add statistical info for numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info.update({
                    'mean': df[col].mean() if not df[col].empty else None,
                    'median': df[col].median() if not df[col].empty else None,
                    'std': df[col].std() if not df[col].empty else None,
                    'min': df[col].min() if not df[col].empty else None,
                    'max': df[col].max() if not df[col].empty else None,
                    'quartiles': df[col].quantile([0.25, 0.5, 0.75]).to_dict()
                })
                
                # Add distribution data for numeric columns
                overview['visualizations']['distributions'][col] = {
                    'y': df[col].dropna().tolist(),
                    'name': col
                }
            
            # Detect potential datetime columns
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col].dropna().head())
                    col_info['potential_datetime'] = True
                except:
                    col_info['potential_datetime'] = False
            
            overview['columns'][col] = col_info
        
        return overview
    @staticmethod
    def get_data_description(df: pd.DataFrame, groq_api_key: str) -> str:
        """Get a brief description of the data using Groq LLM"""
        try:
            # Create LLM instance
            llm = ChatGroq(
                temperature=0.1,
                groq_api_key=groq_api_key,
                model_name="llama-3.3-70b-versatile"
            )
            
            # Prepare sample data for prompt
            sample_data = df.head()
            columns_info = [f"- {col}: {', '.join(map(str, sample_data[col].dropna().head().tolist()))}" 
                           for col in df.columns]
            columns_str = "\n".join(columns_info)
            
            prompt = f"""Given this dataset with {len(df.columns)} columns and {len(df)} rows, 
            here are the column names and some sample values:
            
            {columns_str}
            
            Provide a brief 2-3 line description of what this dataset appears to be about. 
            Keep it concise and professional."""
            
            description = llm.predict(prompt)
            return description.strip()
        except Exception as e:
            logger.error(f"Error getting data description: {str(e)}")
            return "Unable to generate description at this time."

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

class APIKeyValidator:
    """Class to handle API key validation"""
    
    @staticmethod
    def validate_groq_api_key(api_key: str) -> bool:
        """Validate Groq API key by attempting to create a client"""
        try:
            llm = ChatGroq(
                temperature=0.1,
                groq_api_key=api_key,
                model_name="llama-3.3-70b-versatile",
                max_tokens=1
            )
            # Try a simple completion to verify the key
            llm.predict("test")
            return True
        except Exception:
            return False

def render_visualization(viz_type: str, data: dict) -> None:
    """Render visualization based on the type and data structure from backend"""
    if viz_type == "bar":
        fig = go.Figure(data=[
            go.Bar(
                x=data['labels'],
                y=series['data'],
                name=series['label']
            ) for series in data['values']
        ])
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "horizontal_bar":
        fig = go.Figure(data=[
            go.Bar(
                y=data['labels'],
                x=series['data'],
                name=series['label'],
                orientation='h'
            ) for series in data['values']
        ])
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "line":
        fig = go.Figure(data=[
            go.Scatter(
                x=data['xValues'],
                y=series['data'],
                mode='lines+markers',
                name=series['label']
            ) for series in data['yValues']
        ])
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "scatter":
        fig = go.Figure(data=[
            go.Scatter(
                x=[point['x'] for point in series['data']],
                y=[point['y'] for point in series['data']],
                mode='markers',
                name=series['label'],
                text=[f"ID: {point['id']}" for point in series['data']]
            ) for series in data['series']
        ])
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "pie":
        fig = go.Figure(data=[
            go.Pie(
                labels=[item['label'] for item in data],
                values=[item['value'] for item in data]
            )
        ])
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function"""
    init_session_state()
    
    if not st.session_state.is_configured:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title("📊 Data Analysis Assistant")
            
            # Configuration page with improved validation
            groq_api_key = st.text_input(
                "Enter Groq API Key",
                type="password",
                help="Enter your Groq API key. Get one at https://console.groq.com"
            )
            
            if groq_api_key:
                with st.spinner("Validating API key..."):
                    if not APIKeyValidator.validate_groq_api_key(groq_api_key):
                        st.error("❌ Invalid API key. Please check your key and try again.")
                        return
            
            uploaded_file = st.file_uploader(
                "Upload your data file",
                type=['csv', 'xlsx', 'json', 'parquet'],
                help="Supported formats: CSV, Excel, JSON, Parquet"
            )
            
            if uploaded_file and groq_api_key:
                try:
                    with st.spinner("Loading and processing your data..."):
                        # File loading logic with improved error handling
                        file_extension = uploaded_file.name.split('.')[-1].lower()
                        
                        if file_extension not in ['csv', 'xlsx', 'json', 'parquet']:
                            st.error("❌ Unsupported file format")
                            return
                            
                        # Load data based on file type
                        df = None
                        try:
                            if file_extension == 'csv':
                                df = pd.read_csv(uploaded_file)
                            elif file_extension == 'xlsx':
                                df = pd.read_excel(uploaded_file)
                            elif file_extension == 'json':
                                df = pd.read_json(uploaded_file)
                            elif file_extension == 'parquet':
                                df = pd.read_parquet(uploaded_file)
                        except Exception as e:
                            st.error(f"❌ Error reading file: {str(e)}")
                            return
                        
                        if df is None or df.empty:
                            st.error("❌ The uploaded file contains no data")
                            return
                            
                        # Update session state
                        st.session_state.df = df
                        st.session_state.data_overview = DataAnalyzer.get_data_overview(df)
                        st.session_state.db_path = DatabaseManager.setup_database(df)
                        st.session_state.groq_api_key = groq_api_key
                        st.session_state.data_description = DataAnalyzer.get_data_description(df, groq_api_key)
                        st.session_state.is_configured = True
                        
                        st.success("✅ Configuration successful! Redirecting to main interface...")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error during configuration: {str(e)}")
    
    else:
        # Add resizable sidebar
        add_resizable_sidebar()
        
        # Sidebar content
        with st.sidebar:
            st.markdown("<h2 style='text-align: center; font-size: 28px;'>📊 Data Overview</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Created by <a href='https://www.linkedin.com/in/yogeshvar-mugilvannan-70653a200/'>Yogeshvar M</a></p>", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Add description box
            if st.session_state.data_description:
                with st.expander("📝 Dataset Description", expanded=True):
                    st.markdown(
                        f"""
                        <div style='text-align: center; padding: 10px; font-size: 18px;'>
                        {st.session_state.data_description}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                   
            
            # Display metrics
            if 'visualizations' in st.session_state.data_overview:
                cols = st.columns(3)
                for idx, metric in enumerate(st.session_state.data_overview['visualizations']['metrics']):
                    cols[idx].metric(metric["label"], metric["value"])
            
            # Column Analysis in sidebar
            st.markdown("#### Columns")
            for col_name, col_info in st.session_state.data_overview['columns'].items():
                with st.expander(f"{col_name}"):
                    st.write(f"📝 Type: {col_info['dtype']}")
                    st.write(f"❌ Missing: {col_info['missing_percentage']:.1f}%")
                    st.write(f"🔢 Unique: {col_info['unique_values']}")
                    
                    # Display sample unique values
                    sample_values = st.session_state.df[col_name].dropna().unique()[:5]  # Get first 5 unique values
                    if len(sample_values) > 0:
                        st.write("📋 Sample Values:")
                        for val in sample_values:
                            st.write(f"  • {val}")
                    
                    # Show distribution plot for numeric columns
                    if ('visualizations' in st.session_state.data_overview and 
                        'distributions' in st.session_state.data_overview['visualizations'] and
                        col_name in st.session_state.data_overview['visualizations']['distributions']):
                        try:
                            data = st.session_state.data_overview['visualizations']['distributions'][col_name]
                            fig = go.Figure()
                            fig.add_trace(go.Box(
                                y=data['y'],
                                name='Distribution'
                            ))
                            fig.update_layout(
                                title=f"Distribution of {col_name}",
                                showlegend=False,
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not create visualization: {str(e)}")
            
            # Safely check for correlation matrix
            if ('visualizations' in st.session_state.data_overview and 
                'correlation_matrix' in st.session_state.data_overview['visualizations'] and
                st.session_state.data_overview['visualizations']['correlation_matrix']):
                st.subheader("Correlation Analysis")
                corr_data = st.session_state.data_overview['visualizations']['correlation_matrix']
                fig = go.Figure(data=go.Heatmap(
                    z=corr_data['z'],
                    x=corr_data['x'],
                    y=corr_data['y'],
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1
                ))
                fig.update_layout(
                    title='Correlation Matrix',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Reset button
            st.divider()
            if st.button("⚠️ Reset", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        
        # Main chat area - full width
        st.title("💬 Chat with Your Data")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and "visualization_data" in message:
                    render_visualization(
                        message['visualization_data']['type'],
                        message['visualization_data']['data']
                    )
        
        # Chat input and response handling
        if user_input := st.chat_input("Ask me about your data"):
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = WorkflowManager(
                        db_path=st.session_state.db_path,
                        api_key=st.session_state.groq_api_key
                    ).run_sql_agent(user_input, st.session_state.chat_history)
                    
                    # Display the text response
                    st.write(response["answer"])
                    
                    # Create message data for chat history
                    message_data = {
                        "role": "assistant",
                        "content": response["answer"]
                    }
                    
                    # Add visualization data if available
                    if response.get("visualization") != "none" and response.get("formatted_data_for_visualization"):
                        viz_data = {
                            "type": response["visualization"],
                            "data": response["formatted_data_for_visualization"]
                        }
                        message_data["visualization_data"] = viz_data
                        render_visualization(viz_data['type'], viz_data['data'])
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append(message_data)

if __name__ == "__main__":
    main()