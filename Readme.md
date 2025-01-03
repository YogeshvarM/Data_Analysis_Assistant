# Data Analysis Assistant 🤖

An interactive data analysis tool that combines the power of LLMs (using Groq) with data visualization to help users analyze their datasets through natural language conversations.

> **Note: This is an initial version** 
> - Currently only supports CSV files
> - Label generation needs improvement
> - Response quality improvements in progress
> - More visualization customization options coming soon
> - Enhanced error handling in development

## Features

- 📊 Interactive Data Analysis through Natural Language
- 🔍 Automatic SQL Query Generation
- 📈 Dynamic Data Visualization
- 📝 Automated Dataset Description
- 📊 Comprehensive Data Overview
- 🔄 Real-time Data Processing
- 📱 Responsive UI with Resizable Sidebar

## Architecture
![Workflow Architecture](docs/images/workflow.png)

The above diagram illustrates the data flow and component interaction in our Data Analysis Assistant.
System Architecture
The system is organized into three main layers:
1. Configuration Layer

Initial Setup: Handles user uploads and API configuration

Validates Groq API key for LLM access
Processes uploaded data files (CSV, Excel, JSON, Parquet)
Creates SQLite database from uploaded data
Generates comprehensive data analytics



2. Streamlit Interface

Dashboard View: Interactive data visualization interface

Data Metrics: Statistical overview of the dataset
Visualizations: Dynamic charts and graphs
Chat Interface: Natural language interaction with data



3. Core Processing (SQL Agent Workflow)
The SQL Agent processes queries through a sequential workflow:

parse_question

Analyzes user's natural language input
Identifies analytical intent


get_unique_nouns

Extracts key terms from question
Identifies relevant database fields


generate_sql

Creates SQL query based on intent
Structures query for database


validate_and_fix_sql

Validates SQL syntax
Optimizes query structure
Ensures safe execution


execute_sql

Runs query against SQLite database
Splits into two possible paths:

Path 1: format_results for text output
Path 2: choose_visualization for visual output




Response Generation

Formats results into user-friendly response
Integrates with chat interface
Updates dashboard visualizations



Data Flow
CopyUser Input/Upload
      ↓
Configuration Layer (Initial Setup & Validation)
      ↓
Data Processing & Analytics
      ↓
Dashboard Generation
      ↓
User Query via Chat Interface
      ↓
SQL Agent Workflow Processing
      ↓
Response Generation & Display
## Components

### Core Components

1. **WorkflowManager**: Orchestrates the entire analysis pipeline
2. **SQLAgent**: Handles SQL query generation and execution
3. **DataFormatter**: Formats data for various visualization types
4. **LLMManager**: Manages interactions with the Groq LLM
5. **DatabaseManager**: Handles database connections and operations

### Visualization Types Supported

- Bar Charts (Vertical & Horizontal)
- Line Charts
- Scatter Plots
- Pie Charts

### File Support

- CSV
- Excel (XLSX)
- JSON
- Parquet

## Installation

1. Clone the repository:

git clone https://github.com/yourusername/data-analysis-assistant.git
cd data-analysis-assistant

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Groq API key:
   - Sign up at https://console.groq.com
   - Get your API key
   - Have it ready for the application

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Enter your Groq API key when prompted

3. Upload your dataset (CSV, XLSX, JSON, or Parquet)

4. Start analyzing your data through natural language queries!

## Example Queries

- "Show me the trend of sales over the last 6 months"
- "What's the distribution of customers by region?"
- "Compare the average prices between different product categories"
- "Find the top 5 performing products"

## Project Structure

```
data-analysis-assistant/
├── app.py                    # Main Streamlit application
├── langgraph/
│   ├── DataFormatter.py     # Data formatting for visualizations
│   ├── DatabaseManager.py   # Database operations and management
│   ├── LLMManager.py        # LLM interaction management
│   ├── SQLAgent.py          # SQL query generation and execution
│   ├── State.py            # State management
│   ├── WorkflowManager.py   # Workflow orchestration
│   └── graph_instructions.py # Visualization formatting instructions
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

### Current Limitations
- File Support: Currently only CSV files are supported (Excel, JSON, and Parquet support coming soon)
- Basic visualization settings (customization options in development)
- Label generation needs refinement for better context
- Limited error handling for edge cases
- Basic chat history implementation

### Planned Improvements
- Enhanced LLM response quality
- Smarter context handling
- More visualization types and customization options
- Advanced error handling and user feedback
- Support for multiple file formats
- Improved data validation and preprocessing
- Enhanced chat history with context management

## System Architecture

The system is organized into three main layers:

### 1. Configuration Layer

**Initial Setup**: Handles user uploads and API configuration
- Validates Groq API key for LLM access
- Processes uploaded data files (CSV, Excel, JSON, Parquet)
- Creates SQLite database from uploaded data
- Generates comprehensive data analytics

### 2. Streamlit Interface

**Dashboard View**: Interactive data visualization interface
- Data Metrics: Statistical overview of the dataset
- Visualizations: Dynamic charts and graphs
- Chat Interface: Natural language interaction with data

### 3. Core Processing (SQL Agent Workflow)
The SQL Agent processes queries through a sequential workflow:

**parse_question**
- Analyzes user's natural language input
- Identifies analytical intent

**get_unique_nouns**
- Extracts key terms from question
- Identifies relevant database fields

**generate_sql**
- Creates SQL query based on intent
- Structures query for database

**validate_and_fix_sql**
- Validates SQL syntax
- Optimizes query structure
- Ensures safe execution

**execute_sql**
- Runs query against SQLite database
- Splits into two possible paths:
  - Path 1: format_results for text output
  - Path 2: choose_visualization for visual output

**Response Generation**
- Formats results into user-friendly response
- Integrates with chat interface
- Updates dashboard visualizations

### Data Flow
User Input/Upload
↓
Configuration Layer (Initial Setup & Validation)
↓
Data Processing & Analytics
↓
Dashboard Generation
↓
User Query via Chat Interface
↓
SQL Agent Workflow Processing
↓
Response Generation & Display
