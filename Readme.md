# Data Analysis Assistant 🤖

An interactive data analysis tool that combines the power of LLMs (using Groq) with data visualization to help users analyze their datasets through natural language conversations.

## Features

- 📊 Interactive Data Analysis through Natural Language
- 🔍 Automatic SQL Query Generation
- 📈 Dynamic Data Visualization
- 📝 Automated Dataset Description
- 📊 Comprehensive Data Overview
- 🔄 Real-time Data Processing
- 📱 Responsive UI with Resizable Sidebar

## Components

### Core Components

1. **WorkflowManager**: Orchestrates the entire analysis pipeline
2. **SQLAgent**: Handles SQL query generation and execution
3. **DataFormatter**: Formats data for various visualization types
4. **LLMManager**: Manages interactions with the Groq LLM

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

git clone [https://github.com/yourusername/data-analysis-assistant.git](https://github.com/YogeshvarM/Data_Analysis_Assistant/tree/main)
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

