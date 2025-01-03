from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.DatabaseManager import DatabaseManager
from langgraph.LLMManager import LLMManager

class SQLAgent:
    def __init__(self, db_path: str = None, api_key: str = None):
        self.db_manager = DatabaseManager(db_path) if db_path else DatabaseManager()
        self.llm_manager = LLMManager(api_key=api_key)

    def parse_question(self, state: dict) -> dict:

        """Parse user question and identify relevant tables and columns."""
        question = state['question']
        chat_history = state.get('chat_history', [])
        schema = self.db_manager.get_schema()
        print('I came here')

        # Get previous results if they exist
        previous_results = None
        previous_viz = None
        for msg in reversed(chat_history):
            if msg["role"] == "assistant":
                if "visualization_data" in msg:
                    previous_viz = msg["visualization_data"]
                if "results" in msg:
                    previous_results = msg["results"]
                    break

        # First check - determine question type
        system_prompt = ChatPromptTemplate.from_template("""You are an expert data analyst and friendly AI assistant who helps users explore and understand their data.

Determine if the user's question is:
1. A greeting or general question
2. A question about system capabilities
3. A question about available data
4. A specific data analysis question
5. A request for summary/explanation of previous analysis

Remember users might:
- Start with casual greetings ("hi", "hello", "hey there")
- Ask vague questions ("what can you tell me about the data?")
- Use informal language ("gimme stats about sales")
- Need help getting started ("where should I begin?")
- Ask for explanations ("what does this mean?")
- Want summaries ("give me a summary", "what did we find?")

Question: {user_question}
Schema Available: {schema_str}
Has Previous Results: {has_previous}

Respond in JSON format:
{{
    "question_type": "greeting" | "generic" | "schema" | "analysis" | "meta_analysis",
    "requires_sql": boolean,
    "requires_previous_context": boolean,
    "generic_response": string | null
}}

Examples:

For "Hello":
{{
    "question_type": "greeting",
    "requires_sql": false,
    "requires_previous_context": false,
    "generic_response": "Hello! I'm here to help you analyze data about [key tables/topics]. Would you like to explore any specific aspect?"
}}

For "What data do you have?":
{{
    "question_type": "schema",
    "requires_sql": false,
    "requires_previous_context": false,
    "generic_response": "I have information about [describe main tables]. You could ask questions like [2-3 examples]."
}}

For "Give summary of this analysis":
{{
    "question_type": "meta_analysis",
    "requires_sql": false,
    "requires_previous_context": true,
    "generic_response": null
}}

Response:""")

        # Check question type
        system_response = self.llm_manager.invoke(
            system_prompt,
            user_question=question,
            schema_str=str(schema),
            has_previous=previous_results is not None
        )
        print(system_response)
        
        question_info = JsonOutputParser().parse(system_response)
        
        # Handle meta-analysis requests
        if question_info["question_type"] == "meta_analysis":
            if not previous_results:
                return {
                    "parsed_question": {
                        "is_relevant": False,  # Changed to false since no tables needed
                        "is_system_query": True,
                        "system_response": "I don't see any previous analysis to summarize. What would you like to know about the data?"
                    }
                }
            return {
                "parsed_question": {
                    "is_relevant": False,  # Changed to false since no tables needed
                    "is_meta_analysis": True,
                    "previous_results": previous_results,
                    "previous_visualization": previous_viz
                }
            }

        # Handle non-analysis questions (greetings, schema, generic)
        if not question_info["requires_sql"]:
            print("I came to greet")
            return {
                "parsed_question": {
                    "is_relevant": True,  # Changed to false since no tables needed
                    "is_system_query": True,
                    "system_response": question_info["generic_response"]
                }
            }

        # For data analysis questions, process with regular analysis prompt
        context_messages = []
        for msg in chat_history[-5:]:
            if msg["role"] == "user":
                context_messages.append(f"Human: {msg['content']}")
            else:
                viz_info = ""
                if "visualization_data" in msg:
                    viz_info = f"\nVisualization used: {msg['visualization_data']['type']} chart"
                context_messages.append(f"Assistant: {msg['content']}{viz_info}")
        
        context = "\n".join(context_messages) if context_messages else "No previous context"

        analysis_prompt = ChatPromptTemplate.from_template("""You are a data analyst that helps analyze and visualize data from a database. You understand both direct questions about data and follow-up requests about visualization changes.

Database Schema:
{schema_str}

Recent Conversation History:
{chat_context}

Current Question:
{user_question}

Analyze the question and determine:
1. If it's a follow-up question about changing visualization
2. If it's a follow-up question about the same data but different analysis
3. If it's a completely new question

Your response should be in the following JSON format:
{{
    "is_relevant": boolean,
    "is_visualization_change": boolean,
    "relevant_tables": [
        {{
            "table_name": string,
            "columns": [string],
            "noun_columns": [string]
        }}
    ],
    "previous_query_relevant": boolean,
    "visualization_preference": string | null,
    "suggested_analysis": string
}}

Response:""")

        response = self.llm_manager.invoke(
            analysis_prompt,
            schema_str=str(schema),
            chat_context=context,
            user_question=question
        )
        print(response)
        output_parser = JsonOutputParser()
        parsed_response = output_parser.parse(response)
        print(parsed_response)
        return {"parsed_question": parsed_response}

    def get_unique_nouns(self, state: dict) -> dict:
        """Find unique nouns in relevant tables and columns."""
        parsed_question = state['parsed_question']
            # Check for system or meta queries first
        if parsed_question.get('is_system_query') or parsed_question.get('is_meta_analysis'):
            return {"unique_nouns": []}
    
        
        if not parsed_question['is_relevant']:
            return {"unique_nouns": []}

        unique_nouns = set()
        for table_info in parsed_question['relevant_tables']:
            table_name = table_info['table_name']
            noun_columns = table_info['noun_columns']
            
            if noun_columns:
                column_names = ', '.join(f"`{col}`" for col in noun_columns)
                query = f"SELECT DISTINCT {column_names} FROM `{table_name}`"
                results = self.db_manager.execute_query(query)
                for row in results:
                    unique_nouns.update(str(value) for value in row if value)

        return {"unique_nouns": list(unique_nouns)}

    def generate_sql(self, state: dict) -> dict:
        """Generate SQL query based on parsed question and unique nouns."""
        question = state['question']
        parsed_question = state['parsed_question']
        unique_nouns = state['unique_nouns']
        if parsed_question.get('is_system_query') or parsed_question.get('is_meta_analysis'):
            return {"sql_query": "NOT_RELEVANT"}

        if not parsed_question['is_relevant']:
            return {"sql_query": "NOT_RELEVANT"}
    
        schema = self.db_manager.get_schema(state)

        prompt = ChatPromptTemplate.from_messages([
            ("system", '''
You are an AI assistant that generates SQL queries based on user questions, database schema, and unique nouns found in the relevant tables. Generate a valid SQL query to answer the user's question.

If there is not enough information to write a SQL query, respond with "NOT_ENOUGH_INFO".

Here are some examples:

1. What is the top selling product?
Answer: SELECT product_name, SUM(quantity) as total_quantity FROM sales WHERE product_name IS NOT NULL AND quantity IS NOT NULL AND product_name != "" AND quantity != "" AND product_name != "N/A" AND quantity != "N/A" GROUP BY product_name ORDER BY total_quantity DESC LIMIT 1

2. What is the total revenue for each product?
Answer: SELECT \`product name\`, SUM(quantity * price) as total_revenue FROM sales WHERE \`product name\` IS NOT NULL AND quantity IS NOT NULL AND price IS NOT NULL AND \`product name\` != "" AND quantity != "" AND price != "" AND \`product name\` != "N/A" AND quantity != "N/A" AND price != "N/A" GROUP BY \`product name\`  ORDER BY total_revenue DESC

3. What is the market share of each product?
Answer: SELECT \`product name\`, SUM(quantity) * 100.0 / (SELECT SUM(quantity) FROM sa  les) as market_share FROM sales WHERE \`product name\` IS NOT NULL AND quantity IS NOT NULL AND \`product name\` != "" AND quantity != "" AND \`product name\` != "N/A" AND quantity != "N/A" GROUP BY \`product name\`  ORDER BY market_share DESC

4. Plot the distribution of income over time
Answer: SELECT income, COUNT(*) as count FROM users WHERE income IS NOT NULL AND income != "" AND income != "N/A" GROUP BY income

THE RESULTS SHOULD ONLY BE IN THE FOLLOWING FORMAT, SO MAKE SURE TO ONLY GIVE TWO OR THREE COLUMNS:
[[x, y]]
or 
[[label, x, y]]
             
For questions like "plot a distribution of the fares for men and women", count the frequency of each fare and plot it. The x axis should be the fare and the y axis should be the count of people who paid that fare.
SKIP ALL ROWS WHERE ANY COLUMN IS NULL or "N/A" or "".
Just give the query string. Do not format it. Make sure to use the correct spellings of nouns as provided in the unique nouns list. All the table and column names should be enclosed in backticks.
'''),
            ("human", '''===Database schema:
{schema}

===User question:
{question}

===Relevant tables and columns:
{parsed_question}

===Unique nouns in relevant tables:
{unique_nouns}

Generate SQL query string'''),
        ])

        response = self.llm_manager.invoke(prompt, schema=schema, question=question, parsed_question=parsed_question, unique_nouns=unique_nouns)
        
        if response.strip() == "NOT_ENOUGH_INFO":
            return {"sql_query": "NOT_RELEVANT"}
        else:
            return {"sql_query": response}

    def validate_and_fix_sql(self, state: dict) -> dict:
        """Validate and fix the generated SQL query."""
        sql_query = state['sql_query']

        if sql_query == "NOT_RELEVANT":
            return {"sql_query": "NOT_RELEVANT", "sql_valid": False}
        
        schema = self.db_manager.get_schema(state)

        prompt = ChatPromptTemplate.from_messages([
            ("system", '''
You are an AI assistant that validates and fixes SQL queries. Your task is to:
1. Check if the SQL query is valid.
2. Ensure all table and column names are correctly spelled and exist in the schema. All the table and column names should be enclosed in backticks.
3. If there are any issues, fix them and provide the corrected SQL query.
4. If no issues are found, return the original query.

Respond in JSON format with the following structure. Only respond with the JSON:
{{
    "valid": boolean,
    "issues": string or null,
    "corrected_query": string
}}
'''),
            ("human", '''===Database schema:
{schema}

===Generated SQL query:
{sql_query}

Respond in JSON format with the following structure. Only respond with the JSON:
{{
    "valid": boolean,
    "issues": string or null,
    "corrected_query": string
}}

For example:
1. {{
    "valid": true,
    "issues": null,
    "corrected_query": "None"
}}
             
2. {{
    "valid": false,
    "issues": "Column USERS does not exist",
    "corrected_query": "SELECT * FROM \`users\` WHERE age > 25"
}}

3. {{
    "valid": false,
    "issues": "Column names and table names should be enclosed in backticks if they contain spaces or special characters",
    "corrected_query": "SELECT * FROM \`gross income\` WHERE \`age\` > 25"
}}
             
'''),
        ])

        output_parser = JsonOutputParser()
        response = self.llm_manager.invoke(prompt, schema=schema, sql_query=sql_query)
        result = output_parser.parse(response)

        if result["valid"] and result["issues"] is None:
            return {"sql_query": sql_query, "sql_valid": True}
        else:
            return {
                "sql_query": result["corrected_query"],
                "sql_valid": result["valid"],
                "sql_issues": result["issues"]
            }

    def execute_sql(self, state: dict) -> dict:
        """Execute SQL query and return results."""
        query = state['sql_query']
     
        
        if query == "NOT_RELEVANT":
            return {"results": "NOT_RELEVANT"}

        try:
            results = self.db_manager.execute_query( query)
            return {"results": results}
        except Exception as e:
            return {"error": str(e)}

    def format_results(self, state: dict) -> dict:
        """Format query results into a human-readable response."""
        question = state['question']
        parsed_question = state['parsed_question']
        results = state['results']
        chat_history = state.get('chat_history', [])

        # Create a context from chat history
        context = "\n".join([
            f"{'Assistant' if msg['role'] == 'assistant' else 'Human'}: {msg['content']}"
            for msg in chat_history[-5:]  # Use last 5 messages for context
        ])
         # First check for system queries
        if parsed_question.get('is_system_query'):
            return {"answer": parsed_question['system_response']}
        
        if parsed_question.get('is_meta_analysis'):
            return {"answer": self._format_meta_analysis(parsed_question)}

        if results == "NOT_RELEVANT":
            return {"answer": "Sorry, I can only give answers relevant to the database."}

        prompt = ChatPromptTemplate.from_messages([
            ("system", '''You are an AI assistant that formats database query results into a human-readable response. 
Give a conclusion to the user's question based on the query results and chat history. 
Consider the context of the conversation when formulating your response.
Do not give the answer in markdown format. Only give the answer in one line.'''),
            ("human", "===Chat History:\n{context}\n\n===User question:\n{question}\n\n===Query results:\n{results}\n\nFormatted response:"),
        ])

        response = self.llm_manager.invoke(prompt, context=context, question=question, results=results)
        return {"answer": response}

    def choose_visualization(self, state: dict) -> dict:
        """Choose an appropriate visualization for the data."""
        question = state['question']
        results = state['results']
        sql_query = state['sql_query']

        if results == "NOT_RELEVANT":
            return {"visualization": "none", "visualization_reasoning": "No visualization needed for irrelevant questions."}

        prompt = ChatPromptTemplate.from_messages([
            ("system", '''
You are an AI assistant that recommends appropriate data visualizations. Based on the user's question, SQL query, and query results, suggest the most suitable type of graph or chart to visualize the data. If no visualization is appropriate, indicate that.

Available chart types and their use cases:
- Bar Graphs: Best for comparing categorical data or showing changes over time when categories are discrete and the number of categories is more than 2. Use for questions like "What are the sales figures for each product?" or "How does the population of cities compare? or "What percentage of each city is male?"
- Horizontal Bar Graphs: Best for comparing categorical data or showing changes over time when the number of categories is small or the disparity between categories is large. Use for questions like "Show the revenue of A and B?" or "How does the population of 2 cities compare?" or "How many men and women got promoted?" or "What percentage of men and what percentage of women got promoted?" when the disparity between categories is large.
- Scatter Plots: Useful for identifying relationships or correlations between two numerical variables or plotting distributions of data. Best used when both x axis and y axis are continuous. Use for questions like "Plot a distribution of the fares (where the x axis is the fare and the y axis is the count of people who paid that fare)" or "Is there a relationship between advertising spend and sales?" or "How do height and weight correlate in the dataset? Do not use it for questions that do not have a continuous x axis."
- Pie Charts: Ideal for showing proportions or percentages within a whole. Use for questions like "What is the market share distribution among different companies?" or "What percentage of the total revenue comes from each product?"
- Line Graphs: Best for showing trends and distributionsover time. Best used when both x axis and y axis are continuous. Used for questions like "How have website visits changed over the year?" or "What is the trend in temperature over the past decade?". Do not use it for questions that do not have a continuous x axis or a time based x axis.

Consider these types of questions when recommending a visualization:
1. Aggregations and Summarizations (e.g., "What is the average revenue by month?" - Line Graph)
2. Comparisons (e.g., "Compare the sales figures of Product A and Product B over the last year." - Line or Column Graph)
3. Plotting Distributions (e.g., "Plot a distribution of the age of users" - Scatter Plot)
4. Trends Over Time (e.g., "What is the trend in the number of active users over the past year?" - Line Graph)
5. Proportions (e.g., "What is the market share of the products?" - Pie Chart)
6. Correlations (e.g., "Is there a correlation between marketing spend and revenue?" - Scatter Plot)

Provide your response in the following format:
Recommended Visualization: [Chart type or "None"]. ONLY use the following names: bar, horizontal_bar, line, pie, scatter, none
Reason: [Brief explanation for your recommendation]
'''),
            ("human", '''
User question: {question}
SQL query: {sql_query}
Query results: {results}

Recommend a visualization:'''),
        ])

        response = self.llm_manager.invoke(prompt, question=question, sql_query=sql_query, results=results)
        
        lines = response.split('\n')
        visualization = lines[0].split(': ')[1]
        reason = lines[1].split(': ')[1]

        return {"visualization": visualization, "visualization_reason": reason}
