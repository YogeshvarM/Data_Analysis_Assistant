from langgraph.graph import StateGraph
from langgraph.State import InputState, OutputState
from langgraph.graph import END
from langgraph.SQLAgent import SQLAgent
from langgraph.DataFormatter import DataFormatter


class WorkflowManager:
    def __init__(self, db_path: str = None, api_key: str = None):
        self.db_path = db_path
        self.sql_agent = SQLAgent(db_path=db_path, api_key=api_key) if db_path else SQLAgent(api_key=api_key)
        self.data_formatter = DataFormatter(api_key=api_key)

    def create_workflow(self) -> StateGraph:
        """Create and configure the workflow graph."""
        workflow = StateGraph(input=InputState, output=OutputState)

        # Add nodes to the graph
        workflow.add_node("parse_question", self.sql_agent.parse_question)
        workflow.add_node("get_unique_nouns", self.sql_agent.get_unique_nouns)
        workflow.add_node("generate_sql", self.sql_agent.generate_sql)
        workflow.add_node("validate_and_fix_sql", self.sql_agent.validate_and_fix_sql)
        workflow.add_node("execute_sql", self.sql_agent.execute_sql)
        workflow.add_node("format_results", self.sql_agent.format_results)
        workflow.add_node("choose_visualization", self.sql_agent.choose_visualization)
        workflow.add_node("format_data_for_visualization", self.data_formatter.format_data_for_visualization)
        
        # Define edges
        workflow.add_edge("parse_question", "get_unique_nouns")
        workflow.add_edge("get_unique_nouns", "generate_sql")
        workflow.add_edge("generate_sql", "validate_and_fix_sql")
        workflow.add_edge("validate_and_fix_sql", "execute_sql")
        workflow.add_edge("execute_sql", "format_results")
        workflow.add_edge("execute_sql", "choose_visualization")
        workflow.add_edge("choose_visualization", "format_data_for_visualization")
        workflow.add_edge("format_data_for_visualization", END)
        workflow.add_edge("format_results", END)
        workflow.set_entry_point("parse_question")

        return workflow
    
    def returnGraph(self):
        return self.create_workflow().compile()

    def run_sql_agent(self, question: str, chat_history: list) -> dict:
        """Run the SQL agent workflow and return the formatted answer and visualization recommendation."""
        app = self.create_workflow().compile()
        result = app.invoke({
            "question": question, 
            "chat_history": chat_history
        })
        
        return {
            "answer": result['answer'],
            "visualization": result.get('visualization'),
            "visualization_reason": result.get('visualization_reason'),
            "formatted_data_for_visualization": result.get('formatted_data_for_visualization')
        }