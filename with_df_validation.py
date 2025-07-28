import streamlit as st
import pandas as pd
import pandasql as ps
import json
import os
import requests
import threading
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for data management
DATA_FILE = "database.json"
API_URL = os.getenv("API_URL")
UPDATE_INTERVAL = 20 * 60  # 20 minutes in seconds

# Configuration for large dataset handling
CONFIG = {
    'MAX_ROWS_FOR_LLM': 50,      # Maximum rows to send to LLM for response generation
    'MAX_CHARS_FOR_LLM': 2000,   # Maximum characters to send to LLM
    'LARGE_RESULT_THRESHOLD': 100, # Threshold for "large" results in UI
    'PREVIEW_ROWS': 10,          # Number of rows to show in preview
    'MAX_SAMPLE_ROWS': 10        # Maximum rows for LLM analysis
}

# Load environment variables
load_dotenv()

def load_data_from_file():
    """Load data from JSON file"""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                json_data = json.load(f)
            df = pd.DataFrame(json_data)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df, datetime.now()
        else:
            logger.warning(f"File {DATA_FILE} not found")
            return pd.DataFrame(), None
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), None

def fetch_data_from_api():
    """Fetch data from API and save to JSON file"""
    if not API_URL:
        logger.warning("No API URL configured")
        return False
        
    try:
        response = requests.get(API_URL, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        data = data["ServiceRes"]
        
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Data updated from API. Records: {len(data) if isinstance(data, list) else 'N/A'}")
        return True
        
    except Exception as e:
        logger.error(f"Error fetching from API: {str(e)}")
        return False

def should_update_data(last_update):
    """Check if data should be updated"""
    if not last_update:
        return True
    return datetime.now() - last_update > timedelta(seconds=UPDATE_INTERVAL)

def get_current_data():
    """Get current data, update if necessary"""
    # Check session state for cached data
    if 'df' not in st.session_state or 'last_update' not in st.session_state:
        df, last_update = load_data_from_file()
        st.session_state.df = df
        st.session_state.last_update = last_update
    
    # Check if we need to update
    if should_update_data(st.session_state.last_update):
        if API_URL and fetch_data_from_api():
            df, last_update = load_data_from_file()
            st.session_state.df = df
            st.session_state.last_update = last_update
    
    return st.session_state.df

def initialize_llm():
    """Initialize the language model"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
        st.stop()
    
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=api_key)

def create_sql_prompt():
    """Create the SQL generation prompt template"""
    return PromptTemplate(
        input_variables=["user_input", "columns", "sample_data"],
        template="""
You are a pandasql expert working with a DataFrame called 'df'.

Table Information:
- Columns: {columns}
- Sample data (first 3 rows): {sample_data}

Generate a **SQLite-compatible SQL query** that answers the user's question.
- Use proper SQL syntax for pandasql
- Handle case-insensitive searches when appropriate
- Use proper aggregations and grouping
- Return only the SQL query, no explanation

User question: {user_input}
SQL query:"""
    )

def create_response_prompt():
    """Create the response generation prompt template"""
    return PromptTemplate(
        input_variables=["user_input", "result"],
        template="""
Generate a user-friendly answer based on the user's question and query result.
Make it conversational and informative.

User question: {user_input}
Query result: {result}

Provide a clear, concise answer:"""
    )

def generate_sql_query(user_input, df, llm):
    """Generate SQL query from user input"""
    sql_prompt = create_sql_prompt()
    sql_chain = LLMChain(llm=llm, prompt=sql_prompt)
    
    columns = ", ".join(df.columns)
    sample_data = df.head(3).to_string() if not df.empty else "No data available"
    
    return sql_chain.run({
        "user_input": user_input,
        "columns": columns,
        "sample_data": sample_data
    })

def generate_user_response(user_input, result, llm):
    """Generate user-friendly response - optimized for large datasets"""
    
    if result.empty:
        return "No results found for your query."
    
    # Check if result is too large to send to LLM
    result_str = result.to_string()
    is_large_result = (len(result) > CONFIG['MAX_ROWS_FOR_LLM'] or 
                      len(result_str) > CONFIG['MAX_CHARS_FOR_LLM'])
    
    if is_large_result:
        # For large results, provide a summary without LLM
        total_rows = len(result)
        total_cols = len(result.columns)
        
        # Get basic statistics
        summary_parts = [
            f"Found {total_rows:,} records with {total_cols} columns."
        ]
        
        # Add column info
        if total_cols <= 10:  # Only show column names if not too many
            summary_parts.append(f"Columns: {', '.join(result.columns)}")
        
        # Add basic stats for numeric columns
        numeric_cols = result.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary_parts.append("The complete results are displayed in the table below.")
        
        return " ".join(summary_parts)
    
    else:
        # For smaller results, use LLM to generate friendly response
        response_prompt = create_response_prompt()
        response_chain = LLMChain(llm=llm, prompt=response_prompt)
        
        # Limit the data sent to LLM
        limited_result = result.head(CONFIG['MAX_SAMPLE_ROWS']) if len(result) > CONFIG['MAX_SAMPLE_ROWS'] else result
        limited_result_str = limited_result.to_string()
        
        try:
            response = response_chain.run({
                "user_input": user_input,
                "result": limited_result_str
            })
            
            # If original result was truncated, mention it
            if len(result) > CONFIG['MAX_SAMPLE_ROWS']:
                response += f"\n\n(Showing analysis of first {CONFIG['MAX_SAMPLE_ROWS']} rows. Total results: {len(result)} rows)"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            # Fallback to simple response
            return f"Query executed successfully. Found {len(result)} records. Results are displayed below."

def execute_query(sql_query, df):
    """Execute SQL query using pandasql"""
    try:
        result = ps.sqldf(sql_query, {"df": df})
        return result, None
    except Exception as e:
        return None, str(e)

def process_user_question(user_input, df, llm):
    """Process user question and return complete response"""
    try:
        if df.empty:
            return {
                "error": "No data available",
                "sql": None,
                "result": None,
                "response": "I don't have any data to query. Please check your database file."
            }
        
        # Generate SQL
        sql_query = generate_sql_query(user_input, df, llm)
        
        # Execute SQL
        result, error = execute_query(sql_query, df)
        
        if error:
            return {
                "error": error,
                "sql": sql_query,
                "result": None,
                "response": f"I encountered an error executing the query: {error}"
            }
        
        # Generate user-friendly response
        response = generate_user_response(user_input, result, llm)
        
        return {
            "error": None,
            "sql": sql_query,
            "result": result,
            "response": response
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {
            "error": str(e),
            "sql": None,
            "result": None,
            "response": f"I encountered an error: {str(e)}"
        }

def display_sidebar_info(df):
    """Display database information in sidebar"""
    with st.sidebar:
        st.header("Database Info")
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            if API_URL and fetch_data_from_api():
                df, last_update = load_data_from_file()
                st.session_state.df = df
                st.session_state.last_update = last_update
                st.success("Data refreshed!")
                st.rerun()
            else:
                df, last_update = load_data_from_file()
                st.session_state.df = df
                st.session_state.last_update = last_update
                st.info("Data reloaded from file")
                st.rerun()
        
        # Show data info
        if not df.empty:
            st.success(f"✅ {len(df)} records loaded")
            st.write(f"**Columns:** {', '.join(df.columns)}")
            if st.session_state.get('last_update'):
                st.write(f"**Last Update:** {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Show sample data
            with st.expander("Sample Data"):
                st.dataframe(df.head())
        else:
            st.error("❌ No data loaded")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

def display_chat_history():
    """Display chat history with optimized display for large results"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    for i, chat in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**You:** {chat['user']}")
            st.markdown(f"**Bot:** {chat['bot']}")
            
            if chat.get('sql') and chat.get('result') is not None:
                result_df = chat['result']
                
                # Determine how to display results based on size
                if not result_df.empty:
                    result_size = len(result_df)
                    
                    if result_size > CONFIG['LARGE_RESULT_THRESHOLD']:
                        # For large results, show summary first
                        with st.expander(f"📊 View Results & SQL #{i+1} ({result_size:,} rows)"):
                            st.code(chat['sql'], language='sql')
                            
                            # Show summary statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Rows", f"{result_size:,}")
                            with col2:
                                st.metric("Columns", len(result_df.columns))
                            with col3:
                                if result_df.select_dtypes(include=['number']).columns.any():
                                    numeric_cols = len(result_df.select_dtypes(include=['number']).columns)
                                    st.metric("Numeric Columns", numeric_cols)
                            
                            # Show first and last few rows
                            st.subheader("📄 Data Preview")
                            
                            # First N rows
                            st.write(f"**First {CONFIG['PREVIEW_ROWS']} rows:**")
                            st.dataframe(result_df.head(CONFIG['PREVIEW_ROWS']), use_container_width=True)
                            
                            if result_size > (CONFIG['PREVIEW_ROWS'] * 2):
                                st.write(f"**Last {CONFIG['PREVIEW_ROWS']} rows:**")
                                st.dataframe(result_df.tail(CONFIG['PREVIEW_ROWS']), use_container_width=True)
                            
                            # Download option for large datasets
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Full Results as CSV",
                                data=csv,
                                file_name=f"query_results_{i+1}.csv",
                                mime="text/csv"
                            )
                    else:
                        # For smaller results, show everything
                        with st.expander(f"View SQL & Results #{i+1} ({result_size} rows)"):
                            st.code(chat['sql'], language='sql')
                            st.dataframe(result_df, use_container_width=True)
                else:
                    with st.expander(f"View SQL & Results #{i+1}"):
                        st.code(chat['sql'], language='sql')
                        st.write("No results found")
            
            st.divider()

def display_example_queries():
    """Display example queries"""
    st.markdown("---")
    st.header("💡 Example Questions")
    
    examples = [
        "What are the top 5 suppliers by PO quantity?",
        "Show me all shipments from January",
        "Which buyer has the highest shipped quantity?",
        "What's the total quantity for each style?",
        "Show me orders where shipped quantity is less than PO quantity"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                # Simulate user input
                st.session_state.example_clicked = example
                st.rerun()

def handle_example_query(llm, df):
    """Handle example query if clicked"""
    if 'example_clicked' in st.session_state:
        user_input = st.session_state.example_clicked
        del st.session_state.example_clicked
        
        with st.spinner("Processing example question..."):
            result = process_user_question(user_input, df, llm)
        
        # Add to chat history
        chat_entry = {
            "user": user_input,
            "bot": result["response"],
            "sql": result["sql"],
            "result": result["result"],
            "error": result["error"]
        }
        st.session_state.chat_history.append(chat_entry)
        st.rerun()

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="Talk to Database",
        page_icon="🗣️",
        layout="wide"
    )
    
    st.title("🗣️ Talk to Database")
    st.markdown("Ask questions about your data in natural language!")
    
    # Initialize components
    try:
        llm = initialize_llm()
        df = get_current_data()
        
        # Display sidebar
        display_sidebar_info(df)
        
        # Handle example queries
        handle_example_query(llm, df)
        
        # Chat interface
        st.header("💬 Chat Interface")
        
        # Display chat history
        display_chat_history()
        
        # Chat input
        user_input = st.chat_input("Ask a question about your data...")
        
        if user_input:
            with st.spinner("Processing your question..."):
                result = process_user_question(user_input, df, llm)
            
            # Add to chat history
            chat_entry = {
                "user": user_input,
                "bot": result["response"],
                "sql": result["sql"],
                "result": result["result"],
                "error": result["error"]
            }
            st.session_state.chat_history.append(chat_entry)
            st.rerun()
        
        # Display example queries
        display_example_queries()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")

# Background data updater function (optional)
def start_background_updater():
    """Start background data updater if API_URL is configured"""
    if not API_URL:
        return
    
    def update_loop():
        while True:
            time.sleep(UPDATE_INTERVAL)
            fetch_data_from_api()
    
    thread = threading.Thread(target=update_loop, daemon=True)
    thread.start()
    logger.info("Background data updater started")

if __name__ == "__main__":
    # Start background updater
    start_background_updater()
    
    # Run main app
    main()