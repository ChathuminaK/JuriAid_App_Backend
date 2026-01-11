import os
import json
from datetime import datetime
from typing import Dict, Any, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from langchain.memory import ConversationBufferMemory

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import settings

from orchestrator.core import analyze_case_content, generate_legal_summary
from orchestrator.tools import (
    retrieve_family_law_statutes,
    find_divorce_case_precedents,
    generate_family_law_client_questions,
    retrieve_contract_law_statutes,
    generate_contract_review_questions,
    summarize_case,
    update_knowledge_base
)


class LangChainPlannerAgent:
    """LangChain-powered legal case analysis agent"""
    
    def __init__(self, kb_path: str, model_name: str = "gemini-2.0-flash-exp"):
        self.kb_path = kb_path
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.7,
            convert_system_message_to_human=True  # For Gemini compatibility
        )
        
        # Define tools
        self.tools = self._create_tools()
        
        # Create agent
        self.agent_executor = self._create_agent()
        
        print(f"[LangChainPlannerAgent] Initialized with model: {model_name}")
    
    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools from existing functions"""
        return [
            Tool(
                name="family_statutes",
                func=retrieve_family_law_statutes,
                description="Retrieve relevant Family Law statutes from Sri Lankan law. Use this for divorce, custody, maintenance cases."
            ),
            Tool(
                name="family_precedents",
                func=find_divorce_case_precedents,
                description="Find similar divorce and family law case precedents. Use for legal precedent research."
            ),
            Tool(
                name="family_questions",
                func=generate_family_law_client_questions,
                description="Generate client interview questions for family law cases. Use to prepare for client meetings."
            ),
            Tool(
                name="contract_statutes",
                func=retrieve_contract_law_statutes,
                description="Retrieve Contract Law statutes. Use for contract disputes, breach of contract cases."
            ),
            Tool(
                name="contract_questions",
                func=generate_contract_review_questions,
                description="Generate contract review questions. Use for contract analysis and due diligence."
            ),
            Tool(
                name="summarize",
                func=lambda x: summarize_case(x),
                description="Summarize a legal case into key points. Always use this for case overview."
            ),
            Tool(
                name="update_kb",
                func=lambda x: self._update_kb_wrapper(x),
                description="Save case to knowledge base for future reference. Use when user wants to save/store case."
            ),
        ]
    
    def _update_kb_wrapper(self, case_text: str) -> Dict[str, Any]:
        """Wrapper for update_knowledge_base with context"""
        analysis = analyze_case_content(case_text)
        metadata = {
            "case_type": analysis["case_type"],
            "tags": self._extract_tags(case_text),
            "timestamp": datetime.now().isoformat()
        }
        return update_knowledge_base(case_text, metadata, self.kb_path)
    
    def _extract_tags(self, case_text: str) -> List[str]:
        """Extract relevant tags from case text"""
        keywords = ["divorce", "custody", "maintenance", "contract", "breach", "agreement", 
                   "property", "child", "alimony", "separation"]
        text_lower = case_text.lower()
        return [kw for kw in keywords if kw in text_lower]
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent with structured chat"""
        
        # System prompt
        system_message = """You are a Sri Lankan legal analysis AI assistant specializing in Family Law and Contract Law.

Your task is to analyze legal cases and use available tools to provide comprehensive legal insights.

Available tools:
- family_statutes: Get Family Law statutes
- family_precedents: Find divorce case precedents  
- family_questions: Generate client interview questions
- contract_statutes: Get Contract Law statutes
- contract_questions: Generate contract review questions
- summarize: Summarize the case
- update_kb: Save case to knowledge base

IMPORTANT:
1. Always start by summarizing the case
2. Identify case type (Family Law or Contract Law)
3. Use relevant tools based on case type
4. Provide structured, professional legal analysis
5. Format output clearly with sections and bullet points

Think step by step about which tools to use."""

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent
        agent = create_structured_chat_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create executor with memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=memory,
            verbose=True,  # Show reasoning steps
            max_iterations=5,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    def run(self, case_text: str, prompt: str) -> Dict[str, Any]:
        """
        Run the LangChain agent on a case
        
        Args:
            case_text: The legal case text
            prompt: User's instruction/question
            
        Returns:
            Structured analysis results
        """
        # Prepare input
        full_input = f"""Case Type Analysis Needed

User Request: {prompt}

Case Details:
{case_text[:2000]}...

Please analyze this case and provide comprehensive legal insights using the available tools."""

        try:
            # Run agent
            result = self.agent_executor.invoke({"input": full_input})
            
            # Extract intermediate steps (tool calls)
            steps = []
            if "intermediate_steps" in result:
                for action, observation in result["intermediate_steps"]:
                    steps.append({
                        "tool": action.tool,
                        "tool_input": action.tool_input,
                        "output": observation
                    })
            
            # Analyze case for metadata
            analysis = analyze_case_content(case_text)
            
            # Format response
            return {
                "success": True,
                "case_type": analysis["case_type"],
                "agent_output": result.get("output", ""),
                "tool_calls": steps,
                "prompt": prompt,
                "timestamp": datetime.now().isoformat(),
                "framework": "langchain",
                "model": "gemini-2.0-flash-exp"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "case_type": "unknown",
                "timestamp": datetime.now().isoformat()
            }