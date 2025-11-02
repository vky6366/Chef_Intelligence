from typing import List, Tuple
import os

class PromptBuilder:
    """
    Custom prompt construction for Method 1
    No LangChain - Direct prompt building
    """

    def __init__(self):
        """Initialize prompt builder"""
        self.base_prompt_path = "templates/base_prompt.txt"

    def load_template(self, template_path: str) -> str:
        """Load prompt template from file"""
        if os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    def build_base_prompt(self, query: str, context_chunks: List[str]) -> Tuple[str, str]:
        """
        Build base prompt for recipe queries
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """

        # Try to load template, fallback to default
        template = self.load_template(self.base_prompt_path)
        
        if template:
            system_prompt = template.split("---USER_PROMPT---")[0].strip()
            user_template = template.split("---USER_PROMPT---")[1].strip()
        else:
            # Default prompts
            system_prompt = """You are Chef Intelligence, an AI culinary assistant.
Use the provided recipe information to answer cooking questions accurately.
Provide clear, step-by-step answers with ingredients and instructions when applicable.
If the context doesn't contain relevant information, politely say so."""
            
            user_template = """Context from recipes: 

{context}

User Question: {query}

Please provide a helpful answer:"""
        
        # Combine context chunks
        context = "\n\n---\n\n".join(context_chunks)
        
        # Build user prompt
        user_prompt = user_template.format(context=context, query=query)
        
        return system_prompt, user_prompt
    