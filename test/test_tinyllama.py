"""
Test TinyLlama model loading and generation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.llm_manager import TinyLlamaManager
from app.config import Config

def test_tinyllama():
    """Test TinyLlama initialization and generation"""
    
    print("="*70)
    print("🤖 TINYLLAMA MODEL TEST")
    print("="*70)
    
    print(f"\nModel: {Config.LLM_MODEL_NAME}")
    print(f"Device: {Config.LLM_DEVICE}")
    print(f"Max Tokens: {Config.LLM_MAX_NEW_TOKENS}")
    print(f"Temperature: {Config.LLM_TEMPERATURE}")
    
    print("\n📥 Loading TinyLlama model...")
    print("(First time may take a few minutes to download)")
    
    try:
        llm_manager = TinyLlamaManager()
        print("✓ Model loaded successfully!")
        
        # Test generation
        print("\n🧪 Testing generation...")
        test_context = [
            "Pasta carbonara is made with eggs, cheese, pancetta, and black pepper.",
            "Cook spaghetti in boiling salted water until al dente."
        ]
        test_query = "How do I make pasta carbonara?"
        
        print(f"\nQuery: {test_query}")
        print("\nGenerating response...")
        
        response = llm_manager.generate_response(test_query, test_context)
        
        print("\n" + "="*70)
        print("RESPONSE:")
        print("="*70)
        print(response)
        print("="*70)
        
        print("\n✅ Test successful!")
        
        # Cleanup
        llm_manager.cleanup()
        print("\n🧹 Model cleaned up from memory")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tinyllama()
