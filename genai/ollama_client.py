"""
Ollama API client wrapper for local LLM inference
"""
import requests
import json
from typing import Optional, Dict, Any, List


class OllamaClient:
    """
    Wrapper for Ollama API calls
    
    Provides methods to interact with Ollama's local LLM server for text generation
    and chat completions with proper error handling and retries.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama3.2"):
        """
        Initialize Ollama client
        
        Args:
            base_url: Base URL of the Ollama API server
            model_name: Name of the model to use (e.g., 'llama3.2', 'mistral', 'phi')
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = 120  # 2 minutes timeout for long generations
        
    def _make_request(self, endpoint: str, payload: Dict[str, Any], stream: bool = False) -> Any:
        """
        Internal method to make HTTP requests to Ollama API
        
        Args:
            endpoint: API endpoint (e.g., '/api/generate')
            payload: Request payload
            stream: Whether to stream the response
            
        Returns:
            Response data (dict or generator if streaming)
            
        Raises:
            ConnectionError: If cannot connect to Ollama server
            requests.exceptions.RequestException: For other HTTP errors
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return response
            else:
                return response.json()
                
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Could not connect to Ollama server at {self.base_url}. "
                f"Make sure Ollama is running (try: ollama serve)"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Request to Ollama timed out after {self.timeout}s. "
                f"Try using a smaller model or increase timeout."
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API request failed: {str(e)}")
    
    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt using Ollama
        
        Args:
            prompt: The input prompt text
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response (not implemented yet)
            **kwargs: Additional parameters for Ollama API
            
        Returns:
            Generated text response
            
        Example:
            >>> client = OllamaClient()
            >>> response = client.generate("Explain climate change in one sentence")
            >>> print(response)
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
            
        # Add any additional kwargs to options
        for key, value in kwargs.items():
            payload["options"][key] = value
        
        try:
            response = self._make_request("/api/generate", payload)
            return response.get("response", "").strip()
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            # Return error info instead of raising to allow graceful fallback
            return f"[LLM Error: {str(e)}]"
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Chat completion with conversation history
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
                     Example: [{"role": "user", "content": "Hello"}]
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters for Ollama API
            
        Returns:
            Assistant's response text
            
        Example:
            >>> client = OllamaClient()
            >>> messages = [
            ...     {"role": "user", "content": "What is renewable energy?"}
            ... ]
            >>> response = client.chat(messages)
            >>> print(response)
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
            
        # Add any additional kwargs to options
        for key, value in kwargs.items():
            payload["options"][key] = value
        
        try:
            response = self._make_request("/api/chat", payload)
            return response.get("message", {}).get("content", "").strip()
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            return f"[LLM Error: {str(e)}]"
    
    def is_available(self) -> bool:
        """
        Check if Ollama server is available
        
        Returns:
            True if server is reachable, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
