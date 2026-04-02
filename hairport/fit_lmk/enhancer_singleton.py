"""
Singleton manager for CodeFormerEnhancer to ensure only one instance is created.
"""
import threading
from typing import Optional


class CodeFormerEnhancerSingleton:
    """Thread-safe singleton for CodeFormerEnhancer."""
    
    _instance = None
    _lock = threading.Lock()
    _enhancer = None
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_enhancer(cls, device: str = 'cuda'):
        """
        Get or create the CodeFormerEnhancer instance.
        
        Args:
            device: Device to run the enhancer on ('cuda' or 'cpu')
            
        Returns:
            CodeFormerEnhancer instance or None if initialization fails
        """
        instance = cls()
        
        # If enhancer already exists and device matches, return it
        if instance._enhancer is not None:
            if instance._device == device:
                return instance._enhancer
            else:
                print(f"Warning: CodeFormer enhancer already initialized on {instance._device}, "
                      f"cannot reinitialize on {device}. Using existing instance.")
                return instance._enhancer
        
        # Create new enhancer instance
        with cls._lock:
            # Double-check after acquiring lock
            if instance._enhancer is None:
                try:
                    from hairport.core import CodeFormerEnhancer
                    print("Initializing CodeFormer enhancer (singleton)...")
                    instance._enhancer = CodeFormerEnhancer(device=device, ultrasharp=True)
                    instance._device = device
                    print(f"✓ CodeFormer enhancer initialized on {device}")
                except Exception as e:
                    print(f"Warning: Failed to initialize CodeFormer enhancer: {e}")
                    instance._enhancer = None
                    instance._device = None
            
            return instance._enhancer
    
    @classmethod
    def reset(cls):
        """Reset the singleton (useful for testing or device changes)."""
        with cls._lock:
            cls._enhancer = None
            cls._device = None
            print("CodeFormer enhancer singleton reset")
