from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_name = "Jayanta8509/m2m100-en-bn-finetuned-opus100"
        
    def load_model(self):
        """Load the M2M100 model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
            logger.info("Tokenizer loaded successfully")
            
            # Load model
            self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name)
            logger.info("Model loaded successfully")
            
            # Set device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info(f"Model moved to device: {self.device}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def translate_en_to_bn(self, text: str) -> str:
        """
        Translate English text to Bengali
        
        Args:
            text (str): English text to translate
            
        Returns:
            str: Bengali translation
        """
        try:
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model not loaded. Call load_model() first.")
            
            # Set source and target languages
            self.tokenizer.src_lang = "en"
            self.tokenizer.tgt_lang = "bn"
            
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.get_lang_id("bn"),
                    max_length=128,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode translation
            translation = self.tokenizer.batch_decode(
                generated_tokens, 
                skip_special_tokens=True
            )[0]
            
            return translation
            
        except Exception as e:
            logger.error(f"Error during translation: {str(e)}")
            raise e
    
    def translate_bn_to_en(self, text: str) -> str:
        """
        Translate Bengali text to English
        
        Args:
            text (str): Bengali text to translate
            
        Returns:
            str: English translation
        """
        try:
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model not loaded. Call load_model() first.")
            
            # Set source and target languages
            self.tokenizer.src_lang = "bn"
            self.tokenizer.tgt_lang = "en"
            
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.get_lang_id("en"),
                    max_length=128,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode translation
            translation = self.tokenizer.batch_decode(
                generated_tokens, 
                skip_special_tokens=True
            )[0]
            
            return translation
            
        except Exception as e:
            logger.error(f"Error during translation: {str(e)}")
            raise e

# Global model instance
translation_model = TranslationModel()
