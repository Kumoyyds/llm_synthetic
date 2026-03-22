"""
Concept Rephrasing Module using LLM calls.

This module provides flexible rephrasing capabilities with controls for:
- Length (short ↔ standard)
- Tone/Style (Clinical, Marketing, Conversational)
- Point of View (Second Person, Third Person)
- Content Order (Problem-First, Feature-First, Benefit-First)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Literal
import re
import dotenv
dotenv.load_dotenv()
import os
api_key = os.getenv("LITE_LLM_KEY_ALL")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class Tone(str, Enum):
    """Tone/style options for rephrasing."""
    CLINICAL = "clinical"
    MARKETING = "marketing"
    CONVERSATIONAL = "conversational"
    
    @property
    def description(self) -> str:
        descriptions = {
            Tone.CLINICAL: (
                "Clinical/Objective tone: Neutral, factual, and dry. "
                "Example: 'This device heats to 180°C in 10 seconds.' "
                "Remove emotional bias, focus on facts and specifications."
            ),
            Tone.MARKETING: (
                "Marketing/Hype tone: Uses buzzwords and excitement. "
                "Example: 'Experience the revolutionary new way to...' "
                "Emphasize benefits, use power words, create urgency and desire."
            ),
            Tone.CONVERSATIONAL: (
                "Conversational/Peer tone: Casual and relatable. "
                "Example: 'It's basically a smart-mug that actually keeps your coffee hot all morning.' "
                "Use everyday language, be friendly and approachable."
            ),
        }
        return descriptions[self]


class PointOfView(str, Enum):
    """Point of view options."""
    SECOND_PERSON = "second_person"
    THIRD_PERSON = "third_person"
    
    @property
    def description(self) -> str:
        descriptions = {
            PointOfView.SECOND_PERSON: (
                "Second Person: Address the reader directly using 'you', 'your', 'yours'. "
                "Example: 'You will experience...', 'Your morning routine will change...'"
            ),
            PointOfView.THIRD_PERSON: (
                "Third Person: Refer to users/consumers in third person using 'they', 'users', 'consumers', 'customers'. "
                "Example: 'Users will experience...', 'Consumers can expect...'"
            ),
        }
        return descriptions[self]


class ContentOrder(str, Enum):
    """Content ordering strategy."""
    PROBLEM_FIRST = "problem_first"
    FEATURE_FIRST = "feature_first"
    BENEFIT_FIRST = "benefit_first"
    
    @property
    def description(self) -> str:
        descriptions = {
            ContentOrder.PROBLEM_FIRST: (
                "Problem-First: Start with the pain point. "
                "Build empathy by describing the problem or frustration first, then present the solution. "
    
            ),
            ContentOrder.FEATURE_FIRST: (
                "Feature-First: Start with the novel technology or ingredient. "
                "Lead with what makes this unique - the innovation, the technology, the special ingredient. "
            ),
            ContentOrder.BENEFIT_FIRST: (
                "Benefit-First: Start with the ultimate outcome or feeling. "
                "Lead with the end result - how the consumer will feel or what they will achieve. "
            ),
        }
        return descriptions[self]


class LengthCategory(str, Enum):
    """Length categories for content."""
    SHORT = "short"      # < 100 words
    STANDARD = "standard"  # 100-170 words


@dataclass
class RephraseConfig:
    """Configuration for rephrasing operations."""
    change_length: bool = False
    tone: Optional[Tone] = None
    point_of_view: Optional[PointOfView] = None
    content_order: Optional[ContentOrder] = None
    
    # Length thresholds
    short_max_words: int = 99
    standard_min_words: int = 100
    standard_max_words: int = 170


@dataclass
class RephraseResult:
    """Result of a rephrasing operation."""
    original_text: str
    rephrased_text: str
    original_word_count: int
    new_word_count: int
    original_length_category: LengthCategory
    target_length_category: Optional[LengthCategory]
    config_applied: RephraseConfig


class ConceptRephraser:
    """
    A class that uses LLM to rephrase concepts with various transformations.
    
    Example:
        >>> rephraser = ConceptRephraser(model="gpt-4o-mini")
        >>> config = RephraseConfig(
        ...     change_length=True,
        ...     tone=Tone.CONVERSATIONAL,
        ...     point_of_view=PointOfView.SECOND_PERSON,
        ...     content_order=ContentOrder.BENEFIT_FIRST
        ... )
        >>> result = rephraser.rephrase("Your original concept text here...", config)
        >>> print(result.rephrased_text)
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        keep_brand_names: bool = False,
        keep_price_information: bool = False
    ):
        """
        Initialize the rephraser with an LLM model.
        
        Args:
            model: The OpenAI model to use (default: gpt-4o-mini for efficiency)
            temperature: Creativity level (0-1, default: 0.7)
            keep_brand_names: Whether to preserve brand names (default: False)
            keep_price_information: Whether to preserve price info (default: False)
        """
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url="https://ipsos.litellm-prod.ai/"
        )
        self.output_parser = StrOutputParser()
        self.keep_brand_names = keep_brand_names
        self.keep_price_information = keep_price_information
        
    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def _detect_length_category(self, text: str, config: RephraseConfig) -> LengthCategory:
        """Detect the length category of the text."""
        word_count = self._count_words(text)
        if word_count <= config.short_max_words:
            return LengthCategory.SHORT
        return LengthCategory.STANDARD
    
    def _get_target_length_category(
        self, 
        current_category: LengthCategory
    ) -> LengthCategory:
        """Get the opposite length category."""
        if current_category == LengthCategory.SHORT:
            return LengthCategory.STANDARD
        return LengthCategory.SHORT
    
    def _build_prompt(self, config: RephraseConfig, current_length: LengthCategory) -> ChatPromptTemplate:
        """Build the prompt template based on configuration."""
        
        instructions = []
        
        # Length instruction
        if config.change_length:
            target_length = self._get_target_length_category(current_length)
            if target_length == LengthCategory.SHORT:
                instructions.append(
                    f"LENGTH: Condense the text to be SHORT (less than {config.short_max_words} words). "
                    "Be concise while preserving key information."
                )
            else:
                instructions.append(
                    f"LENGTH: Expand the text to be STANDARD ({config.standard_min_words}-{config.standard_max_words} words). "
                    "Add relevant details and elaboration while maintaining coherence."
                )
        
        # Tone instruction
        if config.tone:
            instructions.append(f"TONE: {config.tone.description}")
        
        # POV instruction
        if config.point_of_view:
            instructions.append(f"POINT OF VIEW: {config.point_of_view.description}")
        
        # Content order instruction
        if config.content_order:
            instructions.append(f"CONTENT ORDER: {config.content_order.description}")
        
        # Build the full prompt
        if not instructions:
            instructions.append("Rephrase the text while maintaining its original meaning, length, and style.")
        
        instructions_text = "\n\n".join(instructions)

        price_insert = "erase the price information, never include any price information" if not self.keep_price_information else ""
        brand_insert = "erase the brand names, never include any brand names" if not self.keep_brand_names else ""
        
        system_prompt = """You are an expert copywriter and content strategist. Your task is to rephrase the given concept text according to the specific instructions provided.

INSTRUCTIONS:
{instructions}

CRITICAL RULES - PRESERVE KEY INFORMATION:
1. NEVER remove, alter, or omit any of these from the original:
   - Product/concept name
   - Key features and specifications (numbers, measurements, technical details)
   - Core benefits and claims
   - Unique selling points and differentiators
   - Ingredients, materials, or technologies mentioned
2. Do NOT invent new features, claims, or benefits not present in the original.
3. Do NOT exaggerate or downplay any claims beyond what the original states.
4. All factual statements must remain accurate and traceable to the original.
5. Output ONLY the rephrased text, no explanations or meta-commentary.
6. Ensure the text flows naturally and reads well while strictly preserving core content.
7. {price_insert}
8. {brand_insert}
"""

        human_prompt = """Original text to rephrase:

{text}

Rephrased text:"""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt),
        ]).partial(instructions=instructions_text, price_insert=price_insert, brand_insert=brand_insert)
    
    def rephrase(self, text: str, config: Optional[RephraseConfig] = None) -> RephraseResult:
        """
        Rephrase the given text according to the configuration.
        
        Args:
            text: The original concept text to rephrase
            config: Configuration for the rephrasing operation
            
        Returns:
            RephraseResult containing the original and rephrased text with metadata
        """
        if config is None:
            config = RephraseConfig()
        
        original_word_count = self._count_words(text)
        original_length_category = self._detect_length_category(text, config)
        
        target_length_category = None
        if config.change_length:
            target_length_category = self._get_target_length_category(original_length_category)
        
        # Build and execute the chain
        prompt = self._build_prompt(config, original_length_category)
        chain = prompt | self.llm | self.output_parser
        
        rephrased_text = chain.invoke({"text": text})
        
        # Clean up the response
        rephrased_text = rephrased_text.strip()
        
        return RephraseResult(
            original_text=text,
            rephrased_text=rephrased_text,
            original_word_count=original_word_count,
            new_word_count=self._count_words(rephrased_text),
            original_length_category=original_length_category,
            target_length_category=target_length_category,
            config_applied=config,
        )
    
    async def arephrase(self, text: str, config: Optional[RephraseConfig] = None) -> RephraseResult:
        """
        Async version of rephrase for batch processing.
        
        Args:
            text: The original concept text to rephrase
            config: Configuration for the rephrasing operation
            
        Returns:
            RephraseResult containing the original and rephrased text with metadata
        """
        if config is None:
            config = RephraseConfig()
        
        original_word_count = self._count_words(text)
        original_length_category = self._detect_length_category(text, config)
        
        target_length_category = None
        if config.change_length:
            target_length_category = self._get_target_length_category(original_length_category)
        
        # Build and execute the chain
        prompt = self._build_prompt(config, original_length_category)
        chain = prompt | self.llm | self.output_parser
        
        rephrased_text = await chain.ainvoke({"text": text})
        
        # Clean up the response
        rephrased_text = rephrased_text.strip()
        
        return RephraseResult(
            original_text=text,
            rephrased_text=rephrased_text,
            original_word_count=original_word_count,
            new_word_count=self._count_words(rephrased_text),
            original_length_category=original_length_category,
            target_length_category=target_length_category,
            config_applied=config,
        )
    
    def batch_rephrase(
        self, 
        texts: list[str], 
        config: Optional[RephraseConfig] = None
    ) -> list[RephraseResult]:
        """
        Rephrase multiple texts with the same configuration.
        
        Args:
            texts: List of texts to rephrase
            config: Configuration for the rephrasing operation
            
        Returns:
            List of RephraseResult objects
        """
        return [self.rephrase(text, config) for text in texts]


# Convenience function for quick rephrasing
def rephrase_concept(
    text: str,
    change_length: bool = False,
    tone: Optional[Literal["clinical", "marketing", "conversational"]] = None,
    point_of_view: Optional[Literal["second_person", "third_person"]] = None,
    content_order: Optional[Literal["problem_first", "feature_first", "benefit_first"]] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> str:
    """
    Convenience function to quickly rephrase a concept.
    
    Args:
        text: The concept text to rephrase
        change_length: Whether to toggle between short/standard length
        tone: 'clinical', 'marketing', or 'conversational'
        point_of_view: 'second_person' or 'third_person'
        content_order: 'problem_first', 'feature_first', or 'benefit_first'
        model: OpenAI model to use
        temperature: Creativity level (0-1)
        
    Returns:
        The rephrased text
        
    Example:
        >>> rephrased = rephrase_concept(
        ...     "Our new blender uses advanced motor technology...",
        ...     tone="conversational",
        ...     point_of_view="second_person",
        ...     content_order="benefit_first"
        ... )
    """
    config = RephraseConfig(
        change_length=change_length,
        tone=Tone(tone) if tone else None,
        point_of_view=PointOfView(point_of_view) if point_of_view else None,
        content_order=ContentOrder(content_order) if content_order else None,
    )
    
    rephraser = ConceptRephraser(model=model, temperature=temperature)
    result = rephraser.rephrase(text, config)
    return result.rephrased_text


# Example usage and testing
if __name__ == "__main__":
    # Example concept text
    sample_text = """
    Smart coffee mugs have revolutionized the way people enjoy their morning beverages. 
    The Ember Mug 2 features advanced temperature control technology that maintains your 
    drink at your preferred temperature for up to 1.5 hours on a single charge. The device 
    connects to a smartphone app, allowing users to set custom temperatures and receive 
    notifications when their drink reaches the perfect heat. Made with premium stainless 
    steel and a scratch-resistant coating, this mug combines functionality with elegant design.
    The built-in battery is rechargeable via an included coaster, making it convenient for 
    daily use at home or in the office.
    """
    
    print("=" * 60)
    print("ORIGINAL TEXT")
    print("=" * 60)
    print(sample_text.strip())
    print(f"\nWord count: {len(sample_text.split())}")
    
    # Example 1: Change to conversational tone with second person POV
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Conversational + Second Person + Benefit-First")
    print("=" * 60)
    
    rephraser = ConceptRephraser()
    config = RephraseConfig(
        change_length=False,
        tone=Tone.CONVERSATIONAL,
        point_of_view=PointOfView.SECOND_PERSON,
        content_order=ContentOrder.BENEFIT_FIRST,
    )
    
    result = rephraser.rephrase(sample_text, config)
    print(result.rephrased_text)
    print(f"\nNew word count: {result.new_word_count}")
    
    # Example 2: Shorten with clinical tone
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Shortened + Clinical Tone")
    print("=" * 60)
    
    config2 = RephraseConfig(
        change_length=True,  # Will detect standard and convert to short
        tone=Tone.CLINICAL,
    )
    
    result2 = rephraser.rephrase(sample_text, config2)
    print(result2.rephrased_text)
    print(f"\nOriginal: {result2.original_length_category.value} ({result2.original_word_count} words)")
    print(f"Target: {result2.target_length_category.value}")
    print(f"New: {result2.new_word_count} words")
