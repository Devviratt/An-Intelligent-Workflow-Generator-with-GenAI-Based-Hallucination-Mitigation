"""Tests for the Instruction Parser."""

from __future__ import annotations

from src.engines.instruction_parser import InstructionParser
from src.models.domain import DomainDataset


class TestInstructionParser:
    """Tests for NLP-based instruction parsing."""

    def test_payment_instruction(self, parser: InstructionParser) -> None:
        result = parser.parse("Create an online payment processing workflow")
        assert result.selected_domain == "online_payment"
        assert result.has_confident_match

    def test_registration_instruction(self, parser: InstructionParser) -> None:
        result = parser.parse("Build a user registration and signup flow")
        assert result.selected_domain == "user_registration"

    def test_order_instruction(self, parser: InstructionParser) -> None:
        result = parser.parse("Generate an order fulfillment shipping pipeline")
        assert result.selected_domain == "order_fulfillment"

    def test_incident_instruction(self, parser: InstructionParser) -> None:
        result = parser.parse("Design an incident response triage workflow")
        assert result.selected_domain == "incident_response"

    def test_data_pipeline_instruction(self, parser: InstructionParser) -> None:
        result = parser.parse("Create a data ETL pipeline with quality checks")
        assert result.selected_domain == "data_pipeline"

    def test_cicd_instruction(self, parser: InstructionParser) -> None:
        result = parser.parse("Build a CI/CD deployment pipeline with tests")
        assert result.selected_domain == "ci_cd_deployment"

    def test_ambiguous_instruction(self, parser: InstructionParser) -> None:
        result = parser.parse("do something")
        # Should still produce matches, even if low confidence
        assert len(result.domain_matches) > 0

    def test_keyword_extraction(self, parser: InstructionParser) -> None:
        result = parser.parse("Process credit card payment with fraud detection")
        assert len(result.keywords.cleaned_tokens) > 0

    def test_intent_flags(self, parser: InstructionParser) -> None:
        result = parser.parse("Create payment flow with retry and error handling")
        assert result.intent_flags.get("include_retry") is True
        assert result.intent_flags.get("include_error_handling") is True

    def test_minimal_intent(self, parser: InstructionParser) -> None:
        result = parser.parse("Create a simple minimal payment flow")
        assert result.intent_flags.get("minimal") is True

    def test_domain_matches_sorted_by_confidence(self, parser: InstructionParser) -> None:
        result = parser.parse("payment card transaction")
        confidences = [m.confidence for m in result.domain_matches]
        assert confidences == sorted(confidences, reverse=True)

    def test_cleaned_text_is_lowercase(self, parser: InstructionParser) -> None:
        result = parser.parse("Create A WORKFLOW For Payment")
        assert result.cleaned_text == result.cleaned_text.lower()
