"""Streamlit app for Presidio."""
import json
from json import JSONEncoder
import pandas as pd
import streamlit as st
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import spacy

nlp = spacy.load('/content/drive/MyDrive/DP/models/TrivialCPU')

import logging
from typing import Optional, List, Tuple, Set

from presidio_analyzer import (
    RecognizerResult,
    LocalRecognizer,
    AnalysisExplanation,
)

logger = logging.getLogger("presidio-analyzer")


class SpacyRecognizerCustom(LocalRecognizer):
    """
    Recognize PII entities using a spaCy NLP model.
    Since the spaCy pipeline is ran by the AnalyzerEngine,
    this recognizer only extracts the entities from the NlpArtifacts
    and replaces their types to align with Presidio's.
    :param supported_language: Language this recognizer supports
    :param supported_entities: The entities this recognizer can detect
    :param ner_strength: Default confidence for NER prediction
    :param check_label_groups: Tuple containing Presidio entity names
    and spaCy entity names, for verifying that the right entity
    is translated into a Presidio entity.
    """

    ENTITIES = [
        "PERSON",
        "EMAIL_ADDRESS",
        "LOGIN_NICK",
        "iNSTITUTION",
        "PHONE_NUM",
        "MEDIA_NAME",
        "DOMAIN",
        "NUMBER_EXPR",
        "LOCATION",
        "PRODUCT",
        "DATE_TIME",
        "OTHER"
    ]

    DEFAULT_EXPLANATION = "Identified as {} by Spacy's Named Entity Recognition"

  # NOTE: number expressionc moc nedavaji v nasem kontextu smysl
  #   v Artifact names je pro nas dulezity asi jen product
  #   NRP v nasem datasetu v tuto chvili nedokazeme identifikovat
  #   budu muset vytvorit novy model zalozeny na upravenem datasetu s fine-grained entitami a tohle potom upravit
  #   zaroven asi jeste budu muset opravit tabulku entit v prvni kapitole
    CHECK_LABEL_GROUPS = [
        ({"PERSON"}, {"pd", "pf", "pm", "ps"}),
        ({"EMAIL_ADDRESS"}, {"me"}),
        ({"LOGIN_NICK"}, {"p_"}),
        ({"iNSTITUTION"}, {"ia", "ic", "if", "io", "i_"}),
        ({"PHONE_NUM"}, {"at"}),
        ({"MEDIA_NAME"}, {"mn", "ms"}),
        ({"DOMAIN"}, {"mi"}),
        ({"LOCATION"}, {"ah", "az", "gc", "gh", "gl", "gq", "gr", "gs", "gt", "gu", "g_"}),
        ({"PRODUCT"}, {"op"}),
        ({"DATE_TIME"}, {"td", "tf", "th", "tm", "ty"}),
        ({"OTHER"}, {"oa", "or", "o_", "pc"})
    ]

    def __init__(
        self,
        supported_language: str = "cs",
        supported_entities: Optional[List[str]] = None,
        # TODO doplnit dle aktualni sily modelu
        ner_strength: float = 0.82,
        check_label_groups: Optional[Tuple[Set, Set]] = None,
        context: Optional[List[str]] = None,
    ):
        self.ner_strength = ner_strength
        self.check_label_groups = (
            check_label_groups if check_label_groups else self.CHECK_LABEL_GROUPS
        )
        supported_entities = supported_entities if supported_entities else self.ENTITIES
        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            context=context,
        )

    def load(self) -> None:  # noqa D102
        # no need to load anything as the analyze method already receives
        # preprocessed nlp artifacts
        pass

    def build_spacy_explanation(
        self, original_score: float, explanation: str
    ) -> AnalysisExplanation:
        """
        Create explanation for why this result was detected.
        :param original_score: Score given by this recognizer
        :param explanation: Explanation string
        :return:
        """
        explanation = AnalysisExplanation(
            recognizer=self.__class__.__name__,
            original_score=original_score,
            textual_explanation=explanation,
        )
        return explanation

    def analyze(self, text, entities, nlp_artifacts=None):  # noqa D102
        results = []
        if not nlp_artifacts:
            logger.warning("Skipping SpaCy, nlp artifacts not provided...")
            return results

        ner_entities = nlp_artifacts.entities

        for entity in entities:
            if entity not in self.supported_entities:
                continue
            for ent in ner_entities:
                if not self.__check_label(entity, ent.label_, self.check_label_groups):
                    continue
                textual_explanation = self.DEFAULT_EXPLANATION.format(ent.label_)
                explanation = self.build_spacy_explanation(
                    self.ner_strength, textual_explanation
                )
                spacy_result = RecognizerResult(
                    entity_type=entity,
                    start=ent.start_char,
                    end=ent.end_char,
                    score=self.ner_strength,
                    analysis_explanation=explanation,
                    recognition_metadata={
                        RecognizerResult.RECOGNIZER_NAME_KEY: self.name
                    },
                )
                results.append(spacy_result)

        return results

    @staticmethod
    def __check_label(
        entity: str, label: str, check_label_groups: Tuple[Set, Set]
    ) -> bool:
        return any(
            [entity in egrp and label in lgrp for egrp, lgrp in check_label_groups]
        )

# Create custom recognizer based on NER model NEs
spacy_recognizer_custom = SpacyRecognizerCustom()

# "rodne cislo" custom recognizer (derived from https://gist.github.com/xnekv03/7d684df577a483d8b7734dafb8291e3d#file-verifyrc-php podle https://phpfashion.com/jak-overit-platne-ic-a-rodne-cislo)
# TODO
  # fix multiple overlaping results
  # add additional validation checks

from collections import defaultdict
from typing import List, Optional

from presidio_analyzer import Pattern, PatternRecognizer


class CSRCRecognizer(PatternRecognizer):
    """Recognize CS "rodne cislo" using regex.
    :param patterns: List of patterns to be used by this recognizer
    :param context: List of context words to increase confidence in detection
    :param supported_language: Language this recognizer supports
    :param supported_entity: The entity this recognizer can detect
    """

    PATTERNS = [
        Pattern("rodne cislo (high)", r"\d{2}(0[1-9]|1[0-2]|5[1-9]|6[0-2])(0[1-9]|1[0-9]|2[0-9]|3[0-1])\/?\d{3,4}", 0.5)
    ]

    CONTEXT = [
        "rc",
        "rodne",
        "pojistence"
        "cislo"
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "cs",
        supported_entity: str = "CS_RC",
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language,
        )

rc_recognizer = CSRCRecognizer()

from presidio_analyzer.predefined_recognizers import CreditCardRecognizer, CryptoRecognizer, EmailRecognizer, IbanRecognizer, IpRecognizer, PhoneRecognizer

credit_card_recognizer = CreditCardRecognizer(supported_language="cs", context=["kreditni", "debetni", "karta", "visa", "mastercard", "maestro", "platba"])
crypto_recognizer = CryptoRecognizer(supported_language="cs", context=["wallet", "btc", "bitcoin", "ethereum", "eth", "crypto", "kryptomena"])
email_recognizer = EmailRecognizer(supported_language="cs", context=["email", "mail", "e-mail"])
iban_recognizer = IbanRecognizer(supported_language="cs", context=["iban", "banka", "swift", "zahranicni", "transakce", "platba"])
ip_recognizer = IpRecognizer(supported_language="cs")
# phone recognizer musi byt upraven pro CZ pouziti
#phone_recognizer = PhoneRecognizer()

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider


# Create configuration containing engine name and models
configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "cs", "model_name": "cs_TrivialCPUFineGrained"}],
}

# Create new recognizer registry and add the custom recognizer
recognizer_registry = RecognizerRegistry()

# append custom recognizers
recognizer_registry.add_recognizer(spacy_recognizer_custom)
recognizer_registry.add_recognizer(rc_recognizer)

# append predefined universal presidio recognizers
recognizer_registry.add_recognizer(credit_card_recognizer)
recognizer_registry.add_recognizer(crypto_recognizer)
recognizer_registry.add_recognizer(email_recognizer)
recognizer_registry.add_recognizer(iban_recognizer)
recognizer_registry.add_recognizer(ip_recognizer)
#recognizer_registry.add_recognizer(phone_recognizer)

# Create NLP engine based on configuration
provider = NlpEngineProvider(nlp_configuration=configuration)
nlp_engine_custom = provider.create_engine()

# Pass the created NLP engine and supported_languages to the AnalyzerEngine
analyzer = AnalyzerEngine(
    nlp_engine=nlp_engine_custom, 
    supported_languages=["cs"],
    registry=recognizer_registry
)

# Helper methods
def analyzer_engine():
    """Return AnalyzerEngine."""
    return analyzer
    #return AnalyzerEngine()
@st.cache(allow_output_mutation=True)
def anonymizer_engine():
    """Return AnonymizerEngine."""
    return AnonymizerEngine()
def get_supported_entities():
    """Return supported entities from the Analyzer Engine."""
    return analyzer_engine().get_supported_entities()
def analyze(**kwargs):
    """Analyze input using Analyzer engine and input arguments (kwargs)."""
    if "entities" not in kwargs or "All" in kwargs["entities"]:
        kwargs["entities"] = None
    return analyzer_engine().analyze(**kwargs)
def anonymize(text, analyze_results):
    """Anonymize identified input using Presidio Abonymizer."""

    res = anonymizer_engine().anonymize(text, analyze_results)
    return res.text
st.set_page_config(page_title="Presidio demo", layout="wide")
# Side bar
st_entities = st.sidebar.multiselect(
    label="Výběr jmenných identifikátorů",
    options=get_supported_entities(),
    default=list(get_supported_entities()),
)
st_threhsold = st.sidebar.slider(
    label="Acceptance threshold", min_value=0.0, max_value=1.0, value=0.35
)
st_return_decision_process = st.sidebar.checkbox("Add analysis explanations in json")

# Main panel
analyzer_load_state = st.info("Starting Presidio analyzer...")
engine = analyzer_engine()
analyzer_load_state.empty()
# Create two columns for before and after
col1, col2 = st.columns(2)
# Before:
col1.subheader("Input string:")
st_text = col1.text_area(
    label="Enter text",
    value="Kontaktní telefonní číslo "
    "společnosti StavMat s.r.o. je +420 500 210 596. Zodpovědnou osobou je Jan Lakatoš.",
    height=400,
)
# After
col2.subheader("Output:")
st_analyze_results = analyze(
    text=st_text,
    entities=st_entities,
    language="cs",
    score_threshold=st_threhsold,
    return_decision_process=st_return_decision_process,
)
st_anonymize_results = anonymize(st_text, st_analyze_results)
col2.text_area(label="", value=st_anonymize_results, height=400)
# table result
st.subheader("Findings")
if st_analyze_results:
    df = pd.DataFrame.from_records([r.to_dict() for r in st_analyze_results])
    df = df[["entity_type", "start", "end", "score"]].rename(
        {
            "entity_type": "Entity type",
            "start": "Start",
            "end": "End",
            "score": "Confidence",
        },
        axis=1,
    )

    st.dataframe(df, width=1000)
else:
    st.text("No findings")
# json result
class ToDictEncoder(JSONEncoder):
    """Encode dict to json."""

    def default(self, o):
        """Encode to JSON using to_dict."""
        return o.to_dict()
st.json(json.dumps(st_analyze_results, cls=ToDictEncoder))