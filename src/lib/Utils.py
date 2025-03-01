from lib.InferenceType import InferenceType
from lib.KnowledgeLevel import KnowledgeLevel


class Utils:

    def __init__(self):
        pass

    def restore_capital_letters(self, model_input, fa_attributions):

        # This is a helper function that restores the capital letters
        # in the tuples of word attributions. The capital letters were lost
        # during the feature ablation process.

        words = model_input.split()
        restored_attributions = []

        for word, attribution in fa_attributions:
            for original_word in words:
                if original_word.lower() == word:
                    restored_attributions.append((original_word, attribution))
                    break
            else:
                restored_attributions.append((word, attribution))

        return restored_attributions

    def is_valid_knowledge_level(self, knowledge_level):
        return knowledge_level in KnowledgeLevel

    def is_invalid_knowledge_level(self, knowledge_level):
        return knowledge_level not in KnowledgeLevel

    def is_valid_inference_type(self, inference_type):
        return inference_type in InferenceType

    def is_invalid_inference_type(self, inference_type):
        return inference_type not in InferenceType
