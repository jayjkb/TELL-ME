import torch
from captum.attr import FeatureAblation, LLMAttribution, TextTokenInput

from lib.ChatGPT import ChatGPT
from lib.InferenceType import InferenceType
from lib.Prompts import XAIPrompt

# Settings used for running models on windows without nvidia gpu
torch.set_default_device("cpu")


class XAI:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.chatgpt = ChatGPT()

    def __forward_wrapper(self, input_ids, attention_mask=None):
        # Wrapper to extract logits from the model output (for Feature Ablation)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits

    def __remove_placeholder(self, text_tokens, placeholder):

        return [word.replace(placeholder, "").strip() for word in text_tokens]

    def run_feature_ablation_for_sentiment_analysis(self, inputs, sentiment_id):

        # ---------- Feature Ablation for Sentiment Analysis ----------

        # The forward wrapper is needed to extract the logits from the model's output,
        # ensuring compatibility with the FeatureAblation method from Captum,
        # which requires a specific output format.

        ablator = FeatureAblation(self.__forward_wrapper)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        attributions = ablator.attribute(
            (input_ids, attention_mask), target=sentiment_id
        )
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        attr_scores = attributions[0].detach().numpy()[0]
        token_attr_pairs = list(zip(tokens, attr_scores))

        return token_attr_pairs

    def run_feature_ablation_for_text_generation(self, input_text, output_text):

        # --------- Feature Ablation for Text Generation ----------

        # The Captum library could be used out of the box. Additionally, I removed the
        # placeholder "Ġ" from the tokens. This placeholder was introduced by the models
        # tokenizer to better handle spaces.

        input_text = input_text.strip()
        output_text = output_text.strip()

        fa = FeatureAblation(self.model)
        llm_attr = LLMAttribution(fa, self.tokenizer)
        inp = TextTokenInput(input_text, self.tokenizer)
        attributions = llm_attr.attribute(
            inp, target=torch.tensor(self.tokenizer.encode(output_text))
        ).token_attr

        input_text_tokens = self.__remove_placeholder(
            self.tokenizer.tokenize(input_text), "Ġ"
        )
        output_text_tokens = self.__remove_placeholder(
            self.tokenizer.tokenize(output_text), "Ġ"
        )

        return [input_text_tokens, output_text_tokens, attributions]

    def run_counterfactual_for_sentiment_analysis(self, sentiment, text):

        # ---------- Counterfactuals for Sentiment Analysis ----------

        # Counterfactuals are generated in a three step process
        # Based on the framework proposed in https://arxiv.org/pdf/2309.13340

        manager = XAIPrompt(InferenceType.SentimentAnalysis, "counterfactual")

        prompt1 = manager.get_step(1)
        prompt1 = prompt1.replace("{sentiment}", sentiment)
        prompt1 = prompt1.replace("{text}", text)
        messages = [
            {"role": "system", "content": prompt1},
        ]
        latent_features = self.chatgpt.call(messages)
        latent_features = latent_features.replace(".", "").strip()
        messages.append({"role": "assistant", "content": latent_features})

        prompt2 = manager.get_step(2)
        prompt2 = prompt2.replace("{latent-features}", latent_features)
        messages.append({"role": "user", "content": prompt2})
        identified_words = self.chatgpt.call(messages)
        messages.append({"role": "assistant", "content": identified_words})

        prompt3 = manager.get_step(3)
        prompt3 = prompt3.replace("{identified-words}", identified_words)
        messages.append({"role": "user", "content": prompt3})
        counterfactual = self.chatgpt.call(messages)

        return [latent_features, identified_words, counterfactual]

    def run_counterfactual_for_text_generation(self, input_text, output_text):

        # ---------- Counterfactuals for Text Generation ----------

        # Counterfactuals are generated in a two step process similar to
        # the counterfactuals for sentiment analysis.

        manager = XAIPrompt(InferenceType.TextGeneration, "counterfactual")

        prompt1 = manager.get_step(1)
        prompt1 = prompt1.replace("{input-text}", input_text)
        prompt1 = prompt1.replace("{output-text}", output_text)
        messages = [
            {"role": "system", "content": prompt1},
        ]
        key_components = self.chatgpt.call(messages)
        key_components_list = key_components.split(",")
        messages.append({"role": "assistant", "content": key_components})

        prompt2 = manager.get_step(2)
        messages.append({"role": "user", "content": prompt2})
        alternative_versions = self.chatgpt.call(messages)
        alternative_versions = alternative_versions.split(";")

        return [key_components_list, alternative_versions]
