import base64
import json
import time
from difflib import Differ

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from annotated_text import util as annotated_text_utils
from streamlit_extras.row import row as create_row
from streamlit_extras.tags import _get_html as tagger_component
from streamlit_extras.word_importances import format_word_importances
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from lib.ChatGPT import ChatGPT
from lib.InferenceType import InferenceType
from lib.KnowledgeLevel import KnowledgeLevel
from lib.Prompts import ChatPrompt, XAIPrompt
from lib.Utils import Utils
from lib.XAI import XAI

torch.set_default_device("cpu")

# Streamlit UI
st.set_page_config(page_title="TELL-ME", layout="wide")
st.title("TELL-ME")
st.markdown(
    '<small>Tailored Explanation of Large Language ModEls.<br>To get started, please choose the inference type and base model you would like to use for predictions using the selections in the left sidebar. Next, configure your settings in the "Profile" tab. Once your profile is set up, navigate to the "Home" tab to make your first prediction and receive a personalized explanation!</small>',
    unsafe_allow_html=True,
)

# Constants
SA_MODELS = [
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
]  # Models for Sentiment Analysis
TG_MODELS = ["openai-community/gpt2", "microsoft/phi-2"]  # Models for Text Generation


def reset_cache() -> None:
    """Reset the session state cache. Exclude certain values & user settings."""

    excluded_values = ["inference_type", "inference_type_string", "model_name"]
    keys = list(st.session_state.keys())
    for key in keys:
        if "user_" in key or key in excluded_values:  # Skip excluded settings
            continue
        st.session_state.pop(key)
    st.session_state.give_explanation = False


# Sidebar for model settings
with st.sidebar:

    st.radio(
        "Choose the Inference Type",
        ["Sentiment Analysis", "Text Generation"],
        captions=[
            "Use an LLM to detect if a text is positive or negative.",
            "Use an LLM to generating new text based on your input text.",
        ],
        key="inference_type_string",
        on_change=reset_cache,
        help="Please choose what you would like to do with a language model.",
    )

    inference_type = InferenceType[
        st.session_state.inference_type_string.replace(" ", "")
    ]
    st.session_state.inference_type = inference_type

    if inference_type == InferenceType.SentimentAnalysis:
        model_selection = SA_MODELS
    elif inference_type == InferenceType.TextGeneration:
        model_selection = TG_MODELS
    else:
        model_selection = []

    model_name = st.sidebar.selectbox(
        "Choose the Base Model",
        model_selection,
        key="model_name",
        on_change=reset_cache,
        help="Please choose the language model you would like to use.",
    )


@st.cache_resource
def load_model(model_name):
    """Load the selected model. Cache to avoid reloads on reruns."""

    if model_name == "distilbert/distilbert-base-uncased-finetuned-sst-2-english":
        return DistilBertForSequenceClassification.from_pretrained(
            model_name
        ), DistilBertTokenizer.from_pretrained(model_name)
    if model_name == "openai-community/gpt2":
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        return model, tokenizer
    if model_name == "microsoft/phi-2":
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        return model, tokenizer


model, tokenizer = load_model(model_name)


utils = Utils()
chatgpt = ChatGPT()
xai = XAI(model, tokenizer)


def run_model(model_input: str, save_prediction: bool = True) -> str:
    """Run the model with the given input and make a prediction."""

    model_input_tokens = tokenizer(model_input, return_tensors="pt")

    if model_name in SA_MODELS:
        # Steps for Sentiment Analysis

        with torch.no_grad():
            logits = model(**model_input_tokens).logits
        model_output = logits.argmax().item()
        sentiment_label = model.config.id2label[model_output]

        if save_prediction:
            st.session_state.sentiment_id = model_output
            st.session_state.sentiment_label = sentiment_label
            st.session_state.sentiment_opposite_label = (
                "POSITIVE" if sentiment_label == "NEGATIVE" else "NEGATIVE"
            )
            st.session_state.sentiment_emoji = "😄" if model_output == 1 else "😕"

    elif model_name in TG_MODELS:
        # Steps for Text Generation

        with torch.no_grad():
            input_ids = model_input_tokens["input_ids"]
            output_ids = model.generate(input_ids, max_new_tokens=5)[0]
            model_output = tokenizer.decode(
                output_ids[len(input_ids[0]) :], skip_special_tokens=True
            )

    else:

        raise ValueError(f"Unknown model name: {model_name}")

    if save_prediction:
        st.session_state.model_input = model_input
        st.session_state.model_input_parts = model_input.split()
        st.session_state.model_input_tokens = model_input_tokens
        st.session_state.model_output = model_output

    return model_output


def give_explanation() -> bool:
    return (
        st.session_state.user_personalization_enabled
        and "give_explanation" in st.session_state
        and st.session_state.give_explanation
    )


def is_counterfactual(input: str, original_output: str) -> bool:
    """Check if the counterfactual is actually a counterfactual."""
    return original_output != run_model(input, False)


def input_changed(input: str) -> bool:
    """Check if the user input changed since the last model run."""
    return "model_input" in st.session_state and st.session_state.model_input != input


def valid_profile_settings(error_placeholder) -> bool:
    """Validate user profile settings."""

    if st.session_state.get("user_checkbox_textual") and not st.session_state.get(
        "user_firstname"
    ):
        error_placeholder.error(
            "Please provide your firstname in the profile settings or disable textual explanations."
        )
        return False
    return True


def display_sentiment_analysis() -> None:

    with st.container(border=True):

        st.write("#### Sentiment Analysis")
        st.markdown(
            "<small>Sentiment analysis is the process of determining the emotional tone or attitude expressed in a piece of text—in our case, whether the text is positive or negative.</small>",
            unsafe_allow_html=True,
        )

        examples = [
            "I absolutely love this product! Exceeded my expectations.",
            "Service was fantastic. Staff were friendly and helpful.",
            "Impressed with the quality. Well-made and works well.",
            "Very disappointed. The product broke after one use.",
            "Terrible service. Staff were rude and I had to wait.",
            "Quality of this item is poor. Feels cheap and faulty.",
        ]

        example_options = (
            ["Use my own input ..."]
            + [f"Use Positive Example {i+1}" for i in range(3)]
            + [f"Use Negative Example {i-2}" for i in range(3, len(examples))]
        )
        example_choice = st.selectbox(
            "Choose an example text to analyze or input your own:",
            example_options,
            on_change=reset_cache,
        )

        if example_choice != "Use my own input ...":
            example_index = example_options.index(example_choice) - 1
            user_input = st.text_input(
                "Please input text you would like to analyze:",
                value=examples[example_index],
                max_chars=60,
            )
        else:
            user_input = st.text_input(
                "Please input text you would like to analyze:",
                max_chars=60,
                key="user_input",
                value=(
                    st.session_state.user_input
                    if "user_input" in st.session_state
                    else ""
                ),
            )

        error_placeholder = st.empty()

        if st.button("Analyze sentiment") and valid_profile_settings(error_placeholder):

            if len(user_input) <= 10:
                error_placeholder.error(
                    "Please add more text (at least 10 characters)."
                )
                return

            with st.spinner(text="Analyzing sentiment ..."):

                # Remove old prediction from state
                if input_changed(user_input):
                    reset_cache()

                # Simulate longer processing time for smoother user experience
                time.sleep(1)

                # Make new prediction
                run_model(user_input)

                # Output prediction
                with st.container(
                    border=True,
                ):
                    st.write(
                        "Sentiment prediction: "
                        + st.session_state.sentiment_label
                        + st.session_state.sentiment_emoji
                    )

            st.session_state.give_explanation = True


def display_text_generation() -> None:

    with st.container(border=True):

        st.write("#### Text Generation")
        st.markdown(
            "<small>Text generation is the process of creating human-like text based on a given input or prompt.</small>",
            unsafe_allow_html=True,
        )

        examples = [
            "My name is Julien and I like to",
            "My name is Thomas and my main",
            "My name is Teven and I am",
        ]

        example_options = ["Use my own input ..."] + [
            f" Use Example {i+1}" for i in range(3)
        ]
        example_choice = st.selectbox(
            "Choose an example starter text or input your own:",
            example_options,
            on_change=reset_cache,
        )

        if example_choice != "Use my own input ...":
            example_index = example_options.index(example_choice) - 1
            user_input = st.text_input(
                "Please input a text:",
                value=examples[example_index],
                max_chars=60,
            )
        else:
            user_input = st.text_input(
                "Please input a text:",
                max_chars=60,
                key="user_input",
                value=(
                    st.session_state.user_input
                    if "user_input" in st.session_state
                    else ""
                ),
            )

        error_placeholder = st.empty()

        if st.button("Generate") and valid_profile_settings(error_placeholder):

            if len(user_input) <= 10:
                error_placeholder.error(
                    "Please add more text (at least 10 characters)."
                )
                return

            with st.spinner(text="Generating text ..."):

                # Remove old prediction from state
                if input_changed(user_input):
                    reset_cache()

                # Simulate longer processing time for smoother user experience
                time.sleep(1)

                # Make new prediction
                generated_text = run_model(user_input)

                with st.container(border=True):
                    st.markdown(
                        user_input
                        + "<span style='color: blue;'>"
                        + generated_text
                        + "</span>",
                        unsafe_allow_html=True,
                    )

            st.session_state.give_explanation = True


def display_feature_ablation_for_sentiment_analysis(
    model_input,
    _model_input_tokens,
    sentiment_id,
    sentiment_label,
    user_knowledge_level,
):

    fa_html = ""
    fa_chart = ""

    fa_attributions = xai.run_feature_ablation_for_sentiment_analysis(
        _model_input_tokens, sentiment_id
    )
    fa_attributions = fa_attributions[1:-1]
    fa_attributions = utils.restore_capital_letters(model_input, fa_attributions)

    match user_knowledge_level:

        case KnowledgeLevel.Beginner:

            # Explanations for 'Beginners' have the following characteristics:
            # 1. Highlight the top 3 most influencial words for the prediction
            # 2. Mark them green if the sentiment is positive, and red if the sentiment is negative

            # Filter out words that are the most influencial for the prediction
            positive_scores = [item for item in fa_attributions if item[1] > 0]
            top_3_highest_scores = sorted(
                positive_scores, key=lambda x: x[1], reverse=True
            )[:3]

            annotated_text_items = []
            for word, score in fa_attributions:
                if (word, score) in top_3_highest_scores:
                    color = "#faa" if (score > 0) and (sentiment_id == 0) else "#afa"
                    annotation = (word, "", color)
                    annotated_text_items.append(annotation)
                else:
                    annotated_text_items.append(f" {word} ")
            fa_html = annotated_text_utils.get_annotated_html(*annotated_text_items)
            st.markdown(fa_html, unsafe_allow_html=True)
            st.markdown(
                f"<small>The process of feature ablation has identified the top 3 words from your input that were most influential in predicting a '<i>{sentiment_label}</i>' sentiment.</small>",
                unsafe_allow_html=True,
            )

        case KnowledgeLevel.Advanced:

            # Explanations for people with 'Advanced' knowledge:
            # 1. Use gradient background colors for the text, based on the attribution scores. Coloring goes from dark red to dark green
            # 2. Do not display individual scores

            fa_html = format_word_importances(
                words=[pair[0] for pair in fa_attributions],
                importances=tuple([pair[1] for pair in fa_attributions]),
            )
            st.markdown(fa_html + "<br>", unsafe_allow_html=True)
            st.markdown(
                '<small>The process of feature ablation identified how each word in the sentence influences the predicted sentiment. Green words positively influence the prediction, while red words negatively influence it. The darker the colors, the greater the influence. Note that in this context, "positive" and "negative" refer to their impact on the prediction, not the sentiment itself.</small>',
                unsafe_allow_html=True,
            )

        case KnowledgeLevel.Expert:

            # Explanations for people with 'Expert' knowledge:
            # 1. Use gradient coloring
            # 2. Use barchart to display individual attribution scores

            fa_html = format_word_importances(
                words=[pair[0] for pair in fa_attributions],
                importances=tuple([pair[1] for pair in fa_attributions]),
            )
            st.markdown(fa_html + "<br>", unsafe_allow_html=True)
            st.markdown(
                '<small>The process of feature ablation identified how each word in the sentence influences the predicted sentiment. Green words positively influence the prediction, while red words negatively influence it. The darker the colors, the greater the influence. Note that in this context, "positive" and "negative" refer to their impact on the prediction, not the sentiment itself.<br>For better visualization, the bar chart below ranks the words based on their influence.</small>',
                unsafe_allow_html=True,
            )

            df = pd.DataFrame(fa_attributions, columns=["Word", "Influence"])

            # Create the bar chart using Altair
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X("Influence:Q", title="Influence"),
                    y=alt.Y("Word:N", sort="-x", title="Word"),
                    color=alt.Color(
                        "Influence:Q",
                        scale=alt.Scale(
                            domain=[-1, 0, 1],
                            range=["red", "white", "green"],
                        ),
                    ),
                )
                .properties(
                    title="Word Influence Ranking",
                    width=600,
                    height=400,
                )
            )

            # Display the chart in streamlit
            st.altair_chart(chart, use_container_width=True)

            # Save chart to temp folder
            fa_chart = "./src/files/temp/fa-sa-bar-chart.png"
            chart.save(fa_chart, engine="vl-convert", ppi=200)

    st.session_state.fa_attributions = fa_attributions
    st.session_state.fa_html = fa_html
    st.session_state.fa_chart = fa_chart


def display_feature_ablation_for_text_generation(
    model_input, model_output, user_knowledge_level
):

    # Generate counterfactuals
    fa_input_text_tokens, fa_output_text_tokens, fa_attributions = (
        xai.run_feature_ablation_for_text_generation(model_input, model_output)
    )

    # Set default values for session variables
    fa_most_important_tokens_df = None

    match user_knowledge_level:

        case KnowledgeLevel.Beginner:

            # Explanations for people with 'Beginner' knowledge:
            # Each output token is paired with its most influencial input token (based on attribution scores)
            # and displayed in a table

            data = pd.DataFrame(
                fa_attributions.T,
                columns=fa_output_text_tokens,
                index=fa_input_text_tokens,
            )
            most_important_tokens = (
                data.idxmax()
            )  # Find the most important input tokens for the output tokens
            fa_most_important_tokens_df = pd.DataFrame(
                {
                    "Generated Token": most_important_tokens.index,
                    "Most Important Input Token": most_important_tokens.values,
                }
            )

            st.dataframe(
                fa_most_important_tokens_df.set_index(
                    fa_most_important_tokens_df.columns[0]
                )
            )
            st.markdown(
                "<small>The process of feature ablation identified which input words had the most influence on each word generated by the model, highlighting the relationship between the input and the generated text.</small>",
                unsafe_allow_html=True,
            )

        case KnowledgeLevel.Advanced:

            # Explanations for people with 'Advanced' knowledge:
            # Use the attribution scores to create a heatmap.
            # The x-axis lists the input tokens and the y-axis lists the output tokens
            # Darker green colors indicate a stronger positive influence and darker red colors indicate a stronger negative influence of the input token (column) for the output token (row)

            max_abs_attr_val = fa_attributions.abs().max().item()
            fig, ax = plt.subplots()
            data = fa_attributions.numpy()
            fig.set_size_inches(
                max(data.shape[1] * 1.3, 6.4),
                max(data.shape[0] / 2.5, 4.8),
            )
            im = ax.imshow(
                data,
                vmax=max_abs_attr_val,
                vmin=-max_abs_attr_val,
                cmap="RdYlGn",
                aspect="auto",
            )
            # Adding the color bar without numbers
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Token Attribution", rotation=-90, va="bottom")
            cbar.ax.yaxis.set_ticks([])  # Remove the numbers from the color bar

            ax.set_xticks(np.arange(data.shape[1]), labels=fa_input_text_tokens)
            ax.set_yticks(np.arange(data.shape[0]), labels=fa_output_text_tokens)
            ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
            plt.setp(
                ax.get_xticklabels(),
                rotation=-30,
                ha="right",
                rotation_mode="anchor",
            )

            plt.savefig("./src/files/temp/fa-tg-heatmap-advanced.png")

            st.pyplot(fig)
            st.markdown(
                "<small>The process of feature ablation created this heatmap, which shows the influence of each input token (columns) on the generation of each output token (rows). Darker green colors indicate a stronger positive influence, while darker red colors indicate a stronger negative influence. In this context, darker red colors mean that the predicted word would likely have been different if it weren't for the influence of the other words.</small>",
                unsafe_allow_html=True,
            )

        case KnowledgeLevel.Expert:

            # Explanations for people with 'Expert' knowledge:
            # Use the attribution scores to create a heatmap.
            # The x-axis lists the input tokens and the y-axis lists the output tokens
            # Darker green colors indicate a stronger positive influence and darker red colors indicate a stronger negative influence of the input token (column) for the output token (row)
            # Additionally, each cell displays its actual attribution score.

            # Create heatmap
            max_abs_attr_val = fa_attributions.abs().max().item()
            fig, ax = plt.subplots()
            data = fa_attributions.numpy()
            fig.set_size_inches(
                max(data.shape[1] * 1.3, 6.4),
                max(data.shape[0] / 2.5, 4.8),
            )
            im = ax.imshow(
                data,
                vmax=max_abs_attr_val,
                vmin=-max_abs_attr_val,
                cmap="RdYlGn",
                aspect="auto",
            )
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Token Attribution", rotation=-90, va="bottom")
            ax.set_xticks(np.arange(data.shape[1]), labels=fa_input_text_tokens)
            ax.set_yticks(np.arange(data.shape[0]), labels=fa_output_text_tokens)
            ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
            plt.setp(
                ax.get_xticklabels(),
                rotation=-30,
                ha="right",
                rotation_mode="anchor",
            )
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    val = data[i, j]
                    color = "black" if 0.2 < im.norm(val) < 0.8 else "white"
                    im.axes.text(
                        j,
                        i,
                        "%.4f" % val,
                        horizontalalignment="center",
                        verticalalignment="center",
                        color=color,
                    )

            plt.savefig("./src/files/temp/fa-tg-heatmap-expert.png")

            st.pyplot(fig)
            st.markdown(
                "<small>The process of feature ablation created this heatmap, which shows the influence of each input token (columns) on the generation of each output token (rows), with individual attribution scores. Darker green colors indicate a stronger positive influence, while darker red colors indicate a stronger negative influence.</small>",
                unsafe_allow_html=True,
            )

    # Save information to session state
    st.session_state.fa_input_text_tokens = fa_input_text_tokens
    st.session_state.fa_output_text_tokens = fa_output_text_tokens
    st.session_state.fa_attributions = fa_attributions

    st.session_state.fa_most_important_tokens_df = fa_most_important_tokens_df


def display_counterfactual_for_sentiment_analysis(
    model_input,
    model_input_parts,
    sentiment_label,
    sentiment_opposite_label,
    user_knowledge_level,
):

    # Generate counterfactual and verify it
    cf_verified = False
    for _ in range(3):
        cf_latent_features, cf_key_words, cf_text = (
            xai.run_counterfactual_for_sentiment_analysis(sentiment_label, model_input)
        )
        if is_counterfactual(cf_text, sentiment_label):
            cf_verified = True
            break

    cf_text_html = None
    cf_key_words_html = None
    cf_latent_features_html = None

    if cf_verified:

        # Explanations for all knowledge levels
        # Display the counterfacutal text and highligh in yellow what words were changed / swapped

        # Identify different words between original text and counterfactual and highlight these words
        words_counterfactual = cf_text.split()
        diff = list(Differ().compare(model_input_parts, words_counterfactual))

        annotated_text_items = []
        for word in diff:
            if word.startswith("+ "):
                annotated_text_items.append((word[2:], "", "#FFFF00"))
            elif word.startswith("  "):
                annotated_text_items.append(" " + word[2:] + " ")

        annotated_text_items.extend([" :arrow_right: ", " " + sentiment_opposite_label])

        cf_text_html = annotated_text_utils.get_annotated_html(*annotated_text_items)
        st.markdown(cf_text_html, unsafe_allow_html=True)
        st.markdown(
            "<small>The counterfactual highlights the words that were swapped to change the sentiment prediction to '<i>"
            + sentiment_opposite_label
            + "</i>'</small>.",
            unsafe_allow_html=True,
        )

        if user_knowledge_level == KnowledgeLevel.Advanced:

            # Additional explanation for knowledge level "Advanced"
            # Display the most important keywords of the original input text,
            # that were identified to be most important for the sentiment label and
            # therefore for the counterfactual generation

            key_words_list = [word.strip() for word in cf_key_words.split(",")]
            cf_key_words_html = tagger_component(
                "<b>Keywords:</b>",
                key_words_list,
                color_name=["blue" for _ in key_words_list],
            )
            st.markdown(cf_key_words_html, unsafe_allow_html=True)
            st.markdown(
                f"<small>These words have been identified to be the ones that might be associated with the '<i>{sentiment_label}</i>' sentiment.</small>",
                unsafe_allow_html=True,
            )

        elif user_knowledge_level == KnowledgeLevel.Expert:

            # Additional explanations for knowledge level "Expert"
            # Display the most important keywords and latent features of the original input text,
            # that were identified to be most important for the sentiment label and
            # therefore for the counterfactual generation

            key_words_list = [word.strip() for word in cf_key_words.split(",")]
            cf_key_words_html = tagger_component(
                "<b>Keywords:</b>",
                key_words_list,
                color_name=["blue" for _ in key_words_list],
            )
            st.markdown(cf_key_words_html, unsafe_allow_html=True)
            st.markdown(
                f"<small>These keywords are the words associated with the latent features and therefore influential for the '<i>{sentiment_label}</i>' prediction.</small>",
                unsafe_allow_html=True,
            )

            latent_features_list = [
                word.strip() for word in cf_latent_features.split(",")
            ]
            cf_latent_features_html = tagger_component(
                "<b>Latent features:</b>",
                latent_features_list,
                color_name=["green" for _ in latent_features_list],
            )
            st.markdown(cf_latent_features_html, unsafe_allow_html=True)
            st.markdown(
                "<small>These latent features are hidden patterns and concepts within the text that help the model understand and predict the sentiment more accurately.</small>",
                unsafe_allow_html=True,
            )

    else:

        st.error(
            "Something went wrong while generating the counterfactual. Please try again with another input text."
        )

    # Save counterfactual information to session
    st.session_state.cf_verified = cf_verified
    st.session_state.cf_text = cf_text
    st.session_state.cf_key_words = cf_key_words
    st.session_state.cf_latent_features = cf_latent_features

    st.session_state.cf_text_html = cf_text_html
    st.session_state.cf_key_words_html = cf_key_words_html
    st.session_state.cf_latent_features_html = cf_latent_features_html


def display_counterfactual_for_text_generation(
    model_input, model_input_parts, model_output, user_knowledge_level
):

    # Generate counterfactual
    cf_key_components, cf_alternative_versions = (
        xai.run_counterfactual_for_text_generation(model_input, model_output)
    )

    cf_text_html_list = []
    counterfactual_generated_texts = []

    for counterfactual in cf_alternative_versions:

        # Explanations for all knowledge level
        # Display the counterfactual by outputting the text and highlighting which word(s) have been changed
        # Also output the newly generated text that resulted from the counterfactual text

        if not is_counterfactual(counterfactual, model_output):
            continue

        generated_text = run_model(counterfactual, save_prediction=False)

        if generated_text in counterfactual_generated_texts:
            continue
        counterfactual_generated_texts.append(generated_text)

        words_counterfactual = counterfactual.split()
        diff = list(Differ().compare(model_input_parts, words_counterfactual))

        annotated_text_items = []
        for word in diff:
            if word.startswith("+ "):
                annotated_text_items.append((word[2:], "", "#FFFF00"))
            elif word.startswith("  "):
                annotated_text_items.append(" " + word[2:] + " ")

        with st.container(border=True):
            cf_text_html = (
                annotated_text_utils.get_annotated_html(*annotated_text_items)
                + "<span style='color: blue;'>"
                + generated_text
                + "</span>"
            )
            cf_text_html_list.append(cf_text_html)
            st.markdown(cf_text_html, unsafe_allow_html=True)

    st.markdown(
        "<small>The counterfactuals show possible modifications to the original input text that would make the model predict a different output. The changed words are highlighted in yellow.</small>",
        unsafe_allow_html=True,
    )

    cf_key_components_html = None
    if user_knowledge_level != KnowledgeLevel.Beginner:

        # Additional explanations for "Advanced" and "Expert" knowledge level
        # Display the key components which represent the most important sentence parts
        # identified during the counterfactual generation

        cf_key_components_html = tagger_component(
            "<b>Key Components:</b>",
            cf_key_components,
            color_name=["blue" for _ in cf_key_components],
        )
        st.markdown(cf_key_components_html, unsafe_allow_html=True)
        st.markdown(
            "<small>Key components help identify which words or phrases have the greatest impact on the model's prediction. By focusing on these key components, you can better understand how small changes in the text can lead to different outcomes, making it easier to refine and improve the input for more accurate predictions.</small>",
            unsafe_allow_html=True,
        )

    # Save counterfactual information to session
    st.session_state.cf_verified = len(counterfactual_generated_texts) > 0
    st.session_state.cf_key_components = cf_key_components
    st.session_state.cf_alternative_versions = cf_alternative_versions

    st.session_state.cf_key_components_html = cf_key_components_html
    st.session_state.cf_text_html_list = cf_text_html_list


def display_xai() -> None:

    if not give_explanation():
        return

    with st.container(border=True):

        model_input = st.session_state.model_input
        model_input_parts = st.session_state.model_input_parts
        model_input_tokens = st.session_state.model_input_tokens
        model_output = st.session_state.model_output
        user_knowledge_level = st.session_state.user_knowledge_level

        if user_checkbox_feature_ablation and user_checkbox_counterfactual:
            col1, col2 = st.columns(2)
        else:
            col1 = col2 = st

        # Feature Ablation
        if user_checkbox_feature_ablation:

            with col1:

                st.markdown("##### Feature Ablation")

                with st.spinner("Calculating feature ablation ..."):

                    if inference_type == InferenceType.SentimentAnalysis:
                        sentiment_id = st.session_state.sentiment_id
                        sentiment_label = st.session_state.sentiment_label
                        display_feature_ablation_for_sentiment_analysis(
                            model_input,
                            model_input_tokens,
                            sentiment_id,
                            sentiment_label,
                            user_knowledge_level,
                        )

                    elif inference_type == InferenceType.TextGeneration:
                        display_feature_ablation_for_text_generation(
                            model_input, model_output, user_knowledge_level
                        )

        # Counterfactual
        if user_checkbox_counterfactual:

            with col2:

                st.markdown("##### Counterfactual")

                with st.spinner("Calculating counterfactual ..."):

                    # Display counterfactual based on inferece type
                    if inference_type == InferenceType.SentimentAnalysis:
                        sentiment_label = st.session_state.sentiment_label
                        sentiment_opposite_label = (
                            st.session_state.sentiment_opposite_label
                        )
                        display_counterfactual_for_sentiment_analysis(
                            model_input,
                            model_input_parts,
                            sentiment_label,
                            sentiment_opposite_label,
                            user_knowledge_level,
                        )

                    elif inference_type == InferenceType.TextGeneration:
                        display_counterfactual_for_text_generation(
                            model_input,
                            model_input_parts,
                            model_output,
                            user_knowledge_level,
                        )
    if user_checkbox_textual:
        st.markdown(
            "<small><i>Use the chat environment below to ask follow-up questions or to request clearer explanations for the information provided.</i></small>",
            unsafe_allow_html=True,
        )


def generate_personalized_explanation() -> None:
    """Based on the user profile settings, model input & output and calculated XAI methods,
    generate a personalized explanation for the user in the chat format"""

    user_knowledge_level = st.session_state.user_knowledge_level
    user_wants_feature_ablation = st.session_state.user_checkbox_feature_ablation
    user_wants_counterfactual = st.session_state.user_checkbox_counterfactual

    prompt = ChatPrompt(
        inference_type,
        user_knowledge_level,
        user_wants_feature_ablation,
        user_wants_counterfactual,
    ).build()

    IMAGES_BASE_PATH = "./src/files/temp/"

    user_firstname = st.session_state.user_firstname
    model_input = st.session_state.model_input

    if inference_type == InferenceType.SentimentAnalysis:

        sentiment_label = st.session_state.sentiment_label

        if user_wants_feature_ablation:
            fa_attributions = st.session_state.fa_attributions
            fa_html = st.session_state.fa_html
            fa_chart = st.session_state.fa_chart
            fa_key_words = [pair[0] for pair in fa_attributions if pair[1] != 0]

        if user_wants_counterfactual:
            cf_text = st.session_state.cf_text
            cf_key_words = st.session_state.cf_key_words
            cf_latent_features = st.session_state.cf_latent_features
            cf_sentiment = st.session_state.sentiment_opposite_label
            cf_text_html = st.session_state.cf_text_html
            cf_key_words_html = st.session_state.cf_key_words_html
            cf_latent_features_html = st.session_state.cf_latent_features_html

        # Replace placeholder independent of user knowledge level
        prompt = prompt.replace("{username}", user_firstname)
        prompt = prompt.replace("{model-name}", model_name)
        prompt = prompt.replace("{sentiment}", sentiment_label)
        prompt = prompt.replace("{text}", model_input)

        # Replace placeholder dependent of user knowledge level
        match user_knowledge_level:

            case KnowledgeLevel.Beginner:

                if user_wants_feature_ablation:
                    prompt = prompt.replace("{fa-attributions}", str(fa_attributions))
                    prompt = prompt.replace("{fa-html}", str(fa_html))

                if user_wants_counterfactual:
                    prompt = prompt.replace("{cf-text}", cf_text)
                    prompt = prompt.replace("{cf-sentiment}", cf_sentiment)
                    prompt = prompt.replace("{cf-text-html}", cf_text_html)

                st.session_state.messages = [{"role": "system", "content": prompt}]

            case KnowledgeLevel.Advanced:

                if user_wants_feature_ablation:
                    prompt = prompt.replace("{fa-attributions}", str(fa_attributions))
                    prompt = prompt.replace("{fa-html}", str(fa_html))

                if user_wants_counterfactual:
                    prompt = prompt.replace("{cf-text}", cf_text)
                    prompt = prompt.replace("{cf-sentiment}", cf_sentiment)
                    prompt = prompt.replace("{cf-text-html}", cf_text_html)
                    prompt = prompt.replace("{cf-key-words-html}", cf_key_words_html)

                st.session_state.messages = [{"role": "system", "content": prompt}]

            case KnowledgeLevel.Expert:

                if user_wants_feature_ablation:
                    prompt = prompt.replace("{fa-attributions}", str(fa_attributions))
                    prompt = prompt.replace("{fa-html}", str(fa_html))

                if user_wants_counterfactual:
                    prompt = prompt.replace("{cf-text}", cf_text)
                    prompt = prompt.replace("{cf-sentiment}", cf_sentiment)
                    prompt = prompt.replace("{cf-text-html}", cf_text_html)
                    prompt = prompt.replace("{cf-key-words-html}", cf_key_words_html)
                    prompt = prompt.replace(
                        "{cf-latent-features-html}", cf_latent_features_html
                    )

                # Attach image of chart/visualization to messages, so that chatgpt can access it
                binary_fc = open(IMAGES_BASE_PATH + "fa-sa-bar-chart.png", "rb").read()
                base64_utf8_str = base64.b64encode(binary_fc).decode("utf-8")
                dataurl = f"data:image/png;base64,{base64_utf8_str}"

                st.session_state.messages = [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "This is the bar char."},
                            {"type": "image_url", "image_url": {"url": dataurl}},
                        ],
                    },
                ]

            case _:

                raise ValueError(
                    f"The given user knowledge level '{user_knowledge_level}' does not match any case."
                )

    elif inference_type == InferenceType.TextGeneration:

        model_output = st.session_state.model_output

        if user_wants_feature_ablation:
            fa_input_text_tokens = st.session_state.fa_input_text_tokens
            fa_output_text_tokens = st.session_state.fa_output_text_tokens
            fa_attributions = st.session_state.fa_attributions
            fa_most_important_tokens_df = st.session_state.fa_most_important_tokens_df

        if user_wants_counterfactual:
            cf_verified = st.session_state.cf_verified
            cf_key_components = st.session_state.cf_key_components
            cf_alternative_versions = st.session_state.cf_alternative_versions
            cf_key_components_html = st.session_state.cf_key_components_html
            cf_text_html_list = st.session_state.cf_text_html_list

        # Replace placeholders independent of knowledge level
        prompt = prompt.replace("{username}", user_firstname)
        prompt = prompt.replace("{model-name}", model_name)
        prompt = prompt.replace("{model-input}", model_input)
        prompt = prompt.replace("{model-output}", model_output)

        if user_wants_feature_ablation:
            prompt = prompt.replace("{fa-attributions}", str(fa_attributions))

        if user_wants_counterfactual:
            prompt = prompt.replace(
                "{cf-alternative-versions}", str(cf_alternative_versions)
            )
            prompt = prompt.replace("{cf-key-components}", str(cf_key_components))
            prompt = prompt.replace("{cf-text-html-list}", str(cf_text_html_list))

        # Replace placeholder dependent of knowledge level
        match user_knowledge_level:

            case KnowledgeLevel.Beginner:

                if user_wants_feature_ablation:
                    prompt = prompt.replace(
                        "{fa-most-important-tokens-df}",
                        str(fa_most_important_tokens_df),
                    )

                st.session_state.messages = [{"role": "system", "content": prompt}]

            case KnowledgeLevel.Advanced:

                if user_wants_feature_ablation:
                    # Attach image of heatmap without individual scores to messages, so that chatgpt can access it
                    binary_of_image = open(
                        IMAGES_BASE_PATH + "fa-tg-heatmap-advanced.png", "rb"
                    ).read()
                    base64_utf8_str = base64.b64encode(binary_of_image).decode("utf-8")
                    dataurl = f"data:image/png;base64,{base64_utf8_str}"

                    st.session_state.messages = [
                        {"role": "system", "content": prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "This is the heatmap."},
                                {"type": "image_url", "image_url": {"url": dataurl}},
                            ],
                        },
                    ]

                else:

                    st.session_state.messages = [{"role": "system", "content": prompt}]

            case KnowledgeLevel.Expert:

                if user_wants_feature_ablation:

                    # Attach image of heatmap with individual scores to messages, so that chatgpt can access it
                    binary_of_image = open(
                        IMAGES_BASE_PATH + "fa-tg-heatmap-expert.png", "rb"
                    ).read()
                    base64_utf8_str = base64.b64encode(binary_of_image).decode("utf-8")
                    dataurl = f"data:image/png;base64,{base64_utf8_str}"

                    st.session_state.messages = [
                        {"role": "system", "content": prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "This is the heatmap."},
                                {"type": "image_url", "image_url": {"url": dataurl}},
                            ],
                        },
                    ]

                else:

                    st.session_state.messages = [{"role": "system", "content": prompt}]

            case _:

                raise ValueError(
                    f"The given user knowledge level '{user_knowledge_level}' does not match any case."
                )

    # Send prompt to ChatGPT an save response
    explanation = chatgpt.call(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": explanation})


def add_question_to_chat(question):
    st.session_state.messages.append({"role": "user", "content": question})


def add_action_buttons():
    """Generate three potential follow up questions, to make it easier for the user
    to use the chat. Display them in three buttons below the chat."""

    prompt = XAIPrompt(InferenceType.Undefined, "follow_up_questions").get_step()

    temp_messages = st.session_state.messages.copy()
    temp_messages.append({"role": "system", "content": prompt})

    follow_up_questions = chatgpt.call(temp_messages, "json_object")
    follow_up_questions = json.loads(follow_up_questions)["follow_up_questions"]

    with st.container():

        row = create_row(3, vertical_align="center")

        for question in follow_up_questions:

            row.button(
                question,
                on_click=add_question_to_chat,
                args=(question,),
                use_container_width=True,
            )


@st.experimental_fragment
def display_chat() -> None:

    if give_explanation() and user_checkbox_textual:

        st.markdown("#### Chat with Explanation")

        with st.container():

            with st.spinner("Creating personalized explanation ..."):

                messages_container = st.container()

                if "messages" not in st.session_state:
                    generate_personalized_explanation()

                # Display chat history
                if "messages" in st.session_state:
                    if st.session_state.user_wants_initial_explanation:
                        # Display messages starting with first 'assistant' message
                        # Previous messages include system instructions or other additional information
                        found_assistant = False
                        for msg in st.session_state.messages:
                            if found_assistant or msg["role"] == "assistant":
                                found_assistant = True
                                messages_container.chat_message(msg["role"]).write(
                                    msg["content"]
                                )
                    else:
                        # Display messages starting with first real 'user' message
                        # The user does not want an initial explanation
                        found_user = False
                        for msg in st.session_state.messages:
                            if found_user or msg["role"] == "user":
                                content = msg["content"]
                                if not isinstance(content, list):
                                    found_user = True
                                    messages_container.chat_message(msg["role"]).write(
                                        content
                                    )

                add_action_buttons()

                if (
                    "messages" in st.session_state
                    and len(st.session_state.messages) > 0
                    and st.session_state.messages[len(st.session_state.messages) - 1][
                        "role"
                    ]
                    == "user"
                ):

                    msg = chatgpt.call(st.session_state.messages)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": msg}
                    )
                    messages_container.chat_message("assistant").write(msg)

                if prompt := st.chat_input():
                    st.session_state.messages.append(
                        {"role": "user", "content": prompt}
                    )
                    messages_container.chat_message("user").write(prompt)

                    msg = chatgpt.call(st.session_state.messages)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": msg}
                    )
                    messages_container.chat_message("assistant").write(msg)


ProfileTab, HomeTab = st.tabs(["Profile", "Home"])

with ProfileTab:

    st.write("#### Profile Settings")

    st.markdown(
        "<small>You can choose to receive personalized explanations or simply let the model make its prediction.</small>",
        unsafe_allow_html=True,
    )
    st.toggle(
        "Would you like to receive personalized explanations?",
        key="user_personalization_enabled",
        value=True,
    )

    if st.session_state.user_personalization_enabled:

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("##### Knowledge Level")

            st.radio(
                "How familiar are you with eXplainable Artificial Intelligence (XAI)?",
                ["Beginner", "Advanced", "Expert"],
                captions=[
                    "I am not familiar with XAI.",
                    "I have heard about XAI.",
                    "I know what XAI is and I have seen applications of it.",
                ],
                key="user_xai_knowledge",
                on_change=reset_cache,
            )

            st.session_state.user_knowledge_level = KnowledgeLevel[
                st.session_state.user_xai_knowledge
            ]

        with col2:

            st.markdown("##### Preferred Explanation Methods")
            st.markdown(
                "<small>After you make a prediction using a model, explanation methods are deployed to help you understand why the model made that prediction. Please select the explanation methods you would like to use. You can use the tooltips next to each method to learn more about them.</small>",
                unsafe_allow_html=True,
            )
            user_checkbox_feature_ablation = st.checkbox(
                "Feature Ablation",
                value=True,
                key="user_checkbox_feature_ablation",
                help="Feature ablation is an explanation method that helps us understand how a model works by removing one word at a time from the input text and seeing how it affects the model's prediction / output.",
                on_change=reset_cache,
            )
            user_checkbox_counterfactual = st.checkbox(
                "Counterfactual",
                value=True,
                key="user_checkbox_counterfactual",
                help="Counterfactuals are alternative text inputs used to understand model decisions by showing how small changes to input features (input words) can lead to different outcomes.",
                on_change=reset_cache,
            )
            user_checkbox_textual = st.checkbox(
                "Self-Explainability",
                value=False,
                key="user_checkbox_textual",
                help="Engage with explanations in a chat-style environment, allowing you to ask follow-up questions for a deeper understanding.",
                on_change=reset_cache,
            )

            if user_checkbox_textual:

                col3, col4 = st.columns([0.02, 0.98])

                with col4:
                    st.markdown(
                        "<small>Please choose whether you would like to receive an automatically generated initial explanation or only receive textual explanations upon request.</small>",
                        unsafe_allow_html=True,
                    )
                    st.toggle(
                        "Please generate an initial explanation.",
                        key="user_wants_initial_explanation",
                        value=True,
                    )

                    user_firstname = st.text_input(
                        "Your firstname (needed for textual explanations)",
                        key="user_firstname",
                        help="This will help us personalize the explanations even more.",
                    )

with HomeTab:

    col1, col2 = st.columns([0.3, 0.7])

    with col1:

        if inference_type == InferenceType.SentimentAnalysis:
            display_sentiment_analysis()

        elif inference_type == InferenceType.TextGeneration:
            display_text_generation()

    with col2:
        display_xai()

    display_chat()
