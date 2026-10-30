"""Shared presentation helpers for the cloud UI; all personal data is supplied by the caller."""
import pandas as pd
import streamlit as st

from .modeling import MODEL_FEATURES, MODEL_LABELS, TARGET
from .predictor import ModelArtifact
from .ui_format import METRIC_MODEL_NAMES, RULE_DEFINITION, RULE_NAME
from .web_charts import error_figure
from .web_history import evaluation_frames


def duration(minutes):
    if minutes is None or pd.isna(minutes):
        return "Unavailable"
    rounded = int(round(float(minutes)))
    hours, remainder = divmod(rounded, 60)
    return f"{hours} hr {remainder:02d} min" if hours else f"{remainder} min"


def number(value, suffix="", decimals=1):
    return "Unavailable" if value is None or pd.isna(value) else f"{value:,.{decimals}f}{suffix}"


def plot(figure, key):
    # theme=None keeps the forest palette and fonts set by web_charts._style.
    st.plotly_chart(figure, width="stretch", key=key, theme=None,
                    config={"displaylogo": False, "scrollZoom": False, "displayModeBar": False})


# Theme tokens in .streamlit/config.toml do most of the styling. This CSS covers
# only what tokens cannot express, scoped to keyed containers (st-key-*) where possible.
_CSS = """
<style>
.block-container { max-width: 1200px; padding-bottom: 4rem; }
[data-testid="stCaptionContainer"] { opacity: 1; color: #4a6255; }
[data-testid="stCaptionContainer"] p { color: inherit; }
.st-key-trip_hero { background: #ffffff; }
.st-key-trip_hero [data-testid="stMetricValue"] { font-size: clamp(2.2rem, 6vw, 3.2rem); font-weight: 700; color: #1d4634; line-height: 1.1; }
.st-key-trip_hero .st-key-trip_supporting [data-testid="stMetricValue"] { font-size: 1.25rem; font-weight: 600; color: #1f3329; }
.st-key-home_plan_cta, .st-key-landing_card, .st-key-get_started { background: #ffffff; }
.st-key-danger_zone { border: 1px solid #c9574c !important; border-radius: 0.6rem; padding: 1rem 1.1rem; background: #fdf7f6; }
.st-key-danger_zone [data-testid="stFormSubmitButton"] button { background: #b3261e; border-color: #b3261e; color: #fff; }
.landing-lead { font-size: 1.2rem; line-height: 1.55; color: #33493e; margin: 0.25rem 0 0.75rem; }
.step-number { display: inline-flex; align-items: center; justify-content: center; width: 1.7rem; height: 1.7rem;
  border-radius: 50%; background: #24533f; color: #fff; font-weight: 700; font-size: 0.9rem; margin-right: 0.5rem; }
.st-key-trip_hero .stMarkdownBadge, .st-key-trip_hero [data-testid="stMarkdownBadge"] { white-space: normal; overflow: visible; text-overflow: clip; height: auto; line-height: 1.4; }
[data-testid="stTopNavLink"]:focus-visible { outline: 2px solid #24533f; outline-offset: 2px; }
@media (max-width: 640px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .st-key-trip_hero [data-testid="stMetricValue"] { font-size: 2.3rem; }
}
</style>
"""


def design():
    st.markdown(_CSS, unsafe_allow_html=True)


def _relabel(frame):
    shown = frame.copy()
    shown["model"] = shown.model.map(lambda m: METRIC_MODEL_NAMES.get(m, m))
    return shown


def performance_page(artifact, embedded=False):
    """Saved evaluation of the moving-time model. Embedded in My pace as tabs, without a page title."""
    if not embedded:
        st.title("Model performance")
    if artifact is None:
        st.info("Saved model performance is unavailable because the artifact could not be loaded.")
        artifact = ModelArtifact()
    frames = evaluation_frames(artifact)
    evaluation = artifact.evaluation
    accuracy, how, coefficients, method, provenance = st.tabs(
        ["Accuracy", "How it works", "Coefficients", "Method & limits", "Provenance"])
    with accuracy:
        columns = st.columns(3)
        columns[0].metric("Trail segments used", number(evaluation.get("eligible_segments"), decimals=0),
                          help="Half-mile pieces of your recordings that passed the GPS quality checks.")
        columns[1].metric("Hikes used", number(evaluation.get("eligible_hikes"), decimals=0))
        columns[2].metric("Complete test hikes", number(evaluation.get("verified_held_out_hikes"), decimals=0),
                          help="Whole recordings that could be predicted while left out of training.")
        st.markdown("Average miss is the typical difference between predicted and actual moving minutes. "
                    "RMSE weighs big misses more heavily.")
        for key, title, note in (
            ("whole_hike_metrics", "Whole hikes", "Only complete recordings qualify. A single hike cannot establish reliability."),
            ("segment_metrics", "Half-mile segments", "Each observation is one segment, not an entire hike."),
        ):
            st.markdown(f"**{title}**")
            if frames[key].empty:
                st.caption("Evaluation results are unavailable for this training run.")
                continue
            shown = _relabel(frames[key])
            plot(error_figure(shown), key)
            st.dataframe(shown, hide_index=True, width="stretch", column_order=["model", "hikes", "observations", "MAE_min", "RMSE_min"],
                         column_config={"model": st.column_config.TextColumn("Method", width="large"),
                                        "hikes": st.column_config.NumberColumn("Hikes"),
                                        "observations": st.column_config.NumberColumn("Tested"),
                                        "MAE_min": st.column_config.NumberColumn("Average miss", format="%.1f min"),
                                        "RMSE_min": st.column_config.NumberColumn("RMSE", format="%.1f min")})
            st.caption(note + " These are errors on past hikes, not a guaranteed range for a future trip.")
    with how:
        st.write("Your pace model is a linear regression fitted to half-mile segments of your recordings.")
        left, right = st.columns(2)
        with left, st.container(border=True):
            st.markdown("**Distance only (A)**")
            st.latex(r"\hat{t} = b_0 + b_1 d")
            st.caption("d is distance in miles; t is moving time in minutes.")
        with right, st.container(border=True):
            st.markdown("**Distance + climbing (B, C)**")
            st.latex(r"\hat{t} = b_0 + b_d d + b_g g + b_l l")
            st.caption("g and l are ascent and descent in feet. C also adds a hike-type offset when both types have enough hikes.")
        features = MODEL_FEATURES.get(artifact.model_name, [])
        st.write(f"**Inputs used:** {', '.join(features) if features else 'none yet; no fitted model.'}")
        st.write(f"**Predicts:** {TARGET}, measured moving minutes per segment, excluding stops and unclear GPS time.")
        if artifact.model_name:
            st.caption(f"Saved fit: {MODEL_LABELS[artifact.model_name]}. "
                       f"{'Used for your estimates.' if artifact.selected else f'Preview only; estimates use the {RULE_NAME}.'}")
    with coefficients:
        if frames["coefficients"].empty:
            st.caption("Coefficients appear after a pace model has been fitted.")
        else:
            st.dataframe(frames["coefficients"], hide_index=True, width="stretch", column_order=["term", "value"],
                         column_config={"term": st.column_config.TextColumn("Input"),
                                        "value": st.column_config.NumberColumn("Effect", format="%.4g")})
            st.caption("The distance effect is minutes per extra mile; gain and loss effects are minutes per extra foot. "
                       "Small samples and correlated terrain can make these unstable; a negative intercept is not meaningful.")
    with method:
        st.write(f"**Evaluation method:** {evaluation.get('method', 'not available for this run.')}")
        st.write("All segments of a hike stay together when testing. With 2–20 hikes, each hike is left out once and "
                 "predicted from the others; larger histories use a fixed 25% holdout. The pace model and the "
                 f"{RULE_NAME} are compared on the same hikes.")
        st.latex(r"t_{\mathrm{rule}} = 20d + 0.03g\quad\text{minutes}")
        st.write(f"The {RULE_NAME} (Naismith's rule) is {RULE_DEFINITION}, with no allowance for descent or breaks. "
                 "Your pace model becomes the main estimate only when it beats the rule across at least "
                 f"{evaluation.get('minimum_validation_hikes', 10)} complete test hikes.")
        st.caption("Neighboring segments share terrain and conditions, so segment counts overstate the evidence. "
                   "Weather, footing, fatigue and pack weight are not modeled.")
        if evaluation.get("model_c_status"):
            st.caption(evaluation["model_c_status"])
        if not frames["folds"].empty:
            with st.expander("Training and test folds"):
                st.dataframe(frames["folds"], hide_index=True, width="stretch")
    with provenance:
        if artifact.provenance:
            st.json(artifact.provenance, expanded=False)
        else:
            st.caption("No saved model yet.")
