"""Local UI for the existing hiking predictor. Launch: python -m streamlit run streamlit_app.py."""

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from hiking_time_predictor.break_model import BreakArtifact, load_break_model
from hiking_time_predictor.breaks import CLASSIFICATIONS, save_corrections
from hiking_time_predictor.gpx import ImportConfig
from hiking_time_predictor.web_breaks import break_history_figures, load_break_history
from hiking_time_predictor.modeling import MODEL_FEATURES, MODEL_LABELS, TARGET
from hiking_time_predictor.predictor import ModelArtifact, load_model
from hiking_time_predictor.web_charts import (
    actual_predicted_figure, elevation_figure, error_figure, history_distance_figure,
    segment_csv, segment_figure, trail_map,
)
from hiking_time_predictor.web_history import evaluation_frames, load_history
from hiking_time_predictor.web_upload import predict_upload, preview_upload, trip_statistics


PROJECT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT / "models" / "hiking_predictor.joblib"
BREAK_MODEL_PATH = PROJECT / "models" / "break_predictor.joblib"
PAGES = ("Hiking Time Predictor", "My Hiking Data", "Model Performance")


@st.cache_resource(show_spinner=False, max_entries=2)
def cached_model(path, modified_ns, size, runtime_types):
    """Refresh for file changes and class reloads, retaining strict validation.

    Streamlit's resource cache survives source reloads. The runtime type IDs
    belong in its key so an artifact from an old class is loaded again using
    the current ModelArtifact and ImportConfig definitions.
    """
    return load_model(path)


def model_state():
    if not MODEL_PATH.exists():
        return ModelArtifact(), None
    try:
        stat = MODEL_PATH.stat()
        runtime_types = (id(ModelArtifact), id(ImportConfig))
        return cached_model(str(MODEL_PATH), stat.st_mtime_ns, stat.st_size, runtime_types), None
    except (ValueError, OSError) as error:
        return None, str(error)


def break_model_state():
    try:
        return load_break_model(BREAK_MODEL_PATH, project=PROJECT)
    except (ValueError, OSError) as error:
        return BreakArtifact(reason=str(error))


def duration(minutes):
    if minutes is None or pd.isna(minutes):
        return "Unavailable"
    rounded = int(round(float(minutes)))
    hours, remainder = divmod(rounded, 60)
    return f"{hours} hr {remainder:02d} min" if hours else f"{remainder} min"


def number(value, suffix="", decimals=1):
    return "Unavailable" if value is None or pd.isna(value) else f"{value:,.{decimals}f}{suffix}"


def plot(figure, key):
    st.plotly_chart(figure, width="stretch", key=key,
                    config={"displaylogo": False, "scrollZoom": False})


def design():
    st.markdown("""
    <style>
    .stApp { background: #f8faf8; }
    .block-container { padding-top: 2.3rem; padding-bottom: 3rem; max-width: 1440px; }
    h1, h2, h3 { color: #214936; letter-spacing: -0.025em; }
    h1 { font-weight: 750; }
    [data-testid="stSidebar"] { border-right: 1px solid #d6e2d8; }
    [data-testid="stMetric"] { padding: 1rem 0.9rem; border-left: 3px solid #9bb184;
        background: white; border-radius: 0 7px 7px 0; }
    [data-testid="stMetricLabel"] { color: #4c6557; }
    [data-testid="stMetricValue"] { font-size: clamp(1.3rem, 2vw, 2rem); }
    .trail-intro { border-left: 4px solid #819857; padding: .25rem 0 .25rem 1.2rem;
        margin: .7rem 0 1.7rem; color: #52695b; max-width: 70ch; }
    .trail-brand { font-size: 1.35rem; font-weight: 750; color: #24533f; margin-bottom: .1rem; }
    @media (max-width: 640px) { .block-container { padding-top: 1.25rem; } }
    </style>
    """, unsafe_allow_html=True)


def request_prediction():
    # Streamlit runs callbacks before creating this rerun's navigation widget.
    st.session_state.page = PAGES[0]


def sidebar(artifact, model_error):
    with st.sidebar:
        st.markdown('<div class="trail-brand">⛰ Personal hiking time</div>', unsafe_allow_html=True)
        st.caption("Plan with your own trail history.")
        page = st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
        st.divider()
        st.subheader("Plan a hike")
        uploaded = st.file_uploader("Upload GPX file", type=["gpx"], key="gpx_upload",
                                    help="Choose one intended route with elevations. Timestamps are optional. Maximum 20 MB.")
        hike_label = st.selectbox("Hike type", ["Day Hiking", "Backpacking"], key="hike_type")
        direction_label = st.selectbox("Route direction", ["One Way", "Out and Back"], key="route_direction")
        if direction_label == "Out and Back":
            st.caption("Upload only the outbound trail. The return follows the same points in reverse.")
        include_breaks = st.checkbox("Include predicted break time", value=True, key="include_breaks")
        override = st.checkbox("Override break estimate", value=False, key="override_breaks", disabled=not include_breaks)
        breaks = st.number_input("Planned breaks (minutes)", min_value=0, max_value=10080,
                                  value=0, step=5, key="break_minutes", disabled=not (include_breaks and override))
        st.caption("Breaks are added to moving time. Pack weight is not a feature of the current models.")
        if artifact is not None and artifact.model_name in ("A", "B"):
            st.caption("This saved model uses route geometry; changing hike type does not adjust its time.")
        clicked = st.button("Predict Hiking Time", type="primary", width="stretch",
                             disabled=uploaded is None or model_error is not None, key="predict",
                             on_click=request_prediction)
        st.caption("Uploads stay in this session. Temporary files are removed after parsing; tracks are not saved to the project.")
        st.divider()
        if model_error:
            st.error("The local model could not be loaded.")
        elif artifact.estimator is None:
            st.caption("Predictor: Naismith baseline. No evaluated regression is available.")
        elif artifact.selected:
            st.caption(f"Predictor: {MODEL_LABELS[artifact.model_name]}")
        else:
            st.caption("Predictor: Naismith baseline + experimental personalized regression.")
    return (page, uploaded, "day_hike" if hike_label == "Day Hiking" else "backpacking",
            "one-way" if direction_label == "One Way" else "out-and-back", breaks, clicked, include_breaks, override)


def route_state(uploaded, hike_type, direction, artifact, break_artifact=None):
    """Retain private upload results in the current session, never in a shared cache."""
    if uploaded is None:
        for key in ("route_preview", "route_key", "prediction", "prediction_key"):
            st.session_state.pop(key, None)
        return None, None, None
    contents = uploaded.getvalue()
    config = artifact.config if artifact else None
    stat = MODEL_PATH.stat() if MODEL_PATH.exists() else None
    model_version = (stat.st_mtime_ns, stat.st_size) if stat else None
    break_version = (break_artifact.reason, break_artifact.minutes_per_mile, break_artifact.selected_model,
                     repr(break_artifact.provenance)) if break_artifact else None
    runtime_types = (id(ModelArtifact), id(ImportConfig))
    key = (hashlib.sha256(contents).hexdigest(), uploaded.name, hike_type, direction, repr(config),
           model_version, break_version, runtime_types)
    if st.session_state.get("route_key") != key:
        st.session_state.pop("prediction", None)
        st.session_state.pop("prediction_key", None)
        st.session_state.pop("route_preview", None)
        st.session_state.pop("route_key", None)
        with st.spinner("Reading trail geometry…"):
            route = preview_upload(contents, uploaded.name, hike_type, direction, config=config)
        st.session_state.route_preview = route
        st.session_state.route_key = key
    return st.session_state.route_preview, contents, key


def predictor_page(route, contents, route_key, uploaded, hike_type, direction, breaks, clicked, artifact,
                   break_artifact=None, include_breaks=True, override=False):
    st.title("Hiking Time Predictor")
    st.markdown('<div class="trail-intro">Know the trail. Plan your time.<br>'
                'Explore an unhiked route and estimate moving time from your recorded hiking history.</div>',
                unsafe_allow_html=True)
    if route is None:
        st.info("Upload a GPX file in the sidebar to see the trail and elevation profile.")
        with st.container(border=True):
            st.subheader("Your next trail starts here")
            st.write("Choose Day Hiking or Backpacking, select the route direction, and add any planned breaks. "
                     "Then select Predict Hiking Time.")
            st.caption("Untimed routes are supported. The GPX needs trackpoints or routepoints with elevations. "
                       "Out and Back assumes the file contains only the outbound leg.")
        return
    st.subheader(route.name)
    columns = st.columns(4)
    for col, label, value in zip(columns,
                                ["Distance", "Elevation gain", "Elevation loss", "Segments"],
                                [number(route.segments.distance_miles.sum(), " mi", 2),
                                 number(route.segments.elevation_gain_ft.sum(), " ft", 0),
                                 number(route.segments.elevation_loss_ft.sum(), " ft", 0), str(len(route.segments))]):
        col.metric(label, value)
    st.caption("Route totals include both legs when Out and Back is selected. Elevation uses the model's preprocessing settings.")

    if clicked and artifact is not None:
        with st.spinner("Predicting each trail segment with the saved pipeline…"):
            prediction = predict_upload(contents, uploaded.name, hike_type, direction, artifact=artifact,
                                        break_artifact=break_artifact)
        st.session_state.prediction = prediction
        st.session_state.prediction_key = route_key
        st.success("Prediction ready. Explore the route below or download the segment table.")
    prediction = st.session_state.get("prediction") if st.session_state.get("prediction_key") == route_key else None

    st.subheader("Trail map")
    st_folium(trail_map(route), height=430, use_container_width=True,
              returned_objects=[], key="trail_" + hashlib.sha256(repr(route_key).encode()).hexdigest()[:16])
    st.caption("OpenStreetMap basemap. Green marks the start; blue marks the finish. "
               + ("Orange marks the turnaround; the dotted line is the return." if direction == "out-and-back" else ""))
    if prediction is not None:
        statistics = trip_statistics(prediction, include_predicted_breaks=include_breaks,
                                     manual_break_minutes=breaks if override and include_breaks else None)
        applied_breaks = statistics["break_minutes"]
        st.subheader("Predicted hiking statistics")
        st.caption(f"Primary estimate: {prediction.forecast.label}. These are planning estimates, not guaranteed arrival times.")
        columns = st.columns(4)
        for col, label, value in zip(columns,
                                    ["Estimated moving time", "Planned breaks", "Estimated total trip", "Moving speed"],
                                    [duration(statistics["moving_minutes"]), duration(applied_breaks),
                                     duration(statistics["total_minutes"]), number(statistics["average_speed_mph"], " mph", 2)]):
            col.metric(label, value)
        st.caption(f"Naismith comparison: {duration(statistics['naismith_minutes'])} moving time "
                   "at 3 mph plus 1 hour per 2,000 ft of ascent. Breaks and descent are excluded from that rule.")
        st.subheader("BREAK TIME ESTIMATION")
        if include_breaks:
            forecast = prediction.break_forecast
            left, right = st.columns(2)
            left.metric("Original predicted break time", duration(forecast.predicted_minutes))
            right.metric("Estimated total trail time including breaks", duration(statistics["total_minutes"]))
            if forecast.predicted_minutes is None:
                st.info("A personal break estimate is unavailable. " + " ".join(forecast.warnings))
                if not override:
                    st.caption("The displayed total currently contains moving time only.")
            else:
                st.caption(f"{forecast.label}. Personal distance baseline: {duration(forecast.baseline_minutes)}.")
                left, right = st.columns(2)
                left.metric("Estimated number of breaks", number(forecast.expected_break_count, decimals=1))
                right.metric("Average expected break duration", number(forecast.expected_average_minutes, " min", 2))
                st.caption("Break count and duration use historical rates, not separately validated regression models. "
                           "The expected count can be fractional.")
                with st.expander("Break data quality and reliability"):
                    for warning in forecast.warnings:
                        st.write(warning)
                    for name, value in forecast.experimental_minutes.items():
                        st.write(f"Experimental break regression {name}: {value:.2f} min")
            if override:
                st.caption(f"Manual override applied: {applied_breaks:.2f} break minutes. The original model prediction remains above.")
        else:
            st.caption("Predicted breaks are disabled. Total trail time equals moving time.")
        if prediction.forecast.experimental_minutes is not None:
            with st.container(border=True):
                st.subheader("Experimental personalized estimate")
                left, right = st.columns(2)
                left.metric("Saved regression moving time", duration(prediction.forecast.experimental_minutes))
                right.metric("Regression + planned breaks", duration(prediction.forecast.experimental_minutes + applied_breaks))
                st.write(prediction.artifact.reason)
                st.caption("The saved fit uses your hiking history, but the validation sample does not yet support using it "
                           "as the primary trip estimate. The primary cards and CSV predicted_time_min retain Naismith; "
                           "regression_time_min contains this experimental estimate.")
        with st.expander("Prediction limits and route notes", expanded=bool(prediction.forecast.warnings)):
            for note in dict.fromkeys([prediction.artifact.reason, *route.notes, *prediction.forecast.warnings]):
                st.write(f"• {note}")
            st.caption("Weather, surface conditions, fatigue, navigation, and unmodeled pack weight can change your pace. "
                       "Errors on held-out segments do not form a prediction interval for this route.")
    else:
        st.info("Trail preview ready. Select Predict Hiking Time in the sidebar to calculate moving time.")
    st.subheader("Elevation profile")
    plot(elevation_figure(route), "elevation")
    st.caption("Smoothed GPX elevation. Disconnected sections are plotted separately; gaps add no distance.")
    if prediction is not None:
        st.subheader("Segment predictions")
        plot(segment_figure(prediction), "segments")
        st.download_button("Download segment predictions CSV", segment_csv(prediction),
                           file_name="hiking_segment_predictions.csv", mime="text/csv", key="download")
        st.caption("Segment times exclude planned breaks. CSV and chart use the same unrounded prediction rows.")
        with st.expander("View segment table"):
            st.dataframe(prediction.segments, hide_index=True, width="stretch")


def breaking_habits_page(break_artifact):
    st.subheader("MY BREAKING HABITS")
    if not (PROJECT / "data/processed/break_events.csv").exists():
        st.info("Run python -m hiking_predictor analyze-breaks to analyze historical recordings.")
        return
    # An absent/stale artifact cannot certify that charts match current labels.
    if break_artifact.minutes_per_mile is None:
        st.info(break_artifact.reason)
        return
    hikes, events, held_out = load_break_history(PROJECT)
    ordinary = events.loc[events.break_classification.eq("stationary_break")]
    columns = st.columns(4)
    for col, label, value in zip(columns,
                                ["Historical breaks", "Historical break minutes", "Average individual break", "Personal break minutes per mile"],
                                [str(len(ordinary)), number(ordinary.duration_minutes.sum(), " min", 2),
                                 number(ordinary.duration_minutes.mean(), " min", 2), number(break_artifact.minutes_per_mile, " min/mi", 2)]):
        col.metric(label, value)
    st.caption(break_artifact.reason)
    for key, figure in break_history_figures(hikes, events, held_out, break_artifact.selected_model).items():
        plot(figure, key)
    if held_out.empty:
        st.info("Actual vs. predicted break time is unavailable: too few independent hikes for validation.")
    st.caption("Break labels exclude unknown intervals, camping, non-hiking activity, and short stops. "
               "The personal rate uses eligible recordings. Held-out predictions exclude their entire hike from training.")
    with st.expander("Review and correct detected breaks"):
        st.write("Edit classifications or notes, then save corrections. Rerun python -m hiking_predictor analyze-breaks "
                 "to refresh the dataset and model. Corrections are stored separately and survive reprocessing. "
                 "Changes affect break labels; moving-time preprocessing uses the existing confirmed_rests.csv workflow.")
        review = events.loc[events.break_classification.ne("moving")].copy()
        edited = st.data_editor(review, hide_index=True, width="stretch", key="break_review",
                                disabled=[c for c in review.columns if c not in ("break_classification", "correction_notes")],
                                column_config={"break_classification": st.column_config.SelectboxColumn("Classification", options=CLASSIFICATIONS, required=True)})
        if st.button("Save break corrections", key="save_break_corrections"):
            count = save_corrections(events, edited, PROJECT / "data/break_corrections.csv", break_artifact.config)
            st.success(f"Saved {count} corrections. Rerun analyze-breaks before using refreshed predictions.")
        st.download_button("Download break events CSV", events.to_csv(index=False).encode(), "break_events.csv", "text/csv", key="download_break_events")
    with st.expander("Break measurements by hike"):
        st.dataframe(hikes, hide_index=True, width="stretch")


def history_page(artifact, break_artifact=None):
    st.title("My Hiking Data")
    st.write("The recorded miles and measured moving time behind your predictor.")
    with st.spinner("Reading your existing hiking tables…"):
        history = load_history(PROJECT, artifact.model_name if artifact else None)
    metrics = history.metrics
    columns = st.columns(4)
    for col, label, value in zip(columns,
                                ["Total logged miles", "Completed hike recordings", "Usable training segments", "Average moving pace"],
                                [number(metrics.get("logged_miles"), " mi", 2), number(metrics.get("hike_count"), decimals=0),
                                 number(metrics.get("usable_segments"), decimals=0),
                                 number(metrics.get("average_pace_min_per_mile"), " min/mi", 2)]):
        col.metric(label, value)
    st.caption("Completed hikes counts imported recording IDs; it does not certify complete usable training coverage. "
               "Moving pace is total measured moving minutes divided by total measured moving miles.")
    st.subheader("Elevation in your recordings")
    columns = st.columns(3)
    for col, label, key in zip(columns, ["Total elevation gain", "Average gain per hike", "Largest recorded gain"],
                               ["elevation_gain_total_ft", "elevation_gain_mean_ft", "elevation_gain_max_ft"]):
        col.metric(label, number(metrics.get(key), " ft", 0))
    if not history.summary.empty and metrics.get("average_pace_min_per_mile") is not None:
        st.subheader("Distance and moving time")
        plot(history_distance_figure(history.summary), "history_distance")
        st.caption("Measured moving distance and time from the processed summaries, before segment quality filtering.")
    elif not history.summary.empty:
        st.info("The distance and moving-time chart is unavailable because the summary measurements are incomplete or invalid.")
    st.subheader("Actual and predicted moving time")
    if history.held_out_predictions.empty:
        st.info("Matching held-out predictions are unavailable. This chart needs saved evaluation exports "
                "whose provenance matches the current hiking data.")
    else:
        plot(actual_predicted_figure(history.held_out_predictions), "actual_predicted")
        st.caption("Each dot is an existing held-out segment prediction. Every validation hike was excluded from its training fold. "
                   "This chart compares segments, not whole-trip durations.")
    if not history.hike_totals.empty:
        with st.expander("Held-out hike totals and coverage"):
            st.caption("Most totals may represent only the retained portion of a recording. "
                       "Only rows marked whole_hike_eligible qualify as complete hike comparisons.")
            st.dataframe(history.hike_totals, hide_index=True, width="stretch")
    with st.expander("Data sources and quality notes", expanded=history.summary.empty):
        for note in history.notes:
            st.write(f"• {note}")
        st.caption("The app reads existing local tables and exports. Importing and model training remain explicit CLI/notebook actions.")
    if not history.summary.empty:
        with st.expander("Recorded hike summaries"):
            st.dataframe(history.summary, hide_index=True, width="stretch")
    breaking_habits_page(break_artifact or BreakArtifact())


def performance_page(artifact):
    st.title("Model Performance")
    st.write("Understand what the model learned, how it was tested, and where its predictions need caution.")
    if artifact is None:
        st.info("Saved model performance is unavailable because the local artifact could not be loaded.")
        artifact = ModelArtifact()
    st.info(artifact.reason)
    st.caption("These evaluations and sample counts belong to the saved artifact's training run. "
               "My Hiking Data shows the current processed recordings.")
    frames = evaluation_frames(artifact)
    evaluation = artifact.evaluation
    columns = st.columns(3)
    columns[0].metric("Eligible training segments", number(evaluation.get("eligible_segments"), decimals=0))
    columns[1].metric("Independent eligible hikes", number(evaluation.get("eligible_hikes"), decimals=0))
    columns[2].metric("Verified held-out whole hikes", number(evaluation.get("verified_held_out_hikes"), decimals=0))
    if artifact.model_name:
        st.caption(f"Saved fitted model: {MODEL_LABELS[artifact.model_name]}. "
                   f"Status: {'selected for primary predictions' if artifact.selected else 'experimental; primary estimates use Naismith'}.")
    st.subheader("How the regressions work")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Simple Linear Regression")
        st.write("Model A fits one input: segment distance. It finds the line that minimizes squared errors in moving time.")
        st.latex(r"\hat{t} = b_0 + b_1 d")
        st.caption("d is distance in miles; t is moving time in minutes.")
    with right, st.container(border=True):
        st.subheader("Multiple Linear Regression")
        st.write("Model B adds elevation gain and loss. Model C also encodes hike type when both categories have enough training coverage.")
        st.latex(r"\hat{t} = b_0 + b_d d + b_g g + b_l l")
        st.caption("g and l are ascent and descent in feet. Model C adds a category offset.")
    features = MODEL_FEATURES.get(artifact.model_name, [])
    st.write(f"**Saved model features:** {', '.join(features) if features else 'Unavailable; no fitted model.'}")
    st.write(f"**Target:** {TARGET}, measured moving minutes per distance segment, excluding classified rests and unknown time.")
    st.caption("Identifiers, timestamps, observed duration, and quality flags are excluded from prediction inputs. "
               "Pack weight is not supported by the current feature schemas.")

    st.subheader("Regression coefficients")
    if frames["coefficients"].empty:
        st.info("Coefficients are unavailable because no supported fitted estimator was loaded.")
    else:
        st.dataframe(frames["coefficients"], hide_index=True, width="stretch")
        st.write("The intercept is the fitted offset in minutes per segment. The distance coefficient is minutes per additional mile; "
                 "gain and loss coefficients are minutes per additional foot, holding other inputs fixed. "
                 "A hike-type coefficient is an offset from the encoder's reference category.")
        st.caption("Coefficients describe associations in this small sample. Correlated terrain features and short final segments "
                   "can make them unstable. A negative intercept is not a physically meaningful rest or speed.")

    st.subheader("Validation errors")
    st.write("Mean Absolute Error (MAE) is the average absolute difference between predicted and actual moving minutes. "
             "Root Mean Squared Error (RMSE) squares errors before averaging and taking a square root, so large misses count more.")
    for key, title, note in (
        ("segment_metrics", "Held-out segment errors", "Each observation is one retained segment, not an entire hike."),
        ("whole_hike_metrics", "Verified whole-hike errors", "Only complete recordings with verified coverage qualify. A single hike cannot establish reliability."),
    ):
        st.markdown(f"**{title}**")
        if frames[key].empty:
            st.info("Evaluation metrics are unavailable in the saved artifact.")
        else:
            plot(error_figure(frames[key]), key)
            st.dataframe(frames[key], hide_index=True, width="stretch")
            st.caption(note + " MAE and RMSE are in minutes; they are not uncertainty bounds for a future trip.")
    st.subheader("Training and validation methodology")
    st.write(f"**Saved evaluation method:** {evaluation.get('method', 'Unavailable in the saved artifact.')}")
    st.write("The backend groups by hike ID before splitting: all segments of a hike stay together. "
             "For 2–20 eligible hikes it leaves one hike out at a time; larger datasets use a fixed 25% hike holdout. "
             "Quality filtering happens before evaluation. Candidates and Naismith are compared on identical rows and folds. "
             "The retained model is then fitted to its eligible cohort for future inference.")
    st.caption("The app displays saved results. Uploading a route does not retrain or rerun cross-validation.")
    if not frames["folds"].empty:
        with st.expander("Inspect saved training and validation folds"):
            st.dataframe(frames["folds"], hide_index=True, width="stretch")
    st.subheader("Comparison with Naismith's Rule")
    st.latex(r"t_{\mathrm{Naismith}} = 20d + 0.03g\quad\text{minutes}")
    st.write("This project uses 3 mph plus 1 hour per 2,000 feet of ascent. Naismith makes no descent adjustment and excludes breaks. "
             "A regression becomes the primary predictor only when it beats whole-hike Naismith MAE across at least "
             f"{evaluation.get('minimum_validation_hikes', 10)} verified complete held-out hikes and has valid held-out segment predictions.")
    st.subheader("Sample size and limitations")
    st.write("Adjacent segments share terrain and conditions, so segment count overstates the number of independent examples. "
             "Filtering can leave partial hikes; their summed errors are not whole-hike errors. "
             "Small samples, missing hike types, GPS elevation noise, and routes outside training ranges limit generalization. "
             "The model does not account for weather, footing, fatigue, or pack weight.")
    if evaluation.get("model_c_status"):
        st.caption(evaluation["model_c_status"])
    if artifact.provenance:
        with st.expander("Saved model provenance"):
            st.json(artifact.provenance)


def main():
    st.set_page_config(page_title="Personal Hiking Time", page_icon="⛰", layout="wide")
    design()
    artifact, model_error = model_state()
    break_artifact = break_model_state()
    page, uploaded, hike_type, direction, breaks, clicked, include_breaks, override = sidebar(artifact, model_error)
    if model_error:
        st.error(f"Model loading failed: {model_error}")
    try:
        if page == PAGES[0]:
            route, contents, route_key = route_state(uploaded, hike_type, direction, artifact, break_artifact)
            predictor_page(route, contents, route_key, uploaded, hike_type, direction, breaks, clicked, artifact,
                           break_artifact, include_breaks, override)
        elif page == PAGES[1]:
            history_page(artifact, break_artifact)
        else:
            performance_page(artifact)
    except (ValueError, OSError, UnicodeError) as error:
        st.error(f"Could not complete this request: {error}")
        st.caption("For an upload, choose a valid GPX containing a single intended trail with coordinates and elevations. "
                   "For local data or model errors, check the existing files and README troubleshooting notes.")


if __name__ == "__main__":
    main()
