"""My pace: personal model status, training, and the evaluation behind it."""
import pandas as pd
import streamlit as st

from .breaks import NON_MOVING_TARGET
from .model_status import COMPLETE_HIKE_HELP, break_status, is_stale, moving_status, rule_bias_sentence, selection_counts
from .modeling import MODEL_LABELS, NAISMITH_LABEL
from .personal_hiking import snapshot_fingerprint, train_personal_models
from .ui_format import (BREAK_METHOD_NAMES, HIKE_TYPE_LABELS, METRIC_MODEL_NAMES, RULE_NAME, feet, friendly_date,
                        miles, relative_day)
from .ui_nav import flash, nav_button
from .web_components import performance_page


def _training_summary(moving, breaks):
    hikes = moving.evaluation.get('eligible_hikes', 0)
    method = 'your pace model' if moving.selected else f'the {RULE_NAME}'
    text = f'Trained on {hikes} hike{"s" if hikes != 1 else ""}. Estimates use {method}.'
    if breaks.minutes_per_mile is not None:
        text += f' Breaks: about {breaks.minutes_per_mile:.1f} min per mile.'
    return text


def run_training(repo, codec, selected):
    with st.status('Training your pace model…', expanded=True) as status:
        st.write(f'Reading {selected} selected hike{"s" if selected != 1 else ""} and testing on whole hikes left out of training.')
        try:
            moving, breaks, _ = train_personal_models(repo, codec)
        except ValueError as error:
            status.update(label="Training didn't finish", state='error', expanded=True)
            message = str(error)
            if 'History changed while training' in message:
                message += ' Select Retrain again.'
            st.warning(message, icon=':material/warning:')
            return
        st.write('Saved and verified your new models.')
        status.update(label='Pace model trained', state='complete', expanded=False)
    st.session_state.pop('_models', None)
    flash(_training_summary(moving, breaks), icon=':material/model_training:')
    st.rerun()


def _metric_table(rows, names, value_key, in_use):
    table = pd.DataFrame([{
        'Method': names.get(row['model'], row['model']) + (' · in use' if row['model'] == in_use else ''),
        'Test hikes': row.get('hikes'), 'Average miss': row.get(value_key)} for row in rows])
    st.dataframe(table, hide_index=True, width='stretch', column_config={
        'Method': st.column_config.TextColumn(width='medium'),
        'Test hikes': st.column_config.NumberColumn(width='small'),
        'Average miss': st.column_config.NumberColumn(format='%.1f min', help='Average difference from your actual time per hike')})


def _moving_card(artifact, mstatus):
    with st.container(border=True, height='stretch'):
        st.markdown('**Moving time**')
        st.badge(mstatus.badge, icon=mstatus.icon, color=mstatus.color)
        rows = artifact.evaluation.get('whole_hike_metrics') or []
        if rows:
            in_use = MODEL_LABELS.get(artifact.model_name) if artifact.selected else NAISMITH_LABEL
            _metric_table(rows, METRIC_MODEL_NAMES, 'MAE_min', in_use)
            note = rule_bias_sentence(mstatus)
            if note:
                st.caption(note)
        else:
            st.caption('Accuracy on your own hikes appears after you train with at least 2 complete hikes.')
        ranges = artifact.route_ranges or {}
        if ranges.get('distance_miles') and ranges.get('elevation_gain_ft'):
            (dlo, dhi), (glo, ghi) = ranges['distance_miles'], ranges['elevation_gain_ft']
            st.caption(f'Learned from hikes of {miles(dlo)} to {miles(dhi)} with {feet(glo)} to {feet(ghi)} of climbing. '
                       'Routes outside that range are rougher estimates.')
        if len(artifact.hike_types or []) == 1:
            st.caption(f'Trained on {HIKE_TYPE_LABELS.get(artifact.hike_types[0], artifact.hike_types[0]).lower()} '
                       'recordings only.')


def _break_card(breaks, bstatus, models):
    with st.container(border=True, height='stretch'):
        st.markdown('**Breaks**')
        st.badge(bstatus.badge, icon=':material/local_cafe:', color=bstatus.color)
        st.write(bstatus.headline)
        if bstatus.detail:
            st.caption(bstatus.detail)
        rows = breaks.evaluation.get('metrics') or []
        if rows:
            _metric_table(rows, BREAK_METHOD_NAMES, 'MAE_minutes', breaks.selected_model)
        if models:
            st.caption('Break time counts all stopped time, including short pauses and unclear GPS time, '
                       'so trip totals aren\'t too optimistic.')


def _history(versions, tz):
    frame = pd.DataFrame(versions)
    runs = []
    for version, group in frame.groupby('version', sort=False):
        by_type = {row['model_type']: row for row in group.to_dict('records')}
        moving, breaks = by_type.get('moving', {}), by_type.get('break', {})
        any_row = moving or breaks
        runs.append({'Trained': f"{friendly_date(any_row.get('trained_at'), tz)} ({relative_day(any_row.get('trained_at'), tz=tz)})",
                     'Hikes (moving)': moving.get('training_hike_count'),
                     'Hikes (breaks)': breaks.get('training_hike_count'),
                     'Segments': moving.get('training_segment_count'),
                     'In use': bool(group.is_active.all())})
    st.dataframe(pd.DataFrame(runs), hide_index=True, width='stretch',
                 column_config={'In use': st.column_config.CheckboxColumn()})


def models_page(repo, codec, artifact, breaks, models, *, model_error=None):
    tz = st.context.timezone
    st.title('My pace')
    st.caption('The app learns how fast you move and how long you rest from the hikes you select. To check its '
               'accuracy fairly, it predicts whole hikes that were left out of training.')
    hikes = repo.list_hikes()
    if model_error:
        selected, _ = selection_counts(hikes)
        with st.container(border=True, key='pace_status'):
            st.error("We couldn't verify your saved pace model, so it isn't being used. Retrain to rebuild it. "
                     'Your hikes and settings are safe.', icon=':material/gpp_bad:')
            st.caption(model_error)
            if st.button('Retrain my pace model', key='train_models', type='primary', icon=':material/model_training:',
                         disabled=not selected):
                run_training(repo, codec, selected)
        return
    selected, total = selection_counts(hikes)
    stale = is_stale(models, snapshot_fingerprint(repo.snapshot())) if models else False
    mstatus = moving_status(artifact, models, hikes)
    bstatus = break_status(breaks, models)
    with st.container(border=True, key='pace_status'):
        st.badge(mstatus.badge, icon=mstatus.icon, color=mstatus.color)
        st.markdown(mstatus.headline)
        if mstatus.show_progress and models:
            st.progress(mstatus.progress, text=f'{mstatus.verified} of {mstatus.minimum} complete test hikes')
            st.caption(COMPLETE_HIKE_HELP)
        if mstatus.detail:
            st.caption(mstatus.detail)
        if stale:
            st.warning('Your hikes or selections changed since you last trained. Retrain so estimates include them.',
                       icon=':material/update:')
        elif models:
            st.caption(f":material/check: Up to date with your hikes · trained {relative_day(models[0]['trained_at'], tz=tz)}")
        if models and breaks.evaluation.get('target_definition') != NON_MOVING_TARGET:
            st.info('Break estimates now count all stopped time. Retrain to update your saved model.',
                    icon=':material/info:')
        with st.container(horizontal=True, vertical_alignment='center', gap='small'):
            clicked = st.button('Retrain my pace model' if models else 'Train my pace model', key='train_models',
                                type='primary' if (not models or stale) else 'secondary',
                                icon=':material/model_training:', disabled=not selected,
                                help=None if selected else 'Select at least one hike on My hikes first.')
            st.caption(f'{selected} of {total} hike{"s" if total != 1 else ""} selected')
            nav_button('Change selection', 'pace_change_selection', 'hikes', type='tertiary', icon=':material/checklist:')
        if clicked:
            run_training(repo, codec, selected)
    left, right = st.columns(2, gap='medium')
    with left:
        _moving_card(artifact, mstatus)
    with right:
        _break_card(breaks, bstatus, models)
    versions = repo.models()
    if versions:
        runs = len({row['version'] for row in versions})
        with st.expander(f"Training history ({runs} run{'s' if runs != 1 else ''})", icon=':material/history:'):
            _history(versions, tz)
    if models:
        nav_button('Plan a hike with your pace model', 'pace_plan', 'plan', icon=':material/route:')
    with st.expander('How your pace model works (technical details)', icon=':material/science:'):
        performance_page(artifact, embedded=True)
