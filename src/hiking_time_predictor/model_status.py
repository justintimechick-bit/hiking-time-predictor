"""Plain-language status of a hiker's personal models, derived only from saved fields.

The dashboard, My pace and Plan a hike all describe the models with these
presenters so the pages never contradict each other. Nothing here trains,
evaluates or changes which estimate is primary; that policy stays in
predictor.train_predictor and break_model.train_break_predictor.
"""
from dataclasses import dataclass

from .break_model import BASELINE
from .modeling import MODEL_LABELS, NAISMITH_LABEL
from .ui_format import MOVING_MODEL_NAMES, RULE_DEFINITION, RULE_NAME

COMPLETE_HIKE_HELP = ('A complete test hike is a whole recording that passes the GPS quality checks, '
                      'has no unresolved GPS gaps, and can be predicted while left out of training.')


@dataclass(frozen=True)
class MovingStatus:
    state: str            # none | too_few | unstable | learning | rule_wins | personal
    badge: str
    color: str
    icon: str
    headline: str
    detail: str = ''
    verified: int = 0
    minimum: int = 10
    personal_mae: float | None = None
    rule_mae: float | None = None
    rule_bias: float | None = None

    @property
    def progress(self) -> float:
        return min(self.verified / self.minimum, 1.0) if self.minimum else 0.0

    @property
    def show_progress(self) -> bool:
        return self.state in ('none', 'too_few', 'unstable', 'learning')


@dataclass(frozen=True)
class BreakStatus:
    state: str            # none | unavailable | early | rate | terrain
    badge: str
    color: str
    headline: str
    detail: str = ''
    minutes_per_mile: float | None = None
    mae: float | None = None


def _metric(rows, model):
    row = next((r for r in rows or [] if r.get('model') == model), None)
    return row


def selection_counts(hikes) -> tuple[int, int]:
    """(selected for training, total) for hikes that are not pending deletion."""
    usable = [h for h in hikes or [] if not h.get('deleting')]
    return sum(bool(h.get('training_enabled')) for h in usable), len(usable)


def moving_status(artifact, models, hikes=()) -> MovingStatus:
    """Which estimate Plan a hike uses for moving time, and how close the personal model is."""
    evaluation = getattr(artifact, 'evaluation', None) or {}
    minimum = int(evaluation.get('minimum_validation_hikes') or 10)
    verified = int(evaluation.get('verified_held_out_hikes') or 0)
    rule = _metric(evaluation.get('whole_hike_metrics'), NAISMITH_LABEL)
    rule_mae = rule.get('MAE_min') if rule else None
    rule_bias = rule.get('bias_pred_minus_actual_min') if rule else None
    rule_badge = dict(badge='Standard hiking rule', color='gray', icon=':material/straighten:')
    if not models:
        selected, total = selection_counts(hikes)
        if not total:
            headline = (f'Estimates use the {RULE_NAME} ({RULE_DEFINITION}) for now. '
                        'Add hikes you\'ve recorded and the app will learn your own pace.')
        elif not selected:
            headline = (f'You have {total} hike{"s" if total != 1 else ""}, but none is selected to learn from. '
                        'Choose some on My hikes, then train your pace model.')
        else:
            headline = (f'{selected} hike{"s are" if selected != 1 else " is"} ready. '
                        f'Train your pace model to see how your pace compares with the {RULE_NAME}.')
        return MovingStatus('none', headline=headline, minimum=minimum, **rule_badge)
    model_name = getattr(artifact, 'model_name', None)
    eligible = int(evaluation.get('eligible_hikes') or 0)
    if model_name is None:
        if eligible < 2:
            return MovingStatus(
                'too_few', headline=(f'Your pace model needs at least 2 hikes that pass the GPS quality checks '
                                     f'and found {eligible}. Estimates use the {RULE_NAME}.'),
                verified=verified, minimum=minimum, rule_mae=rule_mae, rule_bias=rule_bias, **rule_badge)
        return MovingStatus(
            'unstable', headline=('Your pace model gave unrealistic times on test hikes, so estimates use the '
                                  f'{RULE_NAME}. Add more hikes and retrain.'),
            verified=verified, minimum=minimum, rule_mae=rule_mae, rule_bias=rule_bias, **rule_badge)
    personal = _metric(evaluation.get('whole_hike_metrics'), MODEL_LABELS.get(model_name))
    personal_mae = personal.get('MAE_min') if personal else None
    common = dict(verified=verified, minimum=minimum, personal_mae=personal_mae,
                  rule_mae=rule_mae, rule_bias=rule_bias)
    comparison = ''
    if personal_mae is not None and rule_mae is not None:
        comparison = (f'On {personal.get("hikes", verified)} test hikes your pace model was off by about '
                      f'{personal_mae:.0f} min on average; the {RULE_NAME} by about {rule_mae:.0f} min.')
    if getattr(artifact, 'selected', False):
        return MovingStatus(
            'personal', badge='Your pace model', color='green', icon=':material/person_pin_circle:',
            headline=f'Estimates use your personal pace model, based on {MOVING_MODEL_NAMES[model_name]}.',
            detail=comparison, **common)
    if verified < minimum:
        remaining = minimum - verified
        detail = (f'Early result: {comparison}' if comparison else '')
        return MovingStatus(
            'learning', badge='Learning your pace', color='blue', icon=':material/trending_up:',
            headline=(f'Estimates use the {RULE_NAME} until your pace model has been tested on {minimum} '
                      f'complete hikes. {verified} so far, {remaining} to go. '
                      'Until then its estimate is shown as a preview.'),
            detail=detail, **common)
    return MovingStatus(
        'rule_wins', badge='Standard rule fits you', color='violet', icon=':material/balance:',
        headline=(f'Tested on {verified} of your complete hikes, the {RULE_NAME} was as accurate as your '
                  'pace model, so estimates keep using it. Retrain after more hikes to compare again.'),
        detail=comparison, **common)


def rule_bias_sentence(status: MovingStatus) -> str:
    """'On your test hikes the standard rule overestimated your time by 17 min on average.'"""
    if status.rule_bias is None or abs(status.rule_bias) < 1:
        return ''
    direction = 'overestimated' if status.rule_bias > 0 else 'underestimated'
    return f'On your test hikes the {RULE_NAME} {direction} your moving time by about {abs(status.rule_bias):.0f} min.'


def break_status(breaks, models) -> BreakStatus:
    """How break time is added to trip totals."""
    if not models:
        return BreakStatus('none', 'No break estimate yet', 'gray',
                           'Break estimates appear after you train with hikes that have complete timing.')
    rate = getattr(breaks, 'minutes_per_mile', None)
    hikes = int(getattr(breaks, 'training_hikes', 0) or 0)
    if rate is None:
        return BreakStatus('unavailable', 'No break estimate yet', 'gray',
                           'None of your selected hikes had complete enough timing to learn your breaks, '
                           'so trip totals show moving time only unless you add break time yourself.')
    evaluation = getattr(breaks, 'evaluation', None) or {}
    minimum = int(evaluation.get('minimum_evaluation_hikes') or 5)
    selected = getattr(breaks, 'selected_model', BASELINE)
    row = _metric(evaluation.get('metrics'), selected)
    mae = row.get('MAE_minutes') if row else None
    per_break = ''
    count, mean = getattr(breaks, 'breaks_per_mile', None), getattr(breaks, 'mean_duration_minutes', None)
    if count and mean:
        per_break = f' That is roughly one stop every {1 / count:.0f} mi, typically about {mean:.0f} min.'
    if hikes < minimum:
        return BreakStatus(
            'early', 'Personal break rate', 'blue',
            f'From {hikes} hike{"s" if hikes != 1 else ""} you stop about {rate:.1f} min per mile, '
            f'including short pauses.{per_break}',
            f'{minimum - hikes} more hike{"s" if minimum - hikes != 1 else ""} will let the app check how accurate this is.',
            minutes_per_mile=rate)
    accuracy = f'On test hikes it was off by about {mae:.0f} min per hike.' if mae is not None else ''
    if selected == BASELINE:
        return BreakStatus('rate', 'Personal break rate', 'green',
                           f'You stop about {rate:.1f} min per mile, based on {hikes} hikes.{per_break}',
                           accuracy, minutes_per_mile=rate, mae=mae)
    return BreakStatus('terrain', 'Terrain-aware breaks', 'green',
                       f'Break time adjusts for distance and climbing, learned from {hikes} hikes.{per_break}',
                       accuracy, minutes_per_mile=rate, mae=mae)


def is_stale(models, fingerprint) -> bool:
    """True when hikes or selections changed since the active pair was trained."""
    return bool(models) and any(row.get('dataset_fingerprint') != fingerprint for row in models)
