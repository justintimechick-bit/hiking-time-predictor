"""Authenticate provenance and bytes BEFORE joblib (which can execute code)."""
import hashlib
import hmac
import io
import json
import platform
import re
import warnings

import joblib
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

from .break_model import BreakArtifact
from .predictor import ModelArtifact
from .modeling import validate_estimator_schema


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def runtime():
    return {'python': '.'.join(platform.python_version_tuple()[:2]),
            'sklearn': sklearn.__version__, 'joblib': joblib.__version__}


class TrustedModelCodec:
    def __init__(self, key):
        if len(key.encode()) < 32:
            raise ValueError('Model signing key needs at least 32 bytes.')
        self.key = key.encode()

    def encode(self, artifact, *, user_id, model_id, kind, dataset_fingerprint, version):
        expected = ModelArtifact if kind == 'moving' else BreakArtifact
        if type(artifact) is not expected:
            raise ValueError('Only application-trained model artifacts can be signed.')
        buffer = io.BytesIO()
        joblib.dump(artifact, buffer)
        contents = buffer.getvalue()
        metadata = {'schema': 1, 'owner': user_id, 'model_id': model_id, 'kind': kind,
                    'dataset_fingerprint': dataset_fingerprint, 'version': version,
                    'sha256': hashlib.sha256(contents).hexdigest(), 'runtime': runtime()}
        metadata['signature'] = hmac.new(self.key, canonical(metadata), hashlib.sha256).hexdigest()
        return contents, metadata

    def decode(self, contents, row, user_id):
        if not isinstance(row.get('provenance'), dict):
            raise ValueError('Model provenance is missing. Retrain your models.')
        metadata = dict(row['provenance'])
        signature = metadata.pop('signature', '')
        if not isinstance(signature, str) or not re.fullmatch(r'[0-9a-f]{64}', signature):
            raise ValueError('Model signature invalid. Retrain using this application.')
        actual = hmac.new(self.key, canonical(metadata), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, actual):
            raise ValueError('Model signature invalid. Retrain using this application.')
        expected = {'schema': 1, 'owner': user_id, 'model_id': row['id'], 'kind': row['model_type'],
                    'dataset_fingerprint': row['dataset_fingerprint'], 'version': row['version'],
                    'sha256': row['artifact_hash'], 'runtime': runtime()}
        if metadata != expected or hashlib.sha256(contents).hexdigest() != metadata['sha256']:
            raise ValueError('Model integrity, owner, or runtime mismatch. Retrain your models.')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', InconsistentVersionWarning)
                artifact = joblib.load(io.BytesIO(contents))
        except Exception:
            raise ValueError('The verified model is incompatible with this application. Retrain your models.') from None
        expected_type = ModelArtifact if row['model_type'] == 'moving' else BreakArtifact
        if type(artifact) is not expected_type:
            raise ValueError('Unsupported application model.')
        if isinstance(artifact, ModelArtifact) and artifact.estimator is not None:
            validate_estimator_schema(artifact.estimator, artifact.model_name)
        return artifact
