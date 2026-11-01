"""Preserve complete parsed answers and structural failures without repair."""
from jsonschema import Draft202012Validator


def answer_outcome(schema, answer):
    errors = list(Draft202012Validator(schema).iter_errors(answer))
    if not errors:
        return {'status': 'response_available', 'answer': answer}
    # Diagnostic paths and keywords suffice; exception messages may copy source
    # text. The entire original proposal remains separately available for audit.
    return {'status': 'invalid_response', 'answer': None, 'parsed_answer': answer,
        'schema_validation': {'valid': False, 'errors': [
            {'keyword': e.validator, 'answer_path': list(e.absolute_path),
             'schema_path': list(e.absolute_schema_path)} for e in errors]}}
