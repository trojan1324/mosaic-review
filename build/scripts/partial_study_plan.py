"""Keep independently runnable study contexts when Results source is absent elsewhere."""

from scripts.results_heading_context import main_results_payload


MISSING_RESULTS = "explicit_main_results_required"


def independently_runnable_studies(contexts):
    """Return existing contexts that have explicit admitted main Results text.

    The predicate is intentionally the same one the Results call uses.  A
    missing Results boundary is recorded per study; every other failure keeps
    its normal fail-closed behavior.
    """
    runnable = []
    skipped = []
    for context in contexts:
        try:
            main_results_payload(context)
        except ValueError as exc:
            if str(exc) != MISSING_RESULTS:
                raise
            skipped.append({"study_id": context["study"]["study_id"], "reason": MISSING_RESULTS})
            continue
        runnable.append(context)
    return runnable, skipped
