"""Required variable roles approved in DE-23; labels differ from historical storage terms."""

ROLE_LABELS = {
    "predictor": "independent variable",
    "moderator": "moderator",
    "mediator": "mediator",
    "outcome": "dependent variable",
    "covariate": "control variable",
}
CORE_ROLES = tuple(ROLE_LABELS)
