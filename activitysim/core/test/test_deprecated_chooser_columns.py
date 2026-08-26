from __future__ import annotations

import pytest

from activitysim.abm.models.school_escorting import SchoolEscortSettings
from activitysim.abm.models.util.vectorize_tour_scheduling import TourSchedulingSettings
from activitysim.core.configuration.logit import (
    TourLocationComponentSettings,
    TourModeComponentSettings,
)


@pytest.mark.parametrize(
    "settings_class, extra_settings",
    [
        (
            TourLocationComponentSettings,
            dict(
                SPEC="spec.csv",
                SAMPLE_SPEC="sample_spec.csv",
                SAMPLE_SIZE=10,
                CHOOSER_ORIG_COL_NAME="home_zone_id",
                ALT_DEST_COL_NAME="alt_dest",
            ),
        ),
        (TourSchedulingSettings, dict(SPEC="spec.csv")),
        (SchoolEscortSettings, dict(ALTS="alts.csv")),
    ],
)
def test_simulate_chooser_columns_is_deprecated(settings_class, extra_settings):
    with pytest.warns(DeprecationWarning, match="SIMULATE_CHOOSER_COLUMNS"):
        settings = settings_class.model_validate(
            dict(SIMULATE_CHOOSER_COLUMNS=["home_zone_id"], **extra_settings)
        )
    # the deprecated setting is ignored
    assert settings.SIMULATE_CHOOSER_COLUMNS is None


def test_logsum_chooser_columns_is_deprecated():
    with pytest.warns(DeprecationWarning, match="LOGSUM_CHOOSER_COLUMNS"):
        settings = TourModeComponentSettings.model_validate(
            dict(SPEC="spec.csv", LOGSUM_CHOOSER_COLUMNS=["age"])
        )
    # the deprecated setting is ignored
    assert settings.LOGSUM_CHOOSER_COLUMNS is None


def test_no_warning_when_settings_omitted():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        TourModeComponentSettings.model_validate(dict(SPEC="spec.csv"))
        TourSchedulingSettings.model_validate(dict(SPEC="spec.csv"))
