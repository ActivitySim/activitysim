from __future__ import annotations

import tempfile

from activitysim.cli.create import sha256_checksum
from activitysim.examples.external import registered_external_example


def test_external_download_unpack():
    """Test the external download mechanism, including unpacking assets"""
    t = tempfile.TemporaryDirectory()
    p = registered_external_example("estimation_example", t.name)
    assert p.joinpath("configs/settings.yaml").is_file()
    assert p.joinpath("data_sf/survey_data").is_dir()
    assert (
        sha256_checksum(p.joinpath("data_sf/households.csv"))
        == "7e782bb59c05a79110503a5f8173e3470b3969a40451af0271795d0d23909069"
    )
    assert (
        sha256_checksum(p.joinpath("data_sf/skims.omx"))
        == "579d6007266db3b055d0f9e4814004f4d5ccfae27a36e40f4881e3662bc3d3f1"
    )
    assert (
        sha256_checksum(p.joinpath("data_sf/land_use.csv"))
        == "83e1051fffa23ad1d6ec339fcb675532f0782c94ddf76d54020631d73bfca12f"
    )
    assert (
        sha256_checksum(p.joinpath("data_sf/persons.csv"))
        == "e24db9ac6c20592e672cd9fc4e8160528fe38a7c16cc54fe4920c516a29d732c"
    )
    assert (
        sha256_checksum(p.joinpath("data_sf/survey_data/survey_tours.csv"))
        == "633f734d964dcf25a20a4032a859982d861e1d327443d4f1bac64af9ef69cc7a"
    )


def test_external_download_basic():
    """Test the external download mechanism, including unpacking assets"""
    t = tempfile.TemporaryDirectory()
    p = registered_external_example("prototype_mtc", t.name)
    assert p.joinpath("configs/settings.yaml").is_file()
    assert p.joinpath("test/prototype_mtc_reference_pipeline.zip").is_file()
    assert (
        sha256_checksum(p.joinpath("test/prototype_mtc_reference_pipeline.zip"))
        == "394e5b403d4c61d5214493cefe161432db840ba4967c23c999d914178d43a1f0"
    )
