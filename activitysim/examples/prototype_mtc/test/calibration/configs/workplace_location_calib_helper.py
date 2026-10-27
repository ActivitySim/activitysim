from __future__ import annotations


def workplace_distance_share(context, minimum, maximum):
    """Return the modeled worker share in a DIST skim interval."""
    persons = context["persons"]
    workers = persons[persons["workplace_zone_id"] > 0]
    distances = context["skim_dict"].lookup(
        workers["home_zone_id"],
        workers["workplace_zone_id"],
        "DIST",
    )
    return ((distances >= minimum) & (distances < maximum)).mean()
