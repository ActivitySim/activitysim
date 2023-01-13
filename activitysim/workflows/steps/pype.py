from pypyr.steps.pype import run_step as _run_step


def run_step(context):
    pype = context.get("pype")
    if isinstance(pype, str):
        pype = {"name": pype}
    pype["loader"] = "activitysim.workflows"
    context["pype"] = pype
    return _run_step(context)
