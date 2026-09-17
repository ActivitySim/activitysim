===========
ActivitySim
===========

The mission of the ActivitySim project is to create and maintain advanced, open-source,
activity-based travel behavior modeling software based on best software development
practices for distribution at no charge to the public.

The ActivitySim project is led by a consortium of Metropolitan Planning Organizations
(MPOs) and other transportation planning agencies, which provides technical direction
and resources to support project development. New member agencies are welcome to join
the consortium. All member agencies help make decisions about development priorities
and benefit from contributions of other agency partners.  Additional information about
the development and management of the ActivitySim is on the `project site <http://www.activitysim.org>`__.

.. grid:: 2

    .. grid-item-card::

        :fa:`book` |nbsp| |nbsp| :ref:`User's Guide <userguide>`

        ^^^

        Start here to learn about using ActivitySim, including how to install,
        the software, and how to configure and run models.

    .. grid-item-card::

        :fa:`terminal` |nbsp| |nbsp| :ref:`Developer's Guide <devguide>`

        ^^^

        Start here to learn about developing ActivitySim, including creating
        model components, or changing the codebase.

    .. grid-item-card::

        :fa:`thumbs-up` |nbsp| |nbsp| Consortium Supported Examples

        ^^^

        The ActivitySim consortium actively supports two example models, each of which
        can be used as a starting point (i.e. a "donor model") for new implementations.

        - The `MTC Example <https://github.com/ActivitySim/activitysim-prototype-mtc>`__,
          our one-zone system prototype. This example is originally based on MTC's Travel Model One (TM1),
          but has evolved to be a slightly different model.
        - The `SANDAG Example <https://github.com/ActivitySim/sandag-abm3-example>`__,
          our two-zone system model. Some effort has been made to keep it aligned
          with SANDAG's model, but it is not an exact copy of SANDAG's production model.

    .. grid-item-card::

        :fa:`square-arrow-up-right` |nbsp| |nbsp| Member Agency Models

        ^^^

        Several consortium member agencies have open-sourced their ActivitySim
        implementations. These open models may or may not be complete calibrated
        tools. Unless clearly marked, users should not assume that mlinked models are
        "official" implementations used for policy analysis; public agencies often
        publish in-progress model development to foster collaboration and transparency.
        Contact the agencies directly with questions.

        - `Puget Sound Regional Commission <https://github.com/psrc/psrc_activitysim>`__ (Seattle)
        - `Atlanta Regional Commission <https://github.com/atlregional/arc-activitysim>`__
        - `Metropolitan Council <https://github.com/Metropolitan-Council/metc-asim-model/tree/main/source/activitysim>`__ (Minneapolis-St. Paul)
        - `Oregon Modeling Statewide Collaborative <https://github.com/OrMSC/SimOR>`__

.. toctree::
   :hidden:

   users-guide/index
   dev-guide/index

.. |nbsp| unicode:: 0xA0
   :trim:
