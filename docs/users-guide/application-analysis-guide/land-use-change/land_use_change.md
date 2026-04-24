# Land Use Change Guide

## Introduction

Many contemporary urban planners are encouraging developers to build denser housing, particularly around transit stops. Naturally, planners will want to gauge what the impact of such a development would be on their jurisdiction's transportation system, particularly regarding metrics such as VMT (and subsequently greenhouse gas emission) and transit boardings (and subsequently farebox revenue). To demonstrate this, we will be analyzing a hypothetical development in the San Diego Region. The particular development will add 2000 households and 1000 retail jobs in the vicinity of the Grossmont Station on the Green and Orange Lines of San Diego's light rail system, where there is an existing auto-oriented shopping mall. The guide will show how to make changes to the ActivitySim inputs, how to run the test, and how to calculate some of the key metrics such as VMT and changes in mode share.

**NOTE: The example provided is a hypothetical project that demonstrates how one would use ActivitySim to model the effects of a land use change and does not necessarily reflect any real planned developments.**

## Setting Up the Test

Three input files need to be changed in order to run this test: the land use file (landuse.csv) and the files defining the synthetic population (households.csv and persons.csv). While updating the land use file may seem very straightforward, it is very easy to overlook some necessary changes that could result in the model understating the impact of the change. A modeler doesn't need to just edit the household and employment fields in the study area--they also need to edit any field derived from those fields. For example, every new household will have at least one person in it, so the population field will need to be updated as well (along with population density if that is present). If the total population is to be kept the same, households will need to be removed outside of the study area as well.

Because Activity-based models use synthetic populations, those input files will need to be updated to reflect the different distribution in the population. There are multiple ways that this could be done. Ideally this would involve rerunning the population synthesizer, though if the total population of the modeling area were to be kept constant, households living outside of the study area could randomly be reassigned into the study area for a simpler analysis.

## Running the Test

## Analyzing the Results