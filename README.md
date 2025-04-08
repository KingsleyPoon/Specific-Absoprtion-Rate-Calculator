# Introduction
This repository contains the Python code to calculate the _Specific Absorption Rate_ (SAR) of a nanoparticle (NP) solution measured for magnetic hyperthermia, developed for my doctoral thesis. The application has been specifically designed for the output file from the commercially available nB nanoScale Biomagnetics D5 series (G2, 1500W). This code can be adapted for other file formats.

# Notes
- The SAR value is calculated using the simple calorimetry equation and does not account for the heat capacity of the NPs itself. This is a valid approach given a sufficiently low concentration.
- If the SAR value output is _**NaN**_, this indicates that the specific function (_e.g.,_ exponential function) could not be fitted to the dataset. The SAR value from the linear function should be used instead.
- The error values provided are calculated based on the Euclidean distance between the datapoint and best fitting function. This should be used to determine the quality of the data fitting.
- Although not an explicit input into the calculation, the volume of solution can greatly impact the SAR and should not be ignored between samples.
