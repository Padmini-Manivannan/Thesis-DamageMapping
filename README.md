# Thesis-DamageMapping
Master Thesis - Urban Infrastructure Damage Detection and Mapping Using Sentinel 1.

The repository contains (part) of the code used for my MSc thesis on localising infrastructural damage mapping after natural disasters. 
To run this, as input, you will need a time series of Sentinel 1 dataset (available on https://scihub.copernicus.eu/dhus/#/home). These files have to be used in conjuction with RIPPL (Radar Interferometric Parallel Processing Lab), a software being developed in the Geosciences and Remote Sensing Faculty of TU Delft.

The idea is to study the affects of amplitude, persistent scatterers (PS) and coherence over a pre-disaster image stack and see how it changes after the disaster (ex. earthquake)  and to detect these changes effectively. The output is in the form of csv files which can be imported into Google Earth or ArcGIS. The final map looks like this:

<p align="center">
  <img width="720" src="OutputMapAmatrice.png?sanitize=true">
</p>

The image shows enlarged results of the methodology applied to all the parameters (amplitude, coherence) during the M6.2 Central Italy earthquake in 2016. The damaged points derived from this methodology are indicated by coloured circles ranging from red to yellow representing heavily damaged to slightly damaged areas. The Building Grading with coloured outlines of buildings including other symbols in the Crisis Information legend are from a reference (validation map) produced by Emergency Management Services (https://emergency.copernicus.eu/mapping/list-of-components/EMSR177). The image represents the north-western part of Amatrice where most of the historical buildings were situated. 
