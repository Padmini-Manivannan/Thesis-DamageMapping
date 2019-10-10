# Thesis-DamageMapping
Master Thesis - Urban Infrastructure Damage Detection and Mapping Using Sentinel 1.

The repository contains (part) of the code used for my MSc thesis on localising infrastructural damage mapping after natural disasters. 
To run this, as input, you will need a time series of Sentinel 1 dataset (avaiable on https://scihub.copernicus.eu/dhus/#/home). These files
have to be used in conjuction with RIPPL (Radar Interferometric Parallel Processing Lab), a software developed in the Geosciences and
Remote Sensing Faculty of TU Delft.

The idea is to study the affects of amplitude and coherence over a pre-disaster image stack and see how it changes after the disaster (ex. earthquake) 
and to detect these changes effectively. The output is in the form of csv files which can be imported into Google Earth or ArcGIS. The final map looks like this:

<p align="center">
  <img width="720" src="OutputMapAmatrice.png?sanitize=true">
</p>
