# Analyse von Pflanzen-Wachstum auf Basis von 3D-Punkwolken

## Abstract

Anwendung zum Analysieren von Pflanzen-Wachstum und weiteren Merkmalen des Wachstumsprozesses einer Pflanze. Es wird aus einer Reihe Biler einer Pflanze eine Aufnahme erzeugt. Aus der Punktwolke werden die Punkte die zur Pflanze gehören extrahiert. Die verbleibenden Punkte werden in Stamm und Blatt Punkte segmentiert zur weiteren Analyse. Die Anwendung soll mit möglichst wenig Bildern auskommen um den Datentransfer zu minimieren.

## Motivation

Im Rahmen einer Zusammenarbeit mit XY soll ein Anwendung geschaffen werden die die Analyse von Pflanzen-Wachstum mit nicht inversiven Maßnahmen ermöglicht. Wichtige Ziele hierbei sind die Datentransferrate zwischen Client und Server möglichst gering zu halten und trotzdem robuste Ergebnisse zu erziehlen.

## Problem Stellung

Die drei Grundlegenden Probleme die gelöst werden müssen sind die Erstellung der Punktwolken mit so wenig Bildern wie möglich, die Registrierung zweier Punktwolken um den Hintergrund zu entfernen und die Segmentierung einer Punktwolke in Stiel und Blätter. 

### Erstellung der Punktwolken

Vergleich existierender Lösungen um eine Anwendung zu finden die gute Ergebnisse bei wenig Bildern liefert.

### Registrierung

Große Herausforderung durch unbekannte Skalierung der Punktwolken. Die meisten Algorithmen beziehen die Skalierung nicht mit ein.

- ICP
- GO-ICP
- PointNetLK
- DCP

### Segmentierung

- Quellen?
- PointNet++

## Plant Net

### Registrirung

NN auf angelehnt an DCP das statt einer Rotations-Matrix und einem Translations-Vektor eine Transformations-Matrix

### Segmentierung

PointNet++ 

## Evalutation
