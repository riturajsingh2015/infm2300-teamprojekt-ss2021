# 4 Gewinnt

Dieses Notebook beschäftigt sich dem Spiel Connect 4, welches im deutschen Sprachraum auch als 4 Gewinnt bekannt ist. 
Ziel des Spiels ist es 4 Steine seiner eigenen Farbe in horizontaler, vertikaler oder diagonaler Richtung in einem Spielbrett zu Platzieren.

## Datensatz
Zu diesem Spiel wurde ein umfangreicher Datensatz erhoben. Der Zustand eines Spiels wurde festgehalten, während beide Spieler je 4 Steine platziert hatten. Zu dieser gespielten Partie wurde dann das Endergebnis aus Sicht des ersten Spielers, also Sieg, Niederlage bzw. Unentschieden vermerkt.

Quelle des Datzensatzen: http://archive.ics.uci.edu/ml/datasets/Connect-4

### Eigenschaften und Aufbau
Das Spielbrett ist aufgebaut als Gitter und ist folgendermaßen unterteilt:
- **7 Felder horizontal**, beschriftet von **a bis g**
- **6 Felder vertikal**, beschriftet von **1 bis 6**
Das ergibt insgesamt 42 mögliche Steinpositionen. Zusammen mit Endergebnis enthält der Datensatz `43 Attribute`. Spieler Eins wird als **x** und Spieler 2 als **o** gekennzeichnet. **b** bedeutet das dort kein Stein liegt.

Der Datensatz ist wie folgt aufgebaut:

- 1. a1: {x,o,b} (x = Stein Spieler 1, o = Stein Spieler 2, b = Leeres Feld)
- 2. a2: {x,o,b}
- 3. a3: {x,o,b}
- 4. a4: {x,o,b}
- 5. ...
- 41. g5: {x,o,b}
- 42. g6: {x,o,b}
- 43. Class: {win,loss,draw} (Sieg Spieler 1, Niederlage Spieler 1, Unentschieden)

Der Datensatz beinhaltet insgesamt **67557** Einträge

# Zielstellung
Mit Hilfe von Maschine Learning Verfahren soll eine Möglichkeit geschaffen werden, den Ausgang des Spiel in dem 9. Zug vorhersagen zu können. Da das Ziel aus 3 konkreten, voneinander getrennten Zuständen besteht, spricht man hier auch von einer Klassifikationsaufgabe.

