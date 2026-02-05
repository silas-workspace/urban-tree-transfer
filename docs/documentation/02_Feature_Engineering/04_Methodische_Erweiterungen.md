# Methodische Erweiterungen: Feature Engineering

Dieses Dokument beschreibt methodische Erweiterungen, die während der Phase 2 Feature Engineering diskutiert, aber aus Zeitgründen oder Scope-Beschränkungen nicht implementiert wurden.

---

## 1. CHM × Pflanzjahr: Wachstumsrate als Feature

### Beschreibung

Statt die absolute Baumhöhe (CHM) oder deren genus-normalisierte Varianten zu verwenden, könnte ein biologisch fundierteres Feature berechnet werden: die **relative Wachstumsrate** basierend auf Höhe und Baumalter.

### Mögliche Features

| Feature                     | Berechnung                                | Was es kodiert                                       |
| --------------------------- | ----------------------------------------- | ---------------------------------------------------- |
| **growth_rate**             | `CHM_1m / (current_year - plant_year)`    | Durchschnittliche Wachstumsrate in m/Jahr            |
| **CHM_residual**            | `CHM_1m - expected_height(genus, age)`    | Abweichung von erwarteter Höhe für Gattung und Alter |
| **height_age_ratio_zscore** | Z-Score von `growth_rate` innerhalb Genus | Relative Wuchsdynamik im Vergleich zu Artgenossen    |

### Biologische Begründung

Die absolute Baumhöhe hängt von vielen stadtspezifischen Faktoren ab:

- **Pflanzjahr/Alter**: Ältere Bäume sind höher (trivial)
- **Standort**: Park vs. Straße, Bodenverdichtung, Versiegelungsgrad
- **Pflege**: Schnittregime unterscheiden sich zwischen Städten
- **Klima/Boden**: Lokale Wachstumsbedingungen

Die **Wachstumsrate** hingegen ist stärker gattungsspezifisch:

- Schnellwüchsige Gattungen (Populus, Salix): 0.5–1.0 m/Jahr
- Mittel (Tilia, Acer): 0.3–0.5 m/Jahr
- Langsam (Quercus, Fagus): 0.2–0.4 m/Jahr

Ein Wachstumsrate-Feature würde den Alters-Confound entfernen und ein biologisch sinnvolleres Signal liefern, das potenziell besser zwischen Städten transferiert.

### Warum nicht implementiert?

1. **Hohe NaN-Rate bei `plant_year`**: Nicht alle Bäume im Kataster haben ein Pflanzjahr. Fehlende Werte würden das Feature für einen signifikanten Anteil der Daten unbrauchbar machen.
2. **Nicht-lineare Wachstumskurven**: Bäume wachsen nicht linear. Junge Bäume wachsen schneller, alte langsamer. Eine einfache Division `Höhe / Alter` ist nur eine grobe Approximation. Gattungsspezifische Wachstumsmodelle (z.B. Chapman-Richards-Kurve) wären nötig, was erheblichen Zusatzaufwand bedeutet.
3. **Datenqualität**: `plant_year` stammt aus Katasterdaten und kann Fehler enthalten (Nachpflanzungen, falsche Einträge). CHM ist per LiDAR/Stereo-Photogrammetrie gemessen und deutlich zuverlässiger.
4. **Scope Phase 2**: Feature Engineering war auf vorhandene Datenquellen (Sentinel-2, CHM, Kataster-Metadaten) fokussiert, nicht auf die Ableitung komplexer biologischer Modelle.

### Potenzial für Folgearbeit

- **Analyse der `plant_year`-Verfügbarkeit** pro Stadt und Genus als erster Schritt
- Einfache Version: `CHM_1m / max(current_year - plant_year, 1)` für Bäume mit bekanntem Pflanzjahr
- Fortgeschritten: Genus-spezifische Wachstumskurven aus Literatur oder aus den eigenen Daten ableiten
- Höhe-Alter-Residuen könnten besonders transferierbar sein, weil sie stadtunabhängige biologische Variation kodieren

---

## Zusammenfassung

| Erweiterung                      | Status              | Priorität für Folgearbeit                      |
| -------------------------------- | ------------------- | ---------------------------------------------- |
| CHM × Pflanzjahr (Wachstumsrate) | Nicht implementiert | Mittel (abhängig von plant_year Verfügbarkeit) |

---

_Letzte Aktualisierung: 2026-02-05_
