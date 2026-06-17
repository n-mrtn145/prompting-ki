import batchFile
import main
import os
from mistralai.client import Mistral
#Run file and job id will be printed into console


df = main.aufbereiten()

zero_shot_prompt = 'Ist folgende Aussage Hassrede? Antworte nur mit ja fuer Hassrede oder nein fuer nicht Hassrede: '

zero_shot_prompt_with_definition = f"""
Ist folgende Aussage Hassrede?
Definition:
HASSREDE = Inhalte, die Personen oder Gruppen aufgrund von Merkmalen wie Herkunft, Ethnie, Religion, Geschlecht, sexueller Orientierung oder ähnlichem beleidigen, entmenschlichen oder zu Gewalt aufrufen.

KEINE_HASSREDE = Kritik, Meinungsäußerung oder Beleidigungen ohne Bezug zu geschützten Gruppen.


 Antworte nur mit ja fuer Hassrede oder nein fuer nicht Hassrede: 
"""

single_shot_prompt = f"""
Aufgabe:
Klassifiziere den folgenden Kommentar als "HASSREDE" oder "KEINE_HASSREDE".

Antworte nur mit: ja wenn es Hassrede ist, oder nein wenn es keine ist

Beispiel:
Kommentar: "Frauen gehören nicht in Führungspositionen."
Antwort: HASSREDE

Jetzt Klassifiziere:
"""


last_few_shot_prompt_with_results_of_presentation= f"""Aufgabe:
Klassifiziere den folgenden Kommentar als "HASSREDE" oder "KEINE_HASSREDE".

Antworte nur mit: ja wenn es Hassrede ist, oder nein wenn es keine ist

Beispiele:

Kommentar: "Diese Ausländer nehmen uns alles weg."
Antwort: HASSREDE

Kommentar: "Ich finde diese Politik einfach nur schlecht umgesetzt."
Antwort: KEINE_HASSREDE

Kommentar: "Frauen gehören nicht in Führungspositionen."
Antwort: HASSREDE

Kommentar: "Du bist einfach unfähig."
Antwort: KEINE_HASSREDE

Kommentar: "Alle Muslime sind gefährlich."
Antwort: HASSREDE

Kommentar: "Das war die dümmste Entscheidung überhaupt."
Antwort: KEINE_HASSREDE

Kommentar: "Schwule sind krank."
Antwort: HASSREDE

Kommentar: "Ich hasse diesen Spieler, er spielt miserabel."
Antwort: KEINE_HASSREDE

Jetzt klassifiziere:
"""

few_shot_prompt_with_flipped_labels= f"""Aufgabe:
Klassifiziere den folgenden Kommentar als "HASSREDE" oder "KEINE_HASSREDE".

Antworte nur mit: ja wenn es Hassrede ist, oder nein wenn es keine ist

Beispiele:

Kommentar: "Diese Ausländer nehmen uns alles weg."
Antwort: KEINE_HASSREDE

Kommentar: "Ich finde diese Politik einfach nur schlecht umgesetzt."
Antwort: HASSREDE

Kommentar: "Frauen gehören nicht in Führungspositionen."
Antwort: KEINE_HASSREDE

Kommentar: "Du bist einfach unfähig."
Antwort: HASSREDE

Kommentar: "Alle Muslime sind gefährlich."
Antwort: KEINE_HASSREDE

Kommentar: "Das war die dümmste Entscheidung überhaupt."
Antwort: HASSREDE

Kommentar: "Schwule sind krank."
Antwort: KEINE_HASSREDE

Kommentar: "Ich hasse diesen Spieler, er spielt miserabel."
Antwort: HASSREDE

Jetzt klassifiziere:
"""

filename = f"batch_wrong_labels.jsonl"
batchFile.create_batch_file(df, zero_shot_prompt, filename)

batchFile.start_batch_file(filename)