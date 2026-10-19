# COV_Projekt-B
# Text-Guided Medical Image Segmentation

This project implements text-guided semantic segmentation on medical images (e.g. CT / X-ray),
based on ideas from TGSAM-2 and TextDiff.

## Project Goal

- Use OpenCV for image preprocessing (normalization, contrast enhancement, ROI extraction).
- Train a baseline segmentation model (U-Net) using only images.
- Train a text-guided segmentation model that fuses image and text prompts.
- Compare performance with and without text guidance using Dice, IoU, and Hausdorff distance.

## Streamlit Interface

![Streamlit UI](assets/streamlit_ui.png)

## Dataset and Models
The datasets and pre-trained models can be downloaded at [Google Drive](https://drive.google.com/file/d/1r_iNLyr6Vlf01WLAL6nmAg0v6y9DJg6f/view?usp=sharing)

## Klarstellung zu Rückfragen bei Präsentation
Es wurde in Frage gestellt ob das U-Net auch nur anhand vom Bild wissen kann, dass bspw nur ein Lungenflügel segmentiert werden soll, was natürlich der Fall ist. Es gibt zahlreiche Beispiele in denen auch das U-Net nur eine Seite segmentiert, weil die Information auch im Bild vorhanden ist. Persönlich könnte ich es anhand vom Bild auch nicht erkennen, aber Ärzte könnten unterscheiden welcher Bereich wirklich entzündet ist und dazu kann grundsätzlich auch das U-Net in der Lage sein. Die Text Prompts sollen die Segmentierung nur unterstützen, indem sie beschreiben was im Bild ohnehin zu sehen ist. Die Text Annotationen wurden sorgfältig an das Bild angepasst und sind nicht willkürlich gewählt, sodass das U-Net keine Chance hätte zu wissen warum etwas nicht segmentiert werden soll.