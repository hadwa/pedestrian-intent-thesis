# Exp074: Exp072 Plus Road-Relative YOLOP

Controlled PIE 16-frame, TTE 30--60 extension of Exp072. The only predictive
change is 60 fixed early/late/delta summaries of the authenticated Exp063
20-channel road-relative sequence. The classifier, cohorts, seeds, and fixed
0.5 threshold remain Exp072-identical.

Train/validation YOLOP outputs are reused. Set03 road-relative extraction is
forbidden unless the frozen validation gate advances.
