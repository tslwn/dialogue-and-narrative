# Dialogue and Narrative

This repository holds the code written in the course of the
[Dialogue and Narrative](https://www.bristol.ac.uk/unit-programme-catalogue/UnitDetails.jsa;?ayrCode=23%2F24&unitCode=COMSM0023)
unit for my PhD in the [UKRI Centre for Doctoral Training in Interactive
Artificial Intelligence](https://www.bristol.ac.uk/cdt/interactive-ai/) at the
University of Bristol.

See also [interactive-ai-cdt](https://github.com/tslwn/interactive-ai-cdt).

## Setup

Create a Python virtual environment:

```bash
conda env create -f environment.yml
conda activate dn
```

Install/update Python dependencies:

```bash
conda install foo
conda env export --from-history > environment.yml
pip list --format=freeze > requirements.txt
```
