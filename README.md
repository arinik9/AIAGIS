# AVIAGIS
an improved AVian InfluenzA surveillance dataset with Genome Isolate Sequences

A set of Python scripts to process raw input event files and perform a network inference task. Please run the following scripts in the same order:
* main_genome.py
* main_event_retrieval.py
* main_event_imprecise_serovar.py
* main_comp_imprecise_serovar.py
* main_comp.py
* main_inference.py

TODOs:
* better handling the date information in BV-BRC (e.g. 01/13/21 and 13-JAN-21) >> main_genome.py
* combine `in/map_shapefiles` and `data/map_shapefiles`
* In the inference task, we need a single cascade for the data paper (not multiple cascades)
* Add the final plots of the data paper in the main script >> main_inference.py 

