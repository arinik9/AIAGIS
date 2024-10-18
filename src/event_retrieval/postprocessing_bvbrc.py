
import pandas as pd
import csv

def postprocessing_bvbrc(raw_events_filepath, processed_events_filepath, new_processed_events_filepath):
    df_raw_events = pd.read_csv(raw_events_filepath, sep=";", keep_default_na=False)
    df_processed_events = pd.read_csv(processed_events_filepath, sep=";", keep_default_na=False)

    id2GenbankAccessions = dict(zip(df_raw_events["id"], df_raw_events["GenBank Accessions"]))
    id2GenomeID = dict(zip(df_raw_events["id"], df_raw_events["Genome ID"]))
    id2GenomeName = dict(zip(df_raw_events["id"], df_raw_events["Genome Name"]))
    id2SegGenomeID = dict(zip(df_raw_events["id"], df_raw_events["Segment2GenomeID"]))
    id2Publication = dict(zip(df_raw_events["id"], df_raw_events["Publication"]))
    id2BioProjectAccession = dict(zip(df_raw_events["id"], df_raw_events["BioProject Accession"]))
    id2SequencingPlatform = dict(zip(df_raw_events["id"], df_raw_events["Sequencing Platform"]))
    id2AssemblyMethod = dict(zip(df_raw_events["id"], df_raw_events["Assembly Method"]))
    id2IsolationSource = dict(zip(df_raw_events["id"], df_raw_events["Isolation Source"]))

    df_processed_events["GenBank Accessions"] = df_processed_events["article_id"].apply(lambda x: id2GenbankAccessions[x])
    df_processed_events["Genome ID"] = df_processed_events["article_id"].apply(lambda x: id2GenomeID[x])
    df_processed_events["Genome Name"] = df_processed_events["article_id"].apply(lambda x: id2GenomeName[x])
    df_processed_events["Segment2GenomeID"] = df_processed_events["article_id"].apply(lambda x: id2SegGenomeID[x])
    df_processed_events["Publication"] = df_processed_events["article_id"].apply(lambda x: id2Publication[x])
    df_processed_events["BioProject Accession"] = df_processed_events["article_id"].apply(lambda x: id2BioProjectAccession[x])
    df_processed_events["Sequencing Platform"] = df_processed_events["article_id"].apply(lambda x: id2SequencingPlatform[x])
    df_processed_events["Assembly Method"] = df_processed_events["article_id"].apply(lambda x: id2AssemblyMethod[x])
    df_processed_events["Isolation Source"] = df_processed_events["article_id"].apply(lambda x: id2IsolationSource[x])

    df_processed_events.to_csv(new_processed_events_filepath, sep=";", quoting=csv.QUOTE_NONNUMERIC)
