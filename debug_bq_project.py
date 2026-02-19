from google.cloud import bigquery
client = bigquery.Client()
print("client.project =", client.project)


# python .\debug_bq_project.py