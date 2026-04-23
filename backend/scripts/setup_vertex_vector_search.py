"""
Setup script to provision a Google Cloud Vertex AI Vector Search Index and Endpoint.

This script creates an Index optimized for STREAM_UPDATE (near real-time indexing)
and deploys it to an IndexEndpoint. 

Usage:
  python scripts/setup_vertex_vector_search.py

WARNING: Provisioning an Index and deploying it can take 45-60 minutes on GCP.
"""
import os
import argparse
from google.cloud import aiplatform
from dotenv import load_dotenv

load_dotenv()

def setup_vector_search(project_id: str, location: str, index_name: str, dimensions: int):
    """Create a Vector Search Index and deploy it to an Endpoint."""
    print(f"Initializing Vertex AI for project '{project_id}' in '{location}'...")
    aiplatform.init(project=project_id, location=location)

    endpoint_name = f"{index_name}-endpoint"

    # 1. Check if Index exists
    print(f"Checking for existing Index named '{index_name}'...")
    existing_indexes = aiplatform.MatchingEngineIndex.list(
        filter=f"display_name={index_name}"
    )
    
    if existing_indexes:
        print(f"Index '{index_name}' already exists (ID: {existing_indexes[0].name}).")
        my_index = existing_indexes[0]
    else:
        print(f"Creating new Index '{index_name}' (this takes metadata into account)...")
        # For our use case (Semantic Search with tenant isolation), we need STREAM_UPDATE
        my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=index_name,
            dimensions=dimensions,
            approximate_neighbors_count=150,
            distance_measure_type="COSINE_DISTANCE",
            leaf_node_embedding_count=500,
            leaf_nodes_to_search_percent=7,
            description="Document Chatbot Vector Search Index",
            index_update_method="STREAM_UPDATE", # VERY IMPORTANT for real-time updates
        )
        print(f"Successfully created Index: {my_index.name}")

    # 2. Check if Endpoint exists
    print(f"Checking for existing IndexEndpoint named '{endpoint_name}'...")
    existing_endpoints = aiplatform.MatchingEngineIndexEndpoint.list(
        filter=f"display_name={endpoint_name}"
    )

    if existing_endpoints:
        print(f"IndexEndpoint '{endpoint_name}' already exists (ID: {existing_endpoints[0].name}).")
        my_index_endpoint = existing_endpoints[0]
    else:
        print(f"Creating new IndexEndpoint '{endpoint_name}'...")
        my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=endpoint_name,
            public_endpoint_enabled=True,
            description="Endpoint for Document Chatbot Vector Search",
        )
        print(f"Successfully created IndexEndpoint: {my_index_endpoint.name}")

    # 3. Deploy Index to Endpoint
    deployed_indexes = my_index_endpoint.deployed_indexes
    is_deployed = any(di.index == my_index.resource_name for di in deployed_indexes)

    if is_deployed:
        print("Index is already deployed to the Endpoint. Nothing to do.")
        # Find the deployed index ID
        deployed_id = next(di.id for di in deployed_indexes if di.index == my_index.resource_name)
    else:
        print(f"Deploying Index '{my_index.name}' to Endpoint '{my_index_endpoint.name}'...")
        print("WARNING: This process can take 45-60 minutes. Please be patient!")
        
        deployed_id = f"chatbot_deployed_{index_name.replace('-', '_')}"
        
        my_index_endpoint = my_index_endpoint.deploy_index(
            index=my_index,
            deployed_index_id=deployed_id,
            display_name=f"Deployed {index_name}",
            # Use minimal compute for a web app backend (can be scaled later)
            machine_type="e2-standard-2",
            min_replica_count=1,
            max_replica_count=1,
        )
        print("Index deployment complete!")

    print("\n" + "="*50)
    print("SETUP SUCCESSFUL. ADD THESE TO YOUR .ENV FILE IN PRODUCTION:")
    print("="*50)
    print(f"VERTEX_VECTOR_LOCATION={location}")
    print(f"VERTEX_VECTOR_INDEX_ID={my_index.name.split('/')[-1]}")
    print(f"VERTEX_VECTOR_ENDPOINT_ID={my_index_endpoint.name.split('/')[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup Vertex AI Vector Search")
    parser.add_argument("--project", type=str, required=False, help="Google Cloud Project ID")
    parser.add_argument("--location", type=str, default="us-central1", help="GCP Region (e.g., us-central1)")
    parser.add_argument("--name", type=str, default="chatbot-vectors", help="Name for the Index")
    parser.add_argument("--dimensions", type=int, default=3072, help="Vector dimensions (e.g., 3072 for gemini-embedding-001)")
    
    args = parser.parse_args()
    
    project = args.project or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        print("ERROR: Must provide --project or set GOOGLE_CLOUD_PROJECT in .env")
        exit(1)
        
    setup_vector_search(project, args.location, args.name, args.dimensions)
