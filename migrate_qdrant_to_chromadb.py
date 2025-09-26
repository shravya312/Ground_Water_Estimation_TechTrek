#!/usr/bin/env python3
"""
Migrate data from Qdrant to ChromaDB
Handles 2 lakh+ records efficiently
"""

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
import time

load_dotenv()

class QdrantToChromaMigrator:
    def __init__(self):
        """Initialize migration tools"""
        print("🔄 Initializing Migration Tools...")
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            persist_directory="./chroma_db",
            anonymized_telemetry=False
        ))
        
        # Initialize Qdrant client
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, check_compatibility=False)
        
        # Initialize embedding model
        self.model = SentenceTransformer('all-mpnet-base-v2')
        
        print("✅ Migration tools ready!")
    
    def migrate_collection(self, qdrant_collection="ingris_groundwater_collection", 
                          chroma_collection="groundwater_data", batch_size=1000):
        """Migrate data from Qdrant to ChromaDB"""
        print(f"🔄 Starting migration from {qdrant_collection} to {chroma_collection}")
        
        try:
            # Get or create ChromaDB collection
            try:
                chroma_col = self.chroma_client.get_collection(chroma_collection)
                print(f"✅ Found existing ChromaDB collection: {chroma_collection}")
            except:
                chroma_col = self.chroma_client.create_collection(
                    name=chroma_collection,
                    metadata={"description": "Migrated groundwater data from Qdrant"}
                )
                print(f"✅ Created new ChromaDB collection: {chroma_collection}")
            
            # Get total count from Qdrant
            print("🔄 Getting data count from Qdrant...")
            scroll_result = self.qdrant_client.scroll(
                collection_name=qdrant_collection,
                limit=1,
                with_payload=True
            )
            total_points = scroll_result[1]  # Total count
            print(f"📊 Total records in Qdrant: {total_points}")
            
            # Migrate in batches
            offset = None
            migrated_count = 0
            
            while True:
                print(f"🔄 Migrating batch starting from offset: {offset}")
                
                # Get batch from Qdrant
                scroll_result = self.qdrant_client.scroll(
                    collection_name=qdrant_collection,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True
                )
                
                points = scroll_result[0]
                if not points:
                    break
                
                # Prepare data for ChromaDB
                ids = [point.id for point in points]
                documents = [point.payload.get('text', '') for point in points]
                metadatas = [point.payload for point in points]
                
                # Generate embeddings
                print(f"🔄 Generating embeddings for {len(documents)} documents...")
                embeddings = self.model.encode(documents).tolist()
                
                # Add to ChromaDB
                chroma_col.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
                
                migrated_count += len(points)
                print(f"✅ Migrated {migrated_count}/{total_points} records")
                
                # Update offset for next batch
                offset = scroll_result[1]
                if offset is None:
                    break
                
                # Small delay to avoid overwhelming the system
                time.sleep(0.1)
            
            print(f"🎉 Migration completed! Migrated {migrated_count} records")
            
            # Verify migration
            chroma_count = chroma_col.count()
            print(f"📊 ChromaDB collection now has {chroma_count} records")
            
            return True
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return False
    
    def test_migrated_data(self, collection_name="groundwater_data"):
        """Test the migrated data"""
        print(f"🔍 Testing migrated data in {collection_name}")
        
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            # Test search
            results = collection.query(
                query_texts=["ground water estimation in karnataka"],
                n_results=5
            )
            
            print(f"✅ Found {len(results['documents'][0])} results")
            for i, doc in enumerate(results['documents'][0][:3]):
                print(f"  {i+1}. {doc[:100]}...")
                print(f"     State: {results['metadatas'][0][i].get('STATE', 'N/A')}")
                print(f"     District: {results['metadatas'][0][i].get('DISTRICT', 'N/A')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False

def main():
    """Main migration function"""
    print("🚀 Starting Qdrant to ChromaDB Migration")
    print("=" * 50)
    
    migrator = QdrantToChromaMigrator()
    
    # Migrate the data
    success = migrator.migrate_collection()
    
    if success:
        # Test the migrated data
        migrator.test_migrated_data()
        print("\n🎉 Migration and testing completed successfully!")
    else:
        print("\n❌ Migration failed!")

if __name__ == "__main__":
    main()
