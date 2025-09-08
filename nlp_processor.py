
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

def process_query(query):
    doc = nlp(query)

    # Extract named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Extract noun chunks as potential keywords
    keywords = [chunk.text for chunk in doc.noun_chunks]

    print(f"Original Query: {query}")
    print(f"Extracted Entities: {entities}")
    print(f"Extracted Keywords (Noun Chunks): {keywords}")
    return entities, keywords

if __name__ == "__main__":
    sample_query1 = "What is the groundwater availability in Andaman and Nicobar Islands for Fresh water?"
    process_query(sample_query1)

    sample_query2 = "Show me the rainfall data for Arunachal Pradesh."
    process_query(sample_query2)

    sample_query3 = "What is the stage of groundwater extraction in Tripura for Gomati district?"
    process_query(sample_query3)
